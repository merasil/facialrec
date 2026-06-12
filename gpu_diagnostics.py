import faulthandler
import json
import os
import signal
import subprocess
import sys


DEFAULT_COMMAND_TIMEOUT = 300
PROCESS_STOP_TIMEOUT = 10


def log(message=""):
    print(message, flush=True)


def interrupt_handler(_signal_number, _frame):
    raise KeyboardInterrupt


def stop_process(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        log(
            f"Process {process.pid} did not exit after SIGKILL. "
            "It may be blocked inside the NVIDIA kernel driver."
        )


def run_command(command, timeout=DEFAULT_COMMAND_TIMEOUT):
    log(f"\n$ {' '.join(command)}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
        )
    except FileNotFoundError:
        log(f"not found: {command[0]}")
        return 127

    try:
        return_code = process.wait(timeout=timeout)
    except KeyboardInterrupt:
        log(f"\nInterrupted; stopping process {process.pid} ...")
        stop_process(process)
        raise
    except subprocess.TimeoutExpired:
        log(f"\nTIMEOUT after {timeout} seconds; stopping process {process.pid} ...")
        stop_process(process)
        return 124

    log(f"exit code: {return_code}")
    return return_code


def tensorflow_check():
    log("TensorFlow: importing")
    import tensorflow as tf

    log(f"TensorFlow: {tf.__version__}")
    log(f"TensorFlow build: {json.dumps(tf.sysconfig.get_build_info(), default=str)}")
    log("TensorFlow: discovering GPUs")
    gpus = tf.config.list_physical_devices("GPU")
    log(f"TensorFlow GPUs: {gpus}")
    if not gpus:
        raise RuntimeError("TensorFlow cannot see a GPU")

    log("TensorFlow: executing GPU operation")
    with tf.device("/GPU:0"):
        value = tf.reduce_sum(tf.random.normal([1024, 1024]))
    log(f"TensorFlow GPU operation: {float(value):.6f}")
    log("TensorFlow: check complete")


def torch_check():
    log("PyTorch: importing")
    import torch

    log(f"PyTorch: {torch.__version__}")
    log(f"PyTorch CUDA runtime: {torch.version.cuda}")
    log("PyTorch: initializing CUDA")
    cuda_available = torch.cuda.is_available()
    log(f"PyTorch CUDA available: {cuda_available}")
    log(f"PyTorch compiled architectures: {torch.cuda.get_arch_list()}")
    if not cuda_available:
        raise RuntimeError("PyTorch cannot see a GPU")

    log(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch GPU capability: {torch.cuda.get_device_capability(0)}")
    log("PyTorch: allocating CUDA tensors")
    with torch.inference_mode():
        left = torch.rand(1024, 1024, device="cuda")
        right = torch.rand(1024, 1024, device="cuda")
        log("PyTorch: executing matrix multiplication")
        value = (left @ right).sum()
        log("PyTorch: synchronizing CUDA")
        torch.cuda.synchronize()
        result = value.item()

    log(f"PyTorch GPU operation: {result:.6f}")
    log("PyTorch: check complete")


def framework_check(option):
    if option == "--tensorflow":
        tensorflow_check()
    elif option == "--torch":
        torch_check()
    else:
        raise ValueError(f"Unknown diagnostic option: {option}")


def main():
    if len(sys.argv) == 2:
        traceback_after = int(
            os.environ.get("GPU_DIAGNOSTICS_TRACEBACK_AFTER", "120")
        )
        faulthandler.enable()
        faulthandler.dump_traceback_later(traceback_after, repeat=True)
        try:
            framework_check(sys.argv[1])
        except Exception as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        finally:
            faulthandler.cancel_dump_traceback_later()
        return 0

    command_timeout = int(
        os.environ.get("GPU_DIAGNOSTICS_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))
    )

    log(f"Python: {sys.version.split()[0]}")
    log(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    log(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', '')}")
    log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    run_command(["nvidia-smi"])

    failures = []
    checks = (("TensorFlow", "--tensorflow"), ("PyTorch", "--torch"))
    for name, option in checks:
        log(f"\n--- {name} check ---")
        result = run_command(
            [sys.executable, "-u", __file__, option],
            timeout=command_timeout,
        )
        if result != 0:
            failures.append(f"{name} check exited with status {result}")
        if result == 124:
            log("Skipping remaining checks after a timeout.")
            break

    if failures:
        print("\nGPU diagnostics failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    log("\nGPU diagnostics passed.")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt_handler)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\nGPU diagnostics interrupted.")
        raise SystemExit(130)
