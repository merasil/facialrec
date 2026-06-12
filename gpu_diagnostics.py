import json
import os
import subprocess
import sys


def run_command(command):
    print(f"\n$ {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(f"not found: {command[0]}")
        return 127

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    print(f"exit code: {result.returncode}")
    return result.returncode


def tensorflow_check():
    import tensorflow as tf

    print(f"TensorFlow: {tf.__version__}")
    print(f"TensorFlow build: {json.dumps(tf.sysconfig.get_build_info(), default=str)}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}")
    if not gpus:
        raise RuntimeError("TensorFlow cannot see a GPU")

    with tf.device("/GPU:0"):
        value = tf.reduce_sum(tf.random.normal([1024, 1024]))
    print(f"TensorFlow GPU operation: {float(value):.6f}")


def torch_check():
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch compiled architectures: {torch.cuda.get_arch_list()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a GPU")

    print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch GPU capability: {torch.cuda.get_device_capability(0)}")
    value = (torch.rand(1024, 1024, device="cuda") @
             torch.rand(1024, 1024, device="cuda")).sum()
    print(f"PyTorch GPU operation: {value.item():.6f}")


def framework_check(option):
    if option == "--tensorflow":
        tensorflow_check()
    elif option == "--torch":
        torch_check()
    else:
        raise ValueError(f"Unknown diagnostic option: {option}")


def main():
    if len(sys.argv) == 2:
        try:
            framework_check(sys.argv[1])
        except Exception as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        return 0

    print(f"Python: {sys.version.split()[0]}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
    print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', '')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    run_command(["nvidia-smi"])

    failures = []
    checks = (("TensorFlow", "--tensorflow"), ("PyTorch", "--torch"))
    for name, option in checks:
        print(f"\n--- {name} check ---")
        result = run_command([sys.executable, __file__, option])
        if result != 0:
            failures.append(f"{name} check exited with status {result}")

    if failures:
        print("\nGPU diagnostics failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nGPU diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
