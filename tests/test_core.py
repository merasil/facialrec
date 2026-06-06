import configparser
import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from app.benchmark import bench_one, bench_spawn, bench_status, bench_warm
from app.benchmark_worker import bench_diagnostics
from app.cli import cli_parser
from app.config import cfg_get_bool, cfg_list, cfg_pick
from app.face import (
    FaceError,
    face_load,
    face_load_detector,
    face_missing,
    face_prepare_detector,
    face_result,
    face_tf,
    face_torch_detector,
)
from app.samples import sample_folder
from app.table import tab_render
from app.vram import vram_worker
from app.vram_worker import vram_run


class TestFrame:
    def __init__(self, test_paths):
        self.empty = not test_paths
        self.test_paths = test_paths

    def __getitem__(self, test_key):
        if test_key != "identity":
            raise KeyError(test_key)
        return TestSeries(self.test_paths)


class TestSeries:
    def __init__(self, test_values):
        self.test_values = test_values

    def tolist(self):
        return self.test_values


class CoreTests(unittest.TestCase):
    def test_cli_live(self):
        test_args = cli_parser().parse_args([])
        self.assertIsNone(test_args.cli_mode)

    def test_cli_before(self):
        test_args = cli_parser().parse_args(
            [
                "--config",
                "custom.ini",
                "benchmark",
                "--input",
                "frames",
                "--name",
                "Alice",
            ]
        )
        self.assertEqual(test_args.cfg_path, "custom.ini")

    def test_cli_after(self):
        test_args = cli_parser().parse_args(
            [
                "benchmark",
                "--config",
                "custom.ini",
                "--input",
                "frames",
                "--name",
                "Alice",
            ]
        )
        self.assertEqual(test_args.cfg_path, "custom.ini")

    def test_config_helpers(self):
        test_cfg = configparser.ConfigParser()
        test_cfg.read_dict({"test": {"enabled": "yes"}})
        self.assertTrue(cfg_get_bool(test_cfg, "test", "enabled"))
        self.assertEqual(cfg_pick(None, "config"), "config")
        self.assertEqual(cfg_pick("cli", "config"), "cli")
        self.assertEqual(cfg_list(None, "a, b"), ["a", "b"])

    def test_face_log(self):
        self.assertEqual(os.environ["DEEPFACE_LOG_LEVEL"], "40")
        self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], "2")

    def test_face_result(self):
        test_frames = [
            TestFrame([]),
            TestFrame(["db/Bob/Bob.jpg", "db/Alice/Alice.jpg"]),
        ]
        self.assertEqual(face_result(test_frames, "Alice"), (True, True, True))
        self.assertEqual(face_result(test_frames, "alice"), (True, True, False))

    def test_face_missing(self):
        test_err = UnboundLocalError(
            "cannot access local variable 'boxes_np' where it is not associated"
        )
        self.assertTrue(face_missing(test_err))
        self.assertFalse(face_missing(RuntimeError("CUDA out of memory")))

    @patch("app.benchmark.bench_warm")
    @patch("app.benchmark.med_iter")
    @patch("app.benchmark.face_find")
    def test_benchmark(self, test_find, test_iter, test_warm):
        test_iter.return_value = iter(
            [(1, "one"), (2, "two"), (3, "three"), (4, "four")]
        )
        test_find.side_effect = [
            [TestFrame(["db/Alice/Alice.jpg"])],
            [TestFrame(["db/Bob/Bob.jpg"])],
            [TestFrame([])],
            ValueError("Face could not be detected in input"),
        ]
        test_row = bench_one(
            "input",
            "Alice",
            "db",
            "retinaface",
            "Facenet512",
            "cosine",
            False,
            True,
            "first",
        )
        self.assertEqual(test_row[3:11], ["4", "3", "75.00", "1", "33.33", "1", "1", "0"])
        self.assertEqual(test_row[-1], "ok")
        test_warm.assert_called_once()

    def test_bench_status(self):
        test_err = ImportError("No module named 'facenet_pytorch'")
        self.assertEqual(
            bench_status(test_err, "fastmtcnn"),
            "missing facenet-pytorch",
        )
        self.assertEqual(
            bench_status(
                ImportError("Please install using 'pip install facenet-pytorch'"),
                "fastmtcnn",
            ),
            "missing facenet-pytorch",
        )
        self.assertEqual(
            bench_status(ValueError("invalid model_name"), "yolov8"),
            "invalid detector; use yolov8n, yolov8m or yolov8l",
        )

    @patch("app.benchmark.subprocess.run")
    def test_bench_spawn(self, test_run):
        test_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='BENCH_JSON={"ok": true, "row": ["retinaface", "ok"]}\n',
            stderr="",
        )
        test_row, test_status = bench_spawn({"detector": "retinaface"})
        self.assertEqual(test_row, ["retinaface", "ok"])
        self.assertEqual(test_status, "ok")
        test_cmd = test_run.call_args.args[0]
        self.assertEqual(test_cmd[1:4], ["-X", "faulthandler", "-u"])
        self.assertEqual(test_run.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertNotIn("stderr", test_run.call_args.kwargs)

    @patch("app.benchmark.subprocess.run")
    def test_bench_sigkill(self, test_run):
        test_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=-9,
            stdout="BENCH_STAGE=detector loading\n",
            stderr="",
        )

        test_row, test_status = bench_spawn({"detector": "yolov8m"})

        self.assertIsNone(test_row)
        self.assertEqual(
            test_status,
            "worker killed by SIGKILL (possible RAM/VRAM OOM) "
            "during detector loading",
        )

    @patch("app.benchmark.subprocess.run")
    def test_bench_sigsegv(self, test_run):
        test_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=-11,
            stdout="BENCH_STAGE=detector runtime import\n",
            stderr="",
        )

        test_row, test_status = bench_spawn({"detector": "yolov8m"})

        self.assertIsNone(test_row)
        self.assertEqual(
            test_status,
            "worker killed by SIGSEGV during detector runtime import",
        )

    @patch("app.benchmark.face_find")
    @patch("app.benchmark.face_load_recognizer")
    @patch("app.benchmark.face_tf")
    @patch("app.benchmark.face_load_detector")
    @patch("app.benchmark.face_prepare_detector")
    def test_bench_pytorch_order(
        self,
        test_prepare,
        test_detector,
        test_tf,
        test_recognizer,
        test_find,
    ):
        test_steps = []
        test_prepare.side_effect = lambda *test_args: test_steps.append("prepare")
        test_detector.side_effect = lambda *test_args: test_steps.append("detector")
        test_tf.side_effect = lambda *test_args: test_steps.append("tensorflow")
        test_recognizer.side_effect = (
            lambda *test_args: test_steps.append("recognizer")
        )

        bench_warm(
            "frame",
            "db",
            "yolov8m",
            "Facenet512",
            "euclidean_l2",
            False,
            True,
        )

        self.assertEqual(
            test_steps,
            ["prepare", "detector", "tensorflow", "recognizer"],
        )
        test_find.assert_called_once()

    @patch("app.benchmark.face_find")
    @patch("app.benchmark.face_load_recognizer")
    @patch("app.benchmark.face_tf")
    @patch("app.benchmark.face_load_detector")
    @patch("app.benchmark.face_prepare_detector")
    def test_bench_tensorflow_order(
        self,
        test_prepare,
        test_detector,
        test_tf,
        test_recognizer,
        test_find,
    ):
        test_steps = []
        test_prepare.side_effect = lambda *test_args: test_steps.append("prepare")
        test_detector.side_effect = lambda *test_args: test_steps.append("detector")
        test_tf.side_effect = lambda *test_args: test_steps.append("tensorflow")
        test_recognizer.side_effect = (
            lambda *test_args: test_steps.append("recognizer")
        )

        bench_warm(
            "frame",
            "db",
            "retinaface",
            "Facenet512",
            "euclidean_l2",
            False,
            True,
        )

        self.assertEqual(test_steps, ["tensorflow", "detector", "recognizer"])
        test_prepare.assert_not_called()
        test_find.assert_called_once()

    def test_torch_detectors(self):
        self.assertTrue(face_torch_detector("yolov8m"))
        self.assertTrue(face_torch_detector("YOLOv11n"))
        self.assertTrue(face_torch_detector("fastmtcnn"))
        self.assertFalse(face_torch_detector("retinaface"))

    @patch("app.face.importlib.import_module")
    def test_prepare_detector(self, test_import):
        test_import.return_value = SimpleNamespace(YOLO=object(), MTCNN=object())

        face_prepare_detector("yolov8m")
        test_import.assert_called_once_with("ultralytics")

        test_import.reset_mock()
        face_prepare_detector("fastmtcnn")
        test_import.assert_called_once_with("facenet_pytorch")

        test_import.reset_mock()
        face_prepare_detector("retinaface")
        test_import.assert_not_called()

    def test_yolo_runtime_before_deepface(self):
        test_steps = []

        class TestUltralytics:
            @property
            def YOLO(self):
                test_steps.append("ultralytics.YOLO")
                return object()

        with patch("app.face.face_api") as test_api:
            with patch("app.face.importlib.import_module") as test_import:
                test_import.side_effect = lambda test_name: (
                    test_steps.append(test_name) or TestUltralytics()
                )
                test_api.side_effect = lambda: (
                    test_steps.append("deepface")
                    or SimpleNamespace(build_model=lambda **test_args: None)
                )

                face_load_detector("yolov8m")

        self.assertEqual(
            test_steps,
            ["ultralytics", "ultralytics.YOLO", "deepface"],
        )

    @patch("app.face.face_load_recognizer")
    @patch("app.face.face_tf")
    @patch("app.face.face_load_detector")
    def test_face_load_order(self, test_detector, test_tf, test_recognizer):
        test_steps = []
        test_detector.side_effect = lambda *test_args: test_steps.append("detector")
        test_tf.side_effect = lambda *test_args: test_steps.append("tensorflow")
        test_recognizer.side_effect = (
            lambda *test_args: test_steps.append("recognizer")
        )

        face_load("yolov8m", "Facenet512")
        self.assertEqual(test_steps, ["detector", "tensorflow", "recognizer"])

        test_steps.clear()
        face_load("retinaface", "Facenet512")
        self.assertEqual(test_steps, ["tensorflow", "detector", "recognizer"])

    def test_face_tf_already_configured(self):
        test_gpu = object()
        test_experimental = SimpleNamespace(
            set_memory_growth=lambda *test_args: (_ for _ in ()).throw(
                RuntimeError("Physical devices cannot be modified")
            ),
            get_memory_growth=lambda test_device: True,
        )
        test_config = SimpleNamespace(
            list_physical_devices=lambda test_type: [test_gpu],
            set_visible_devices=lambda *test_args: None,
            get_visible_devices=lambda test_type: [test_gpu],
            experimental=test_experimental,
        )
        test_tf = SimpleNamespace(config=test_config)

        with patch.dict(sys.modules, {"tensorflow": test_tf}):
            self.assertIs(face_tf(), test_tf)

    def test_face_tf_rejects_wrong_configuration(self):
        test_gpu = object()
        test_other_gpu = object()
        test_experimental = SimpleNamespace(
            set_memory_growth=lambda *test_args: (_ for _ in ()).throw(
                RuntimeError("Physical devices cannot be modified")
            ),
            get_memory_growth=lambda test_device: True,
        )
        test_config = SimpleNamespace(
            list_physical_devices=lambda test_type: [test_gpu, test_other_gpu],
            set_visible_devices=lambda *test_args: None,
            get_visible_devices=lambda test_type: [test_other_gpu],
            experimental=test_experimental,
        )
        test_tf = SimpleNamespace(config=test_config)

        with patch.dict(sys.modules, {"tensorflow": test_tf}):
            with self.assertRaises(FaceError):
                face_tf()

    @patch("app.benchmark_worker.importlib.metadata.version")
    def test_bench_diagnostics(self, test_version):
        test_version.side_effect = lambda test_name: f"{test_name}-version"
        test_stderr = StringIO()

        with patch("sys.stderr", test_stderr):
            bench_diagnostics()

        test_line = test_stderr.getvalue().strip()
        self.assertTrue(test_line.startswith("BENCH_DIAG="))
        test_data = json.loads(test_line.removeprefix("BENCH_DIAG="))
        self.assertEqual(test_data["packages"]["deepface"], "deepface-version")
        self.assertEqual(test_data["packages"]["torch"], "torch-version")
        self.assertEqual(
            test_data["packages"]["nvidia-cudnn-cu12"],
            "nvidia-cudnn-cu12-version",
        )
        self.assertIn("python", test_data)
        self.assertIn("platform", test_data)

    def test_sample_folder(self):
        with tempfile.TemporaryDirectory() as test_root:
            test_first = sample_folder(test_root, 123)
            test_next = sample_folder(test_root, 123)
            self.assertEqual(test_first.name, "123")
            self.assertEqual(test_next.name, "123-01")

    def test_table(self):
        test_table = tab_render(["A", "B"], [["x", 12]])
        self.assertIn("| A | B  |", test_table)
        self.assertIn("| x | 12 |", test_table)

    @patch("app.vram_worker.med_first", return_value="frame")
    @patch("app.vram_worker.face_load_recognizer")
    @patch("app.vram_worker.face_load_detector")
    @patch("app.vram_worker.face_find")
    @patch("app.vram_worker.face_tf")
    @patch("app.vram_worker.face_prepare_detector")
    @patch("app.vram_worker.vram_nvml", return_value=(None, None))
    def test_vram_warm(
        self,
        test_nvml,
        test_prepare,
        test_tf,
        test_find,
        test_detector,
        test_recognizer,
        test_first,
    ):
        test_order = []
        test_exp = SimpleNamespace(
            get_memory_info=lambda test_name: {"current": 100, "peak": 200},
            reset_memory_stats=lambda test_name: None,
        )
        test_cfg = SimpleNamespace(
            experimental=test_exp,
            list_logical_devices=lambda test_name: [object()],
        )
        test_prepare.side_effect = lambda *test_args: test_order.append("prepare")
        test_tf.side_effect = lambda *test_args: (
            test_order.append("tensorflow")
            or SimpleNamespace(config=test_cfg)
        )
        test_detector.side_effect = lambda *test_args: test_order.append("detector")
        test_recognizer.side_effect = (
            lambda *test_args: test_order.append("recognizer")
        )
        test_data = {
            "gpu": 0,
            "input": "input",
            "db": "db",
            "detector": "yolov8m",
            "recognizer": "Facenet512",
            "metric": "euclidean_l2",
            "align": False,
            "enforce": True,
            "runs": 3,
        }

        vram_run(test_data)

        self.assertEqual(
            test_order,
            ["prepare", "detector", "tensorflow", "recognizer"],
        )
        self.assertEqual(test_find.call_count, 4)
        self.assertTrue(test_find.call_args_list[0].args[-1])
        for test_call in test_find.call_args_list[1:]:
            self.assertFalse(test_call.args[-1])

    @patch("app.vram.subprocess.run")
    def test_vram_error(self, test_run):
        test_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                'VRAM_JSON={"error": "CUDA out of memory", '
                '"ok": false, "type": "RuntimeError"}\n'
            ),
            stderr="I0000 gpu_device.cc:2043] Created device GPU:0\n",
        )

        test_result, test_status = vram_worker({"detector": "yolov8m"})

        self.assertIsNone(test_result)
        self.assertEqual(test_status, "CUDA out of memory")

    @patch("app.vram.subprocess.run")
    def test_vram_exit(self, test_run):
        test_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=-9,
            stdout="VRAM_STAGE=model loading\n",
            stderr="I0000 gpu_device.cc:2043] Created device GPU:0\n",
        )

        test_result, test_status = vram_worker({"detector": "yolov8m"})

        self.assertIsNone(test_result)
        self.assertEqual(test_status, "worker exit -9 during model loading")


if __name__ == "__main__":
    unittest.main()
