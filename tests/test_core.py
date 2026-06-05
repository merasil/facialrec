import configparser
import os
import tempfile
import unittest
from unittest.mock import patch

from app.benchmark import bench_one, bench_status
from app.cli import cli_parser
from app.config import cfg_get_bool, cfg_list, cfg_pick
from app.face import face_missing, face_result
from app.samples import sample_folder
from app.table import tab_render


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

    def test_face_log_level(self):
        self.assertEqual(os.environ["DEEPFACE_LOG_LEVEL"], "40")

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


if __name__ == "__main__":
    unittest.main()
