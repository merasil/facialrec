import json
import sys
from typing import Any

from app.benchmark import bench_one
from app.media import med_first


def bench_stage(bench_name: str) -> None:
    print(f"BENCH_STAGE={bench_name}", flush=True)


def bench_run(bench_data: dict[str, Any]) -> list[str]:
    bench_stage("input loading")
    bench_first = med_first(bench_data["input"])
    return bench_one(
        bench_data["input"],
        bench_data["name"],
        bench_data["db"],
        bench_data["detector"],
        bench_data["recognizer"],
        bench_data["metric"],
        bool(bench_data["align"]),
        bool(bench_data["enforce"]),
        bench_first,
        bench_stage,
    )


def bench_main() -> int:
    if len(sys.argv) != 2:
        print("Expected one JSON payload", file=sys.stderr)
        return 2
    try:
        bench_data = json.loads(sys.argv[1])
        bench_row = bench_run(bench_data)
        bench_result = {"ok": True, "row": bench_row}
    except Exception as bench_err:
        bench_result = {
            "ok": False,
            "error": " ".join(str(bench_err).split()),
            "type": type(bench_err).__name__,
        }
    print("BENCH_JSON=" + json.dumps(bench_result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(bench_main())
