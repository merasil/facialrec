#!/usr/bin/env python3

from time import time

from app.cli import cli_run


if __name__ == "__main__":
    raise SystemExit(cli_run(int(time())))
