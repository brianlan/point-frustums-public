# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from subprocess import Popen

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Prepare NuScenes Dataset", description="Download and extract all archives specified in the provided file."
    )
    parser.add_argument("--filename", type=str, default="./urls.txt")
    parser.add_argument(
        "--out_dir", type=str, default=os.path.join(os.path.expandvars("$HOME"), "datasets", "nuscenes")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    downloads_local = os.path.join(args.out_dir, ".cache")
    os.makedirs(downloads_local, exist_ok=True)
    os.makedirs(os.path.join(downloads_local, "logs"), exist_ok=True)
    with open(args.filename, "r", encoding="utf-8") as fp:
        for url in fp.readlines():
            url = url.strip()
            blob_name = url.rsplit("/")[-1]
            archive_local = os.path.join(downloads_local, blob_name)
            logfile = os.path.join(downloads_local, "logs", blob_name.rstrip(".tgz") + ".log")
            cmd_download = f"wget -q -O {archive_local} '{url}'"
            cmd_unpack = f"tar zxvf '{archive_local}' --skip-old-files --directory {args.out_dir}"
            cmd = cmd_download + "|" + cmd_unpack + ">" + logfile
            proc = Popen(cmd, shell=True)  # pylint: disable=consider-using-with
            logger.info(f"Spawned a subprocess with pid={proc.pid} to download and extract '{blob_name}'")


if __name__ == "__main__":
    sys.exit(main())
