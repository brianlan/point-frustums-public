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
    parser.add_argument("--filename", type=str, default="./nuscenes_source/urls_complete.txt")
    parser.add_argument(
        "--out_dir", type=str, default=os.path.join(os.path.expandvars("$HOME"), "datasets", "nuscenes")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    downloads_local = os.path.join(args.out_dir, ".cache")
    os.makedirs(downloads_local, exist_ok=True)
    os.makedirs(os.path.join(downloads_local, "logs"), exist_ok=True)
    pids = []
    with open(args.filename, "r", encoding="utf-8") as fp:
        for url in fp.readlines():
            url = url.strip()
            if url == "" or url.startswith("#"):
                continue
            blob_name = url.rsplit("/")[-1]
            archive_local = os.path.join(downloads_local, blob_name)
            blob_base_name, blob_extension = os.path.splitext(blob_name)
            assert blob_extension in (".tgz", ".zip"), f"Cannot unpack '{blob_extension}' files."
            logfile = os.path.join(downloads_local, "logs", blob_base_name + ".log")
            cmd_download = f"wget -nc -nv -O {archive_local} '{url}'"
            if blob_extension == ".tgz":
                cmd_unpack = f"tar zxf '{archive_local}' --skip-old-files --directory {args.out_dir}"
            elif blob_extension == ".zip":
                cmd_unpack = f"unzip -n '{archive_local}' -d {args.out_dir}"
            # I'd use && to chain the commands but wget returns a non-zero exit code when skipping an existing file
            cmd = cmd_download + "; " + cmd_unpack + ">" + logfile + "&"
            proc = Popen(cmd, shell=True)  # pylint: disable=consider-using-with
            pids.append(proc.pid)
            logger.info(f"Spawned a subprocess with pid={proc.pid} to download and extract '{blob_name}'")
        logger.info(f"Preparing data in {args.out_dir} in the background with the process {pids=}.")


if __name__ == "__main__":
    sys.exit(main())
