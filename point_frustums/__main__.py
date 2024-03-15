# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from pytorch_lightning.cli import LightningCLI


def main():
    LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    sys.exit(main())
