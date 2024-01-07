# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pytorch_lightning.cli import LightningCLI

from point_frustums.cli.lightning_cli_arg_helpers import DatasetConfig


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(DatasetConfig, "dataset")
        parser.link_arguments("dataset", "model.init_args.dataset")
        parser.link_arguments("dataset", "data.init_args.dataset")


def main():
    CLI(parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    sys.exit(main())
