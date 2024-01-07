import os
from typing import Optional

import yaml
from loguru import logger


def data_root_getter(dataloader, version, root_dir, data_root: Optional[str] = None) -> str:
    filepath = os.path.abspath(f"{root_dir}/../.data-root.yaml")

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="UTF-8") as data_root_file:
            data_roots = yaml.load(data_root_file, Loader=yaml.FullLoader)
        if data_roots is None:
            data_roots = {}
    else:
        data_roots = {}

    if data_root is not None:
        if dataloader not in data_roots:
            data_roots[dataloader] = {}
        data_roots[dataloader][version] = os.path.abspath(data_root)

    try:
        data_root = data_roots[dataloader][version]
    except KeyError as err:
        data_root = input(f"Please enter the absolute path to the local dataset '{dataloader}/{version}': ")
        data_root = os.path.abspath(data_root)
        if not os.path.exists(data_root):
            raise ValueError("The specified directory does not exist.") from err

        if dataloader in data_roots:
            data_roots[dataloader].update({version: data_root})
        else:
            data_roots[dataloader] = {version: data_root}

    with open(filepath, "w", encoding="UTF-8") as data_root_file:
        yaml.dump(data_roots, data_root_file)

    logger.debug(f"Local data root: {data_root}")
    return data_root
