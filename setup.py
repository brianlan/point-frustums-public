import os
import glob
from setuptools import setup, find_packages
import numpy as np
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, ROCM_HOME


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = "point_frustums"


def get_extensions():
    extensions_dir = os.path.join(BASE_DIR, PROJECT, "ops", "csrc")
    extensions_main = os.path.join(extensions_dir, "ext.cpp")
    extensions_sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))

    is_rocm_pytorch = bool((torch.version.hip is not None) and (ROCM_HOME is not None))

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    extensions_sources_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )
    sources = [extensions_main] + extensions_sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += extensions_sources_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "point_frustums._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Build main project
setup(
    name=PROJECT,
    version="0.0.0",
    packages=find_packages(
        # All keyword arguments below are optional:
        where=f"{BASE_DIR}",  # '.' by default
        exclude=["tests"],  # empty by default
    ),
    package_dir={"": f"{BASE_DIR}"},
    entry_points={
        "console_scripts": [
            f"{PROJECT.replace('_', '-')} = {PROJECT}.__main__:main",
        ]
    },
    package_data={PROJECT: ["*.dll", "*.dylib", "*.so", "**/*.so"]},
    install_requires=[
        "pytorch-lightning[extra]",
        "torchmetrics",
        "nuscenes-devkit>=1.1.10",
        "numpy>=1.24.2",
        "torch>=2.1.0",
        "torchvision>=0.15.0",
        "tensorboard>=2.12.2",
        "torch_tb_profiler",
        "xformers==0.0.23",
        "numpy-quaternion",
        "pyquaternion",
        "loguru",
        "prettytable",
        "PyYaml",
    ],
    extras_require={"dev": ["pytest", "pylint", "pre-commit", "black", "pycln"], "test": ["pytest"]},
    ext_modules=get_extensions(),
    include_dirs=[np.get_include()],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
