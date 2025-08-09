import os
import glob
import re
import subprocess
import sys

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.9"

def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
    
    tensor_array_lib_path = os.environ['TENSOR_ARRAY_INSTALL_PATH']

    ext_modules = [
        Pybind11Extension(
            "tensor_array.tensor2",
            sources = glob.glob(os.path.join("cpp", "*.cc")),
            include_dirs=[tensor_array_lib_path + "/include"],
            library_dirs=[tensor_array_lib_path + "/lib/tensor-array", tensor_array_lib_path + "/lib64/tensor-array"],
            libraries=["tensorarray_core", "tensorarray_layers"],
            define_macros=[("VERSION_INFO", __version__)],
            ),
    ]

    setup(
        name = "TensorArray",
        version = __version__,
        description = "A machine learning package",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        author = "TensorArray-Creators",
        url = "https://github.com/Tensor-Array/Tensor-Array-Python",
        classifiers = [
            "Development Status :: 2 - Pre-Alpha",

            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",

            "License :: OSI Approved :: MIT License",

            "Environment :: GPU :: NVIDIA CUDA :: 12",
        ],
        packages = find_packages("src"),
        package_dir = {
            "": "src",
        },
        ext_modules = ext_modules,
        cmdclass = {
            "build_ext": build_ext,
        },
        license = "MIT",
        python_requires = ">=3.8",
    )

if __name__ == "__main__":
    main()
