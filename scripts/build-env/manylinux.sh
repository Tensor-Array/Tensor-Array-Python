#!/bin/bash

# Decide if we can proceed or not (root or sudo is required) and if so store whether sudo should be used or not. 
if [ "$is_root" = false ] && [ "$has_sudo" = false ]; then 
    echo "Root or sudo is required. Aborting."
    exit 1
elif [ "$is_root" = false ] ; then
    USE_SUDO=sudo
else
    USE_SUDO=
fi

cd ${PWD}

# Install dependencies for building Tensor-Array on manylinux
echo "Installing dependencies for building Tensor-Array on manylinux..."
chmod +x /tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh
echo "Installing required packages..."
$USE_SUDO yum install -y redhat-lsb-core wget
echo "Running CUDA installation script..."
tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh

# debugging output
echo
echo "------------------------------"
echo
echo "CUDA_PATH="
echo "$CUDA_PATH"
echo
echo "PATH="
echo "$PATH"
echo
echo "LD_LIBRARY_PATH="
echo "$LD_LIBRARY_PATH"
echo
echo "------------------------------"
echo

# Check if nvcc is available
echo "Checking for nvcc..."
if ! command -v nvcc &> /dev/null; then
    echo "nvcc could not be found. Please ensure CUDA is installed correctly."
    exit 1
fi
echo "nvcc is available. Proceeding with the build environment setup."

cd tensor-array-repo/Tensor-Array

pip install "cmake>=3.18,<3.29"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
else
    echo "Build directory already exists."
fi
# Change to the build directory
cd build
# Configure the build with CMake
echo "Configuring the build with CMake..."
cmake ..
cmake --build .
cmake --install .

cd ..
rm -rf build

cd ../..

