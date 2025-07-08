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

cd ../..

# Install dependencies for building Tensor-Array on manylinux
echo "Installing dependencies for building Tensor-Array on manylinux..."
chmod +x tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh
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
