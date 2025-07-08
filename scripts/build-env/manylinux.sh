#!/bin/bash
cd ../..

# Install dependencies for building Tensor-Array on manylinux
chmod +x tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh
dnf -y install redhat-lsb-core wget
tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "nvcc could not be found. Please ensure CUDA is installed correctly."
    exit 1
fi
