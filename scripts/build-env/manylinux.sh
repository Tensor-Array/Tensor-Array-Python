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
chmod +x tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh
$USE_SUDO dnf -y install redhat-lsb-core wget
tensor-array-repo/Tensor-Array/scripts/actions/install-cuda-rhel.sh

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "nvcc could not be found. Please ensure CUDA is installed correctly."
    exit 1
fi
