"""
Setup script for compiling deformable aggregation CUDA extension.
Pure PyTorch CUDA extension without mmcv dependencies.

To compile and install:
    cd reimplementation/ops
    python setup.py install

Or for development:
    python setup.py develop
"""
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get CUDA architecture flags
def get_cuda_arch_flags():
    """Get CUDA architecture flags for current GPU."""
    try:
        # Get GPU compute capability
        capability = torch.cuda.get_device_capability()
        arch_flags = [f'-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}']
        return arch_flags
    except:
        # Default to common architectures if detection fails
        return [
            '-gencode=arch=compute_70,code=sm_70',  # V100
            '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 2080
            '-gencode=arch=compute_80,code=sm_80',  # A100
            '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
            '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
            '-gencode=arch=compute_80,code=sm_90',  # A100
            
        ]

setup(
    name='deformable_aggregation',
    version='1.0.0',
    description='Deformable aggregation CUDA extension for SparseDrive',
    author='SparseDrive Team',
    ext_modules=[
        CUDAExtension(
            name='deformable_aggregation_ext',
            sources=[
                'src/deformable_aggregation.cpp',
                'src/deformable_aggregation_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-DCUDA_HAS_FP16=1',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ] + get_cuda_arch_flags(),
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
