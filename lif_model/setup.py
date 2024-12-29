import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import ctypes.util
import subprocess

# 动态获取CUDA路径
cuda_include_dir = os.path.join(ctypes.util.find_library('cuda'), 'include')
cuda_library_dir = os.path.join(ctypes.util.find_library('cuda'), 'lib64')

# 如果还是找不到路径，可以考虑给出默认值，或者抛出错误
if not cuda_include_dir or not cuda_library_dir:
    raise RuntimeError(
        'Could not find CUDA paths. Please set CUDA_INCLUDE_DIR and CUDA_LIBRARY_DIR environment variables.')


class build_ext_custom(build_ext):
    def run(self):
        subprocess.check_call(['cmake', './cuda'])
        # 调用父类的构建方法
        super().run()


setup(
    name='continal_learning_ms',
    ext_modules=[
        Extension(
            name='ATan',
            sources=[
                'ATan.cu',  # 你的CUDA源文件路径
            ],
            include_dirs=[
                cuda_include_dir,       # CUDA头文件路径
            ],
            library_dirs=[
                cuda_library_dir,           # CUDA库路径
            ],
            libraries=[
                'cuda',                  # 连接CUDA库
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': ['-O3', '--shared', '-Xcompiler', '-fPIC', '-O3', '-gencode arch=compute_86,code=sm_86', '--use_fast_math', '--expt-relaxed-constexpr'
                         '-D_GLIBCXX_USE_CXX11_ABI=0'],  # CUDA编译优化参数
            }
        )
    ],
    cmdclass={
        'build_ext': build_ext_custom
    },
    install_requires=[
        'mindspore-dev',
    ]
)
