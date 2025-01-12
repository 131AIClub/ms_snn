import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import pathlib

here = pathlib.Path(__file__).parent.resolve()
CMAKE_FOLDER = os.path.join(here, "sorrogate", "cuda")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir],
                              cwd=self.build_temp, env=os.environ)
        subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)
        subprocess.check_call(
            ['mv', './libms_snn.so', ext.sourcedir], cwd=self.build_temp)
        print("move file to "+ext.sourcedir)


setup(
    name="ms_snn",
    version="0.1",
    install_requires=["mindspore-dev"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires='>=3.9, <4.0',
    # CUDA 编译会在 build_ext 中处理
    ext_modules=[CMakeExtension('ms_snn', 'ms_snn/surrogate/cuda/')],
    cmdclass={
        "build_ext": CMakeBuild,  # 使用自定义的 CMake 构建
    },
    package_data={
        "ms_snn": ["surrogate/cuda/libms_snn.so"],
    }
)
