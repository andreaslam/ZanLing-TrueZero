from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("aieval6lite.pyx")
)

# command
# python3 pytocpy.py build_ext --inplace
# compile to .exe https://stackoverflow.com/questions/2581784/can-cython-compile-to-an-exe 