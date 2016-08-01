from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

setup(
    name='munkres',
    url='https://github.com/jfrelinger/cython-munkres-wrapper',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("munkres", ["src/munkres.pyx",
                                         "src/cpp/Munkres.cpp"],
                             include_dirs = [get_include(), 'src/cpp'],
                             language='c++', pyrex_gdb=True)], #, extra_compile_args=['-g'])],
    version = '1.0',
    description='Munkres implemented in c++ wrapped by cython',
    author='Jacob Frelinger',
    author_email='jacob.frelinger@duke.edu',
    requires=['numpy (>=1.3.0)', 'cython (>=0.15.1)']
)

