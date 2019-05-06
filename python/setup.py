   #----------------------------------------------------------------------#
   #  distutils setup script for compiling cut-pursuit python extensions  #
   #----------------------------------------------------------------------#
""" 
Compilation command: python setup.py build_ext

Camille Baudoin 2019
"""

from distutils.core import setup, Extension
from distutils.command.build import build
import numpy
import shutil # for rmtree, os.rmdir can only remove _empty_ directory
import os 
import re

###  targets and compile options  ###
to_compile = [ # comment undesired extension modules
    "pfdr_d1_ql1b_cpy",
    "pfdr_d1_lsx_cpy",
]
include_dirs = [numpy.get_include()] # find the Numpy headers
# compilation and linkage options
extra_compile_args = ["-Wextra", "-Wpedantic", "-std=c++11", "-fopenmp", "-g0"]
extra_link_args = ["-lgomp"]

###  auxiliary functions  ###
class MyBuild(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin" 
    def run(self):
        build_path = self.build_lib

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


###  preprocessing  ###
# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.dirname(__file__)))

try:
    os.mkdir("bin")
except FileExistsError:
    pass 

# remove previously compiled lib
for shared_obj in to_compile: 
    purge("bin/", shared_obj) 

###  compilation  ###

name = "pfdr_d1_ql1b_cpy"
if name in to_compile:
    mod = Extension(
            name,
            # list source files
            ["cpython/pfdr_d1_ql1b_cpy.cpp", "../src/pfdr_d1_ql1b.cpp",
             "../src/matrix_tools.cpp", "../src/pfdr_graph_d1.cpp", 
             "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    setup(name=name, ext_modules=[mod], cmdclass=dict(build=MyBuild))
    
    
name = "pfdr_d1_lsx_cpy"
if name in to_compile:
    mod = Extension(
            name,
            # list source files
            ["cpython/pfdr_d1_lsx_cpy.cpp", "../src/pfdr_d1_lsx.cpp",
             "../src/proj_simplex.cpp", "../src/pfdr_graph_d1.cpp",
             "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"], 
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    setup(name=name, ext_modules=[mod], cmdclass=dict(build=MyBuild))

###  postprocessing  ###
try:
    shutil.rmtree("build") # remove temporary compilation products
except FileNotFoundError:
    pass

os.chdir(tmp_work_dir) # get back to initial working directory
