# Build instructions
Follow the build instructions from [Tensorflow](https://www.tensorflow.org/install/install_sources).
For our projects, we use the branch **r1.3**.
Make sure, that you use a new environment for tensorflow as described in the
build instructions.

## Use tensorflow environment in pycharm
If you use gcc>=5, additionally add the flag `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`

## Add a new module to contrib
- Add line `"//tensorflow/contrib/<your-module>:all_files"` to `tensorflow/BUILD`
under section `filegroup(
    name = "all_opensource_files",`
- Add line `"//tensorflow/contrib/<your-module>:<your-module>_py"` to
`tensorflow/contrib/BUILD` under section `py_library(
    name = "contrib_py",`
- Add line `from tensorflow.contrib import <your-module>` to `tensorflow/contrib/__init__.py`
