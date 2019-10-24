<div><div align="center">
<img src="https://www.tugraz.at/uploads/RTEmagicC_vlo_logo_02.png.png" height="250"/>
 <span style="padding-left:70px;"></span>
<img src="https://www.tensorflow.org/images/tf_logo_transp.png" height="250"/>
<br><br><br>
</div>

# About the repository

This repository is forked from the original [tensorflow repository](https://github.com/tensorflow/tensorflow), we are currently up-to-date with the `r1.5` branch. The repository contains newly developed features in `tensorflow/contrib/icg`. We provide custom operators, functions and classes:
- trainable activation functions
- Fourier operations such as 2D centered (I)FT and (i)fftshift
- complex convolution
- a basis for the variational network [1-3]
- iPALM optimizer [4].

## Contact
*  Kerstin Hammernik <hammernik@icg.tugraz.at>
*  Erich Kobler <erich.kobler@icg.tugraz.at>

## References
1.  Y Chen, W Yu, T Pock. [*On learning optimized reaction diffusion processes for effective image restoration*](https://arxiv.org/abs/1503.05768). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5261-5269, 2015.
2. E Kobler, T Klatzer, K Hammernik, T Pock. [*Variational Networks: Connecting Variational Methods and Deep Learning*](https://link.springer.com/chapter/10.1007/978-3-319-66709-6_23). German Conference on Pattern Recognition, pp 281-293, 2017.
3. K Hammernik, T Klatzer, E Kobler, MP Recht, DK Sodickson, T Pock, F Knoll. [*Learning a Variational Network for Reconstruction of Accelerated MRI Data*](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26977/abstract). Magnetic Resonance in Medicine, 79(6), pp. 3055-3071, 2018.
4. T Pock and S Sabach. [*Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems*](https://arxiv.org/abs/1702.02505). SIAM Journal on Imaging Science, 9(4), pp. 1756–1787, 2016.


# Build the framework
We currently use following setup:
*  Ubuntu 16.04
*  Cuda 8.0
*  Cudnn 7.0  (tar file installer for Linux is recommended, see below)
*  Python 3.6
*  Bazel 0.11.1

## Preparations
Please follow the instructions to prepare an environment for linux [here](https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux). Also, read and do the section about GPU prerequisite. Make sure to append the paths of cuda and cupti to `LD_LIBRARY_PATH`, i.e., add following line to `~/.bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64/:/usr/local/cuda-8.0/extras/CUPTI/lib64/
```

## Cudnn installation recommendation:
We recommend to use the tar-file installation for Linux instead of the deb file for Ubuntu.
The debian installation uses other paths than expected by tensorflow and doesn't provide much benefits. Follow the official Nvidia guidelines from:
http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar

## Python
We use [Anaconda](https://www.anaconda.com/download/#linux). Download python 3.6 version, install and verify that the Anaconda path is added to the `~/.bashrc`. Do not forget to `source ~/.bashrc` or log out and log in again. Next, we set up a new environment for tensorflow:

```
conda  create -f <path-to-icg-tensorflow-repo>/anaconda_env_tficg.yml --name icgenv
```

## Build tensorflow
Following these steps will build a wheel package that can be installed via pip. Open a console, navigate to the `icg-tensorflow` root folder, activate the python environment with `source activate tficg` before starting with the configuration. The configuration is stored in the file `tf_exports.sh`. Adapt the settings according to your machine.
* We recommend to use Cuda 8.0. Certain graphics card do not work with Cuda < 8.0 ;)
* If you want to use the compiled package on different machines, list the corresponding compute capabilities.
Save the configuration and type `source tf_exports.sh` followed by `./configure`. Make sure that the path to python points to the tficg environment of Anaconda.

After configuring, type `bash build.sh`. Your computer might be terribly busy while compiling tensorflow, so we suggest reading papers in the meanwhile. If the build is finished, a command starting with `pip install ...` is printed in red. Perform following steps:

```
source activate tficg
pip install <your-build-pkg>.whl
```

## Test your installation
*  Navigate outside the tensorflow root folder. This is important, yes! Otherwise you will get following error message: *ModuleNotFoundError: No module named 'tensorflow.python.pywrap_tensorflow_internal' Failed to load the native TensorFlow runtime.*
*  [Common installation problems](https://www.tensorflow.org/install/install_linux#common_installation_problems)
*  Open `ipython` and simply run `import tensorflow`
* If you get following error message: *ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.23' not found* or something with the `libstdc++.so`, the specific library versions are not found in the tensorflow environment. The libraries can be found on the machine that you used for building tensorflow. Pre-load these libs by adding following line to `~/.bashrc`:
```
export LD_PRELOAD=<path-to-libs>/libstdc++.so:<path-to-libs>/libm-2.23.so
```

## Use it
``` python
import tensorflow as tf
print(help(tf.contrib.icg))  # List available functions
```
Happy coding!
