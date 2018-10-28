macOS: 10.13.3(17D102)

WebDriver: 387.10.10.10.25.156

CUDA Toolkit: 9.1.128

cuDNN: 7

NVDAEGPUSupport: 6

XCode: 8.2

Bazel: 0.15.2(\>=0.15.0)

OpenMP: latest

Python: 3.6.5

TensorFlow: 1.11.0

<https://qiita.com/74th/items/fc6ebb684c23f3655e7c>

<http://paolino.me/tutorial/tensorflow/machine-learning/deep-learning/gpu/2017/11/18/installing-tensorflow-1.4.0-macos-cuda/>

<https://gist.github.com/smitshilu/53cf9ff0fd6cdb64cca69a7e2827ed0f>

[http://melonteam.com/posts/pei\_zhi\_tensorflow\_gpu\_ban\_ben\_tian\_keng\_lu/](http://melonteam.com/posts/pei_zhi_tensorflow_gpu_ban_ben_tian_keng_lu/)

<https://tweakmind.com/tensorflow-1-5-macos-10-13-2/>

eGPU part(out of date, find the latest solutions at egpu.io)
============================================================

<https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/>

<https://egpu.io/forums/implementation-guides/2017-15-macbook-pro-radeon-pro-555-gtx1080ti32gbps-tb3-mantiz-venus-macos10-13-4-win10/>

<https://github.com/vulgo/webdriver.sh>

1 Remove/undo any Info.plist modifications (they aren’t needed anymore and might conflict).

2 Disable SIP

* Restart holding command + r
* Execute 'csrutil disable’
* Restart

3 Install WebDriver with webdriver.sh[](https://github.com/vulgo/webdriver.sh.git)

```sh
git clone https://github.com/vulgo/webdriver.sh.git
cd webdriver.sh
sudo ./webdriver.sh -u https://us.download.nvidia.com/Mac/Quadro_Certified/387.10.10.10.25.156/WebDriver-387.10.10.10.25.156.pkg
```

4 Enable SIP

* Restart holding command + r
* Execute 'csrutil enable --without kext’
* Restart

5 Install nvidia-egpu

* Download https://cdn.egpu.io/wp-content/uploads/wpforo/attachments/71/4376-NVDAEGPUSupport-v6.zip
* Unzip 4376-NVDAEGPUSupport-v6.zip
* Install package NVDAEGPUSupport-v6.pkg

6 Check eGPU status

* Reboot
* Attach the egpu
* Login
* Check whether the GPU is recognized

CUDA & cuDNN
============

1 Download packages

\#cudadriver\_387.128\_macos.dmg

 [http://us.download.nvidia.com/Mac/cuda\_387/cudadriver\_387.128\_macos.dmg](http://us.download.nvidia.com/Mac/cuda_387/cudadriver_387.128_macos.dmg)

 \#cuda\_9.1.128\_mac.dmg

 <https://developer.nvidia.com/cuda-toolkit-archive>

 \#cudnn-9.1-osx-x64-v7-ga.tgz

 <https://developer.nvidia.com/rdp/cudnn-download>

2 Install packages

1) Install cuda\_9.1.128\_mac.dmg with default options

 2) Install cudadriver\_387.128\_macos.dmg with default options

 3) Install cuDNN

```sh
tar -zxf cudnn-9.1-osx-x64-v7-ga.tgz
cd cuda
sudo cp -RPf include/* /Developer/NVIDIA/CUDA-9.1/include/
sudo cp -RPf lib/* /Developer/NVIDIA/CUDA-9.1/lib/
sudo ln -s /Developer/NVIDIA/CUDA-9.1/lib/libcudnn* /usr/local/cuda/lib/
```

3 Add environment variables

```sh
vim ~/.zshrc
#if you use bash, this should be ~/.bash_profile
  export CUDA_HOME=/usr/local/cuda
  export DYLD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/extras/CUPTI/lib
  export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
  export PATH=$CUDA_HOME/bin:$PATH
source ~/.zshrc
```

3 Verify CUDA works fine

```sh
cd /Developer/NVIDIA/CUDA-9.1/samples/1_Utilities/deviceQuery
sudo make 
./deviceQuery
```

Homebrew part
=============

1 Install homebrew

```sh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

2 Install coreutils

```sh
brew install coreutils
```

3 Install OpenMP

https://clang-omp.github.io

 http://antonmenshov.com/2017/09/09/clang-openmp-setup-in-xcode/

 https://iscinumpy.gitlab.io/post/omp-on-high-sierra/

```sh
brew install cliutils/apple/libomp
```

4 Install bazel

```sh
brew install bazel
bazel version
```

5 Install anaconda

```sh
brew install anaconda
vim ~/.zshrc
#If you use bash, this should be ~/.bash_profile
  export PATH=/usr/local/anaconda3/bin:$PATH
source ~/.zshrc
```

Install XCode 8.2
=================

* Download https://download.developer.apple.com/Developer\_Tools/Xcode\_8.2/Xcode\_8.2.xip
* Extract it, and rename Xcode.app to Xcode8.2.app
* Drag Xcode8.2.app to Applications

```sh
sudo xcode-select -s /Applications/Xcode8.2.app
```

You need restore xcode configuration or remove Xcode8.2.app later, since it will break homebrew

```sh
sudo xcode-select -s /Applications/Xcode.app
```

Install TensorFlow
==================

1 Add virtualenv and activate virtualenv

```sh
conda create --p egpu python=3.6
source activate egpu
pip install six numpy wheel mock
pip install keras_applications==1.0.5 --no-deps
pip install keras_preprocessing==1.0.3 --no-deps
pip install h5py==2.8.0
```

2 Clone TensorFlow code

```sh
git clone https://github.com/tensorflow/tensorflow.git -b v1.11.0
```

3 Modify the code, to make it compatible with macOS

```sh
cd tensorflow
sed -i -e "s/ __align__(sizeof(T))//g" tensorflow/core/kernels/concat_lib_gpu_impl.cu.cc
sed -i -e "s/ __align__(sizeof(T))//g" tensorflow/core/kernels/split_lib_gpu.cu.cc
sed -i -e "s/const Subgraph\:\:Identity empty_parent/Subgraph\:\:Identity empty_parent/g" tensorflow/core/grappler/graph_analyzer/graph_analyzer.cc
#disable nccl
sed -i -e "s/\"\/\/tensorflow\/contrib\/nccl/\#\"\/\/tensorflow\/contrib\/nccl/g" tensorflow/contrib/BUILD
sed -i -e "s/\"\/\/tensorflow\/contrib\/nccl/\#\"\/\/tensorflow\/contrib\/nccl/g" tensorflow/contrib/all_reduce/BUILD
sed -i -e "s/\"\/\/tensorflow\/contrib\/nccl/\#\"\/\/tensorflow\/contrib\/nccl/g" tensorflow/contrib/distribute/python/BUILD
```

4 Compile the code

```sh
./configure
 #Please specify the location of python.: Accept the default option
 #Please input the desired Python library path to use.: Accept the default option
 #Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
 #Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
 #Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: n
 #Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
 #Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
 #Do you wish to build TensorFlow with GDR support? [y/N]: n
 #Do you wish to build TensorFlow with VERBS support? [y/N]: n
 #Do you wish to build TensorFlow with nGraph support? [y/N]: n
 #Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
 #Do you wish to build TensorFlow with CUDA support? [y/N]: y
 #Please specify the CUDA SDK version you want to use.: 9.1
 #Please specify the location where CUDA 9.1 toolkit is installed.: Accept the default option
 #Please specify the cuDNN version you want to use.: 7
 #Please specify the location where cuDNN 7 library is installed.: Accept the default option
 ##Please specify a list of comma-separated Cuda compute capabilities you want to build with.
 ##You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.(GTX10X0: 6.1, GTX9X0: 5.2)
 #Please note that each additional compute capability significantly increases your build time and binary size.: 6.1
 #Do you want to use clang as CUDA compiler? [y/N]: n
 #Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: Accept the default option
 #Do you wish to build TensorFlow with MPI support? [y/N]: n
 #Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified: Accept the default option
 #Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n

export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
export PATH=$DYLD_LIBRARY_PATH:$PATH

#bazel clean --expunge
bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```

5 Build the wheel and install

```sh
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.11.0-cp36-cp36m-macosx_10_9_x86_64.whl
```

ERRORS
======

1 CUDA\_ERROR\_OUT\_OF\_MEMORY

<https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow>

 <https://stackoverflow.com/questions/43467586/tensorflow-cuda-error-out-of-memory-always-happen>

 <https://stackoverflow.com/questions/45546737/cuda-error-out-of-memory-how-to-activate-multiple-gpus-from-keras-in-tensorflow>

```python
#TensorFlow
gpu_options = tf.GPUOptions(allow_growth=True) 
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

#Keras
import keras.backend as K
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
```

2 PyCharm Library not loaded: @rpath/libcudnn.7.dylib

<https://stackoverflow.com/questions/37933890/tensorflow-gpu-setup-error-with-cuda-on-pycharm>

Add environment variables to python default configuration:

 CUDA\_HOME=/usr/local/cuda

 DYLD\_LIBRARY\_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib

 LD\_LIBRARY\_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib

