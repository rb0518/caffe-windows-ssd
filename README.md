# 修改记录
## 准备工作
- 在`caffe/window`文件夹中用系统文件浏览器搜索packages，选中所有搜索结果文件，右键删除所有文件
- 用`Visual Studio Code`打开`caffe/window`目录，编辑所有的.vcproj文件，删除如以下的内容
```
  <Import Project="..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.props" Condition="Exists('..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.props')" />
  <Import Project="..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.props" Condition="Exists('..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.props')" />
  <Import Project="..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.props" Condition="Exists('..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.props')" />

```

```
  <ImportGroup Label="ExtensionTargets">
    <Import Project="..\..\..\NugetPackages\OpenBLAS.0.2.14.1\build\native\openblas.targets" Condition="Exists('..\..\..\NugetPackages\OpenBLAS.0.2.14.1\build\native\openblas.targets')" />
    <Import Project="..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.targets" Condition="Exists('..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.targets')" />
    <Import Project="..\..\..\NugetPackages\hdf5-v120-complete.1.8.15.2\build\native\hdf5-v120.targets" Condition="Exists('..\..\..\NugetPackages\hdf5-v120-complete.1.8.15.2\build\native\hdf5-v120.targets')" />
    <Import Project="..\..\..\NugetPackages\boost_chrono-vc120.1.59.0.0\build\native\boost_chrono-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_chrono-vc120.1.59.0.0\build\native\boost_chrono-vc120.targets')" />
    <Import Project="..\..\..\NugetPackages\boost_date_time-vc120.1.59.0.0\build\native\boost_date_time-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_date_time-vc120.1.59.0.0\build\native\boost_date_time-vc120.targets')" />
    <Import Project="..\..\..\NugetPackages\boost_filesystem-vc120.1.59.0.0\build\native\boost_filesystem-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_filesystem-vc120.1.59.0.0\build\native\boost_filesystem-vc120.targets')" />
    <Import Project="..\..\..\NugetPackages\boost_regex-vc120.1.59.0.0\build\native\boost_regex-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_regex-vc120.1.59.0.0\build\native\boost_regex-vc120.targets')" />
	<Import Project="..\..\..\NugetPackages\boost_system-vc120.1.59.0.0\build\native\boost_system-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_system-vc120.1.59.0.0\build\native\boost_system-vc120.targets')" />
  	<Import Project="..\..\..\NugetPackages\boost.1.59.0.0\build\native\boost.targets" Condition="Exists('..\..\..\NugetPackages\boost.1.59.0.0\build\native\boost.targets')" />
    <Import Project="..\..\..\NugetPackages\boost_thread-vc120.1.59.0.0\build\native\boost_thread-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_thread-vc120.1.59.0.0\build\native\boost_thread-vc120.targets')" />
    <Import Project="..\..\..\NugetPackages\boost_python2.7-vc120.1.59.0.0\build\native\boost_python-vc120.targets" Condition="Exists('..\..\..\NugetPackages\boost_python2.7-vc120.1.59.0.0\build\native\boost_python-vc120.targets')" />
    <Import Project="..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.targets" Condition="Exists('..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.targets')" />
    <Import Project="..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.targets" Condition="Exists('..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.targets')" />
    <Import Project="..\..\..\NugetPackages\protobuf-v120.2.6.1\build\native\protobuf-v120.targets" Condition="Exists('..\..\..\NugetPackages\protobuf-v120.2.6.1\build\native\protobuf-v120.targets')" />
    <Import Project="..\..\..\NugetPackages\LevelDB-vc120.1.2.0.0\build\native\LevelDB-vc120.targets" Condition="Exists('..\..\..\NugetPackages\LevelDB-vc120.1.2.0.0\build\native\LevelDB-vc120.targets')" />
    <Import Project="..\..\..\NugetPackages\lmdb-v120-clean.0.9.14.0\build\native\lmdb-v120-clean.targets" Condition="Exists('..\..\..\NugetPackages\lmdb-v120-clean.0.9.14.0\build\native\lmdb-v120-clean.targets')" />
  </ImportGroup>
  <Target Name="EnsureNuGetPackageBuildImports" BeforeTargets="PrepareForBuild">
    <PropertyGroup>
      <ErrorText>This project references NuGet package(s) that are missing on this computer. Enable NuGet Package Restore to download them.  For more information, see http://go.microsoft.com/fwlink/?LinkID=322105. The missing file is {0}.</ErrorText>
    </PropertyGroup>
    <Error Condition="!Exists('..\..\..\NugetPackages\OpenBLAS.0.2.14.1\build\native\openblas.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\OpenBLAS.0.2.14.1\build\native\openblas.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.props')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.props'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\OpenCV.2.4.10\build\native\OpenCV.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\hdf5-v120-complete.1.8.15.2\build\native\hdf5-v120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\hdf5-v120-complete.1.8.15.2\build\native\hdf5-v120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost_chrono-vc120.1.59.0.0\build\native\boost_chrono-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_chrono-vc120.1.59.0.0\build\native\boost_chrono-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost_date_time-vc120.1.59.0.0\build\native\boost_date_time-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_date_time-vc120.1.59.0.0\build\native\boost_date_time-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost_filesystem-vc120.1.59.0.0\build\native\boost_filesystem-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_filesystem-vc120.1.59.0.0\build\native\boost_filesystem-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost_regex-vc120.1.59.0.0\build\native\boost_regex-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_regex-vc120.1.59.0.0\build\native\boost_regex-vc120.targets'))" />
	<Error Condition="!Exists('..\..\..\NugetPackages\boost_system-vc120.1.59.0.0\build\native\boost_system-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_system-vc120.1.59.0.0\build\native\boost_system-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost.1.59.0.0\build\native\boost.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost.1.59.0.0\build\native\boost.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost_thread-vc120.1.59.0.0\build\native\boost_thread-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_thread-vc120.1.59.0.0\build\native\boost_thread-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\boost_python2.7-vc120.1.59.0.0\build\native\boost_python-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\boost_python2.7-vc120.1.59.0.0\build\native\boost_python-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.props')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.props'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\gflags.2.1.2.1\build\native\gflags.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.props')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.props'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\glog.0.3.3.0\build\native\glog.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\protobuf-v120.2.6.1\build\native\protobuf-v120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\protobuf-v120.2.6.1\build\native\protobuf-v120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\LevelDB-vc120.1.2.0.0\build\native\LevelDB-vc120.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\LevelDB-vc120.1.2.0.0\build\native\LevelDB-vc120.targets'))" />
    <Error Condition="!Exists('..\..\..\NugetPackages\lmdb-v120-clean.0.9.14.0\build\native\lmdb-v120-clean.targets')" Text="$([System.String]::Format('$(ErrorText)', '..\..\..\NugetPackages\lmdb-v120-clean.0.9.14.0\build\native\lmdb-v120-clean.targets'))" />
  </Target>

```
修改`caffe.sln`中VS对应版本从`12.0`到`14.0`，修改目录下每一个`.vcproj`文件中`120`到`140`，可以直接用`VS2015`打开代码，而不是`VS2013`了。
`CPU_ONLY` compiler is OK. `GPU ` not success.

# 错误解决:

## 1 boost\regex 相关错误
搜索所有`regex`的字符，将头文件和`regex::xxx`内容都注释掉。

## 2 GLOG_NO_ABBREVIATED_SEVERITIES miss
add GLOG_NO_ABBREVIATED_SEVERITIES to CommonSettings.props中PreprocessDefinitions中。

## 3 找不到".\caffe\3rdparty\hungarian.h"文件
搜索solution发现这个头文件根本未调用过，所以从工程中直接删除文件包含。

## 4 expected an identifier in caffe.pb.h 
在bbox_util.cu，将两处`include "thrust\xxx`相关头文件引用去掉，同时将代码中`thrust::xxx`去掉。

## 5 caffe.proj 连接libcaffe.lib中GPU相关函数无法连接
修改配置后，重新打开工程文件后，要全清然后Rebuild libcaffe

## 6 pycaffe编译出错
参考了[happynear](https://github.com/happynear/caffe-windows)中处理方法，在'_caffe.cpp'文件中添加相尖DEFINE_BOOST_GET_POINTER定义，编译通过
。测试输出如下，进一步细节测试未进行
```ptyhon
>>>import caffe
>>>caffe.__version__
`CAFFEVERSION·
```

# Remark 
全部编译通过，经测试成功，但CPU方式运行速度奇慢无比，还应该有代码问题，明天重新编译后测试。

是通过修改[7oud](https://github.com/7oud/caffe-ssd-win) 克隆版本，参考了[happynear](https://github.com/happynear/caffe-windows)的方法。我自己修改了happynear提供的'thirdparty'库，整理后会提供下载。

懒人包提供地址:[云盘](http://pan.baidu.com/s/1qXMSG0k) 提取码: `wrxx`

### 懒人包内容

- 里面`Boost`和`LEVELDB`有两个目录带_163后缀的是我自己编译的`boost1.631`版本，`leveldb_163`目录是我用`boost_1.63`版本编译后的运行库，在选择时这两个应该保持一致
- `Opencv320`目录是我现在使用的，由于包含`contrib`所以很大，欢迎使用。我将`thirdparty`放到了`caffe`目录外，这样其它项目也能共享这些支撑库，节约空间，将`thirdparty\bins`添加到系统`path`里
- 包里内容是从`happynear`里内容,只是由于`boost1.61`是用`python2.7`选项编译的，所以我替换成了自己编译的支持`python3.x`版本，如果你是使用python2.7版本的，请从[happynear](https://github.com/happynear/caffe-windows)处下载原始的版本。

---
以下是作者原来的相关信息

# SSD: Single Shot MultiBox Detector

By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](http://research.google.com/pubs/DragomirAnguelov.html), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), Cheng-Yang Fu, [Alexander C. Berg](http://acberg.com).

### Introduction

SSD is an unified framework for object detection with a single network. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1512.02325).

<p align="center">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd.png" alt="SSD Framework" width="600px">
</p>

<center>

| System | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes |
|:-------|:-----:|:-------:|:-------:|
| [Faster R-CNN (VGG16)](https://github.com/ShaoqingRen/faster_rcnn) | 73.2 | 7 | 300 |
| [Faster R-CNN (ZF)](https://github.com/ShaoqingRen/faster_rcnn) | 62.1 | 17 | 300 |
| [YOLO](http://pjreddie.com/darknet/yolo/) | 63.4 | 45 | 98 |
| [Fast YOLO](http://pjreddie.com/darknet/yolo/) | 52.7 | 155 | 98 |
| SSD300 (VGG16) | 72.1 | 58 | 7308 |
| SSD300 (VGG16, cuDNN v5) | 72.1 | 72 | 7308 |
| SSD500 (VGG16) | **75.1** | 23 | 20097 |

</center>

### Citing SSD

Please cite SSD in your publications if it helps your research:

    @article{liu15ssd,
      Title = {{SSD}: Single Shot MultiBox Detector},
      Author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      Journal = {arXiv preprint arXiv:1512.02325},
      Year = {2015}
    }
    
## Windows Setup

This branch of Caffe extends [BVLC-led Caffe](https://github.com/BVLC/caffe) by adding Windows support and other functionalities commonly used by Microsoft's researchers, such as managed-code wrapper, [Faster-RCNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf), [R-FCN](https://arxiv.org/pdf/1605.06409v2.pdf), etc.

**Contact**: Kenneth Tran (ktran@microsoft.com)

---

# Caffe

|  **`Linux (CPU)`**   |  **`Windows (CPU)`** |
|-------------------|----------------------|
| [![Travis Build Status](https://api.travis-ci.org/Microsoft/caffe.svg?branch=master)](https://travis-ci.org/Microsoft/caffe) | [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/58wvckt0rcqtwnr5/branch/master?svg=true)](https://ci.appveyor.com/project/pavlejosipovic/caffe-3a30a) |              

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

**Requirements**: Visual Studio 2013

### Pre-Build Steps
Copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`

By defaults Windows build requires `CUDA` and `cuDNN` libraries.
Both can be disabled by adjusting build variables in `.\windows\CommonSettings.props`.
Python support is disabled by default, but can be enabled via `.\windows\CommonSettings.props` as well.
3rd party dependencies required by Caffe are automatically resolved via NuGet.

### CUDA
Download `CUDA Toolkit 7.5` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).
If you don't have CUDA installed, you can experiment with CPU_ONLY build.
In `.\windows\CommonSettings.props` set `CpuOnlyBuild` to `true` and set `UseCuDNN` to `false`.

### cuDNN
Download `cuDNN v4` or `cuDNN v5` [from nVidia website](https://developer.nvidia.com/cudnn).
Unpack downloaded zip to %CUDA_PATH% (environment variable set by CUDA installer).
Alternatively, you can unpack zip to any location and set `CuDnnPath` to point to this location in `.\windows\CommonSettings.props`.
`CuDnnPath` defined in `.\windows\CommonSettings.props`.
Also, you can disable cuDNN by setting `UseCuDNN` to `false` in the property file.

### Python
To build Caffe Python wrapper set `PythonSupport` to `true` in `.\windows\CommonSettings.props`.
Download Miniconda 2.7 64-bit Windows installer [from Miniconda website] (http://conda.pydata.org/miniconda.html).
Install for all users and add Python to PATH (through installer).

Run the following commands from elevated command prompt:

```
conda install --yes numpy scipy matplotlib scikit-image pip
pip install protobuf
```

#### Remark
After you have built solution with Python support, in order to use it you have to either:  
* set `PythonPath` environment variable to point to `<caffe_root>\Build\x64\Release\pycaffe`, or
* copy folder `<caffe_root>\Build\x64\Release\pycaffe\caffe` under `<python_root>\lib\site-packages`.

### Matlab
To build Caffe Matlab wrapper set `MatlabSupport` to `true` and `MatlabDir` to the root of your Matlab installation in `.\windows\CommonSettings.props`.

#### Remark
After you have built solution with Matlab support, in order to use it you have to:
* add the generated `matcaffe` folder to Matlab search path, and
* add `<caffe_root>\Build\x64\Release` to your system path.

### Build
Now, you should be able to build `.\windows\Caffe.sln`

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `%CAFFE_ROOT%`
  ```Shell
  git clone https://github.com/conner99/caffe.git
  cd caffe
  git checkout ssd-microsoft
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `%CAFFE_ROOT%\models\VGGNet\`

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `%CAFFE_ROOT%\data\VOC0712`
  
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```Explorer
# Extract the data manual.
%CAFFE_ROOT%\data\VOC0712\
		VOC2007\Annotations
		VOC2007\JPEGImages
		VOC2012\Annotations
		VOC2012\JPEGImages

  ```

3. Create the LMDB file.
  ```Shell
  cd %CAFFE_ROOT%
  # Create test_name_size.txt in data\VOC0712\
  .\data\VOC0712\get_image_size.bat
  # You can modify the parameters in create_data.bat if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - %CAFFE_ROOT%\data\VOC0712\trainval_lmdb
  #   - %CAFFE_ROOT%\data\VOC0712\test_lmdb
  .\data\VOC0712\create_data.bat
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - %CAFFE_ROOT%\models\VGGNet\VOC0712\SSD_300x300\
  # and job file, log file, and the python script in:
  #   - %CAFFE_ROOT%\jobs\VGGNet\VOC0712\SSD_300x300\
  # and save temporary evaluation results in:
  #   - %CAFFE_ROOT%\data\VOC2007\results\SSD_300x300\
  # It should reach 72.* mAP at 60k iterations.
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz).

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples\ssd\score_ssd_pascal.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples\ssd\ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out `examples/ssd_detect.ipynb` or `examples/ssd/ssd_detect.cpp` on how to detect objects using a SSD model.

5. To train on other dataset, please refer to data/OTHERDATASET for more details.
We currently add support for MSCOCO and ILSVRC2016.

### Models
1. Models trained on VOC0712: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_500x500.tar.gz)

2. Models trained on MSCOCO trainval35k: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_500x500.tar.gz)

3. Models trained on ILSVRC2015 trainval1: [SSD300](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_500x500.tar.gz) (46.4 mAP on val2)
