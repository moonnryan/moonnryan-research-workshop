---
title: Sophgo 模型编译部署教程
created: 2025-12-03
updated: 2026-04-22
status: draft
blog: false
tags: [Sophgo, TPU-MLIR, SAIL, Qwen3-VL, YOLOv5]
---

# Sophgo 模型编译部署教程

本文以算能 BM1684X SoC 部署为主，覆盖 Qwen3-VL 视觉语言模型与 YOLOv5 检测模型的环境准备、编译量化和 Python/C++ 运行。两类模型的导出路径不同，但都需要将编译产物、前后处理和设备运行库对齐。


| 工具或产物 | 在流程中的作用 |
| --- | --- |
| TPU-MLIR | 转换计算图、量化、编译为目标 TPU 可执行模型 |
| BModel | 编译后的设备模型，受芯片、shape 和精度配置约束 |
| SDK / libsophon | 目标设备上的底层运行接口与依赖 |
| SAIL | 封装设备句柄、张量、推理和图像处理的 Python/C++ 接口 |

| 环节 | 执行环境 | 输入与产物 |
| --- | --- | --- |
| 模型编译与量化 | x86 主机的 TPU-MLIR 容器 | 模型权重 / ONNX → BModel；大模型同时保留 config |
| 程序交叉编译 | x86 开发容器，使用 ARM64 工具链及 SDK | SAIL wheel / 动态库、SoC C++ 可执行文件 |
| 部署与验收 | 目标 SoC 设备 | 加载模型及匹配依赖，执行真实图片 / 视频输入 |

按“环境准备 → 模型编译与量化 → 例程构建与设备运行”阅读即可。已有 SDK 或运行包时可以跳过对应构建步骤。示例中的绝对路径是原现场目录，使用时替换为自己的模型和 SDK 路径。

原流程涉及 libsophon 0.5.1、sophon-mw 0.10.0、SAIL 3.8.0 和 Python 3.10.14 等历史组件，另有 `latest` 镜像及未固定的 pip 依赖。这些信息不能组成一套已验证的全局兼容矩阵；复现时应记录实际镜像、工具链、框架和运行库版本。这些命令用于说明完整操作流程，不预设全部历史版本在同一环境中互相兼容。

## 一、环境准备
BModel模型的编译量化：需要在一台x86主机上按照指引配置docker环境，在docker环境内进行模型的编译量化。

BModel模型验证：需要将编译好的bmodel拷贝至SoC设备上运行验证。

程序的交叉编译：x86主机系统建议使用Ubuntu16.04~20.04。按照本章指引在x86主机上安装libsophon、sophon-mw，进行程序的交叉编译，待程序编译好后，需要手动将编译好的程序拷贝到目标SoC中执行。

算法的应用部署：可直接在SoC设备上执行程序。

### 1.1 TPU-MLIR 编译环境
BModel模型编译需要依赖transformers官方仓库和TPU-MLIR工具包，本教程采用 x86 主机进行模型编译。

> 注意:
>
> 1.原小模型示例的准备参考为主机内存不少于 15 GB、磁盘空间不少于 30 GB；这是该流程的环境参考，32B、多设备或多任务并行编译需另外评估峰值资源。
>
> 2.使用能够加载 Qwen3-VL 的 transformers 版本，并与所选 TPU-MLIR 版本配套。不能将“4.49.0 开始支持 Qwen3-VL”作为安装依据；先验证 Qwen3VL 模型类可导入。Python 与框架版本按对应工具链要求固定。
>
> 3.可以选择工具链明确支持的 AWQ 模型，也可以从原始浮点权重编译。源模型量化、转换误差和业务效果分别验证，不预设 AWQ 转 BModel 一定无损。
>

安装好docker后，从 DockerHub 项目资料 下载所需的镜像文件，或使用下方命令直接拉取镜像：

```bash
docker pull sophgo/tpuc_dev:latest
```

若下载失败，可从官网开发资料 项目资料 下载所需镜像文件，或使用下方命令下载镜像：

```bash
mkdir -p /home/sophgo-workspace
cd /home/sophgo-workspace
wget https://sophon-assets.sophon.cn/sophon-prod-s3/drive/25/04/15/16/tpuc_dev_v3.4.tar.gz
docker load -i tpuc_dev_v3.4.tar.gz
```

如果是首次使用Docker, 可执行下述命令进行安装和配置（仅首次执行）：

```bash
sudo yum install -y docker-ce docker-ce-cli docker-buildx-plugin docker-compose-plugin --allowerasing
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

镜像加载后创建容器。以下使用 `sophgo/tpuc_dev:latest` 演示；使用离线镜像时，将镜像名替换为 `docker images` 中实际加载的 tag。复现记录应保留 image digest 和工具版本。

```bash
docker run -it \
  --privileged \
  --name BModel_Compile \
  --memory=32g \
  --memory-swap=64g \
  -v /home/sophgo-workspace/:/workspace \
  sophgo/tpuc_dev:latest

docker exec -it BModel_Compile /bin/bash
```

后文假定已进入编译容器的 `/workspace`。源码构建方式如下；实际项目记录 git commit，不以“最新版”作为版本标识：

```bash
cd /workspace
git clone https://github.com/sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh

echo "source /workspace/tpu-mlir/envsetup.sh" >> /root/.bashrc
pip3 install -U transformers torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
```

也可以使用 pip 安装版，以下命令会安装当前仓库可用版本。源码构建与 pip 安装选择一条路径，避免混用后无法确认 `tpuc-opt` 来源：

```bash
pip install tpu_mlir -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
```

### 1.2 SDK 与 SAIL 构建环境

下面分别保留 RPM 系环境的 Python 准备和 Ubuntu 交叉编译容器的依赖处理，不需要把两组安装命令都执行一遍。已有 Python 3.10 时跳过源码安装；历史环境中的系统 Python/pip 调整只用于对应的开发容器。模型编译容器与程序交叉编译容器承担不同任务。

+ **手动编译 Python 3.10**  若服务器无法访问 SCL 源，可手动编译源码：

```bash
# 安装编译依赖
yum install -y gcc make zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel openssl-devel libffi-devel
ls /usr/include/openssl/  # 应有ssl.h等头文件

# 下载Python 3.10源码
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -xf Python-3.10.14.tgz
cd Python-3.10.14

# 编译安装（指定安装路径，不覆盖系统python）
./configure --prefix=/usr/local/python310 \
            --enable-optimizations \
            --with-openssl=/usr \
            --with-openssl-rpath=auto
make -j $(nproc)
make install

# 替换软链接
ln -s /usr/local/python310/bin/python3.10 /usr/bin/python
ln -s /usr/local/python310/bin/pip3.10 /usr/bin/pip

# 创建全局pip配置文件
vi /etc/pip.conf

# 粘贴内容
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
timeout = 120
retries = 3

[install]
use-mirrors = true
mirrors = https://mirrors.aliyun.com/pypi/simple/
```

+ **使用提供的docker镜像**  作为交叉编译环境，通过dfss下载：

```bash
pip install dfss
python -m dfss --url=open@sophgo.com:sophon-demo/common/docker/stream_dev.tar

docker load -i stream_dev.tar
docker run --privileged --name Sophon_Cross_Dev -v $PWD:/workspace -it stream_dev:latest
docker exec -it Sophon_Cross_Dev /bin/bash
```

+ **打包libsophon**  从算能官网上下载符合环境依赖的SOPHONSDK，在sophon-img_{date}_{time}文件夹中包括libsophon_soc_x.y.z_aarch64.tar.gz，x.y.z表示版本号，并进行解压。

```bash
# 创建依赖文件的根目录
mkdir -p soc-sdk
# 解压libsophon_soc_x.y.z_aarch64.tar.gz
tar -zxf libsophon_soc_0.5.1_aarch64.tar.gz
# 将相关的库目录和头文件目录拷贝到依赖文件根目录下
cp -rf libsophon_soc_0.5.1_aarch64/opt/sophon/libsophon-0.5.1/lib soc-sdk
cp -rf libsophon_soc_0.5.1_aarch64/opt/sophon/libsophon-0.5.1/include soc-sdk
```

+ **打包sophon-ffmpeg和sophon-opencv**  从算能官网上下载符合环境依赖的SOPHONSDK，在sophon-mw_{date}_{time}文件夹中包括sophon-mw-soc_x.y.z_aarch64.tar.gz，x.y.z表示版本号，并进行解压，如果您使用BM1688 SOPHONSDK，“sophon-mw”这个名字或许需要替换成“sophon-media”。

```bash
# 解压sophon-mw-soc_x.y.z_aarch64.tar.gz
tar -zxf sophon-mw-soc_0.10.0_aarch64.tar.gz
# 将ffmpeg和opencv的库目录和头文件目录拷贝到soc-sdk目录下
cp -rf sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-ffmpeg_0.10.0/lib soc-sdk
cp -rf sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-ffmpeg_0.10.0/include soc-sdk
cp -rf sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-opencv_0.10.0/lib soc-sdk
cp -rf sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-opencv_0.10.0/include soc-sdk
```

  这里，交叉编译环境已经搭建完成，接下来可以使用打包好的soc-sdk编译需要在SoC平台上运行的程序，更多交叉编译信息请参考《LIBSOPHON使用手册.pdf》。

+ **编译可被Python3接口调用的Wheel文件-SOC MODE**  在目标 SoC 的 Python 环境中，可以尝试通过 dfss 安装匹配的 SAIL 包；如果没有对应版本或架构，再走下文源码交叉编译：

```bash
pip install dfss && python -m dfss --install sail
```

  如果无法通过以上命令安装，那么需要自己编译，从算能官网上下载符合环境依赖的SOPHONSDK，进入sophon-sail_{date}_{time}文件夹，sophon-sail的发布包命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号，可以打开同级目录下的用户手册(命名为sophon-sail_zh.pdf或SOPHON-SAIL_zh.pdf)。 参考编译安装指南章节，选择需要的模式(C++/Python，SoC MODE)进行安装，**注意需要选择包含ffmpeg和opencv的编译方式。**

```bash
# 安装Python编译必需的依赖
apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
# 下载Python 3.10.14（稳定版）
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
# 解压源码包
tar -xf Python-3.10.14.tgz
cd Python-3.10.14

# 配置编译参数（指定安装路径，启用优化和SSL）
./configure --prefix=/usr/local/python310 \
            --enable-optimizations \
            --with-openssl=/usr \
            --enable-shared
# 编译（-j后跟CPU核心数，加快速度，容器默认至少1核）
make -j $(nproc)
# 安装（altinstall避免覆盖系统python3）
make altinstall

# 关键修复：配置Python共享库路径（解决libpython3.10.so.1.0找不到的问题）
echo "/usr/local/python310/lib" > /etc/ld.so.conf.d/python310.conf
ldconfig

# 1. 先删除系统旧pip软链接（提前删除避免后续创建冲突）
rm -f /usr/bin/pip
# 顺带清理可能存在的旧python软链接
rm -f /usr/bin/python

# 2. 卸载系统级旧pip（避免干扰）
apt remove -y python3-pip

# 3. 清理pip残留缓存
rm -rf ~/.cache/pip
rm -rf /usr/lib/python3/dist-packages/pip*

# 创建python软链接（指向3.10）
ln -s /usr/local/python310/bin/python3.10 /usr/bin/python

# 用Python 3.10的ensurepip重新安装pip（适配3.10）
/usr/local/python310/bin/python3.10 -m ensurepip --upgrade

# 升级到最新版pip（避免旧版本bug）
/usr/local/python310/bin/pip3.10 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

# 创建pip软链接（指向3.10的pip）
ln -s /usr/local/python310/bin/pip3.10 /usr/bin/pip

# 配置pip国内源（放在软链接创建后，确保pip命令可用）
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF

# 卸载旧版pyOpenSSL
pip uninstall -y pyopenssl

# 安装适配Python 3.10的新版pyOpenSSL
pip install pyopenssl -i https://mirrors.aliyun.com/pypi/simple/

# 顺带修复urllib3/requests（pip依赖的核心库）
pip install --upgrade urllib3 requests -i https://mirrors.aliyun.com/pypi/simple/

cd ..
tar -zxf sophon-sail_3.8.0.tar.gz
cd sophon-sail

apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

mkdir build && cd build
cmake -DBUILD_TYPE=soc  \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
    -DPYTHON_EXECUTABLE=/usr/local/python310/bin/python3.10 \
    -DCUSTOM_PY_LIBDIR=/usr/local/python310/lib \
    -DLIBSOPHON_BASIC_PATH=../../libsophon_soc_0.5.1_aarch64/opt/sophon/libsophon-0.5.1 \
    -DFFMPEG_BASIC_PATH=../../sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-ffmpeg_0.10.0 \
    -DOPENCV_BASIC_PATH=../../sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-opencv_0.10.0 ..
make pysail

cd ../python/soc
chmod +x sophon_soc_whl.sh
pip install wheel
./sophon_soc_whl.sh

# 将‘./dist/sophon_arm-3.8.0-py3-none-any.whl’拷贝到目标SOC上,然后执行如下安装命令
pip install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```

+ **编译可被C++接口调用的动态库及头文件-SOC MODE**  通过交叉编译的方式,编译出包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL。

```bash
# 创建编译文件夹build,并进入build文件夹
cd ../..
rm -rf build
mkdir build && cd build

# 执行编译命令
cmake -DBUILD_TYPE=soc -DBUILD_PYSAIL=OFF \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
    -DLIBSOPHON_BASIC_PATH=../../libsophon_soc_0.5.1_aarch64/opt/sophon/libsophon-0.5.1 \
    -DFFMPEG_BASIC_PATH=../../sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-ffmpeg_0.10.0 \
    -DOPENCV_BASIC_PATH=../../sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-opencv_0.10.0 ..
make sail

# 安装SAIL动态库及头文件,程序将自动在源码目录下创建‘build_soc’,编译结果将安装在‘build_soc’下面
make install
cd ../build_soc
zip -r sophon-sail.zip sophon-sail/

# 将交叉编译好的SAIL,也即是‘build_soc’拷贝至‘soc-sdk’文件夹
cd ../..
cp -r sophon-sail/build_soc/sophon-sail/include soc-sdk
cp -r sophon-sail/build_soc/sophon-sail/lib soc-sdk
```

  在按照教程将sophon-sail的库文件拷贝到目标soc上之后，还需要设置以下环境变量：

```bash
echo 'export LD_LIBRARY_PATH=/opt/sophon/sophon-sail/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

  将‘build_soc’文件夹下的`‘sophon-sail’`拷贝至目标SOC的`‘/opt/sophon’`目录下,即可在soc上面进行调用。  在使用PCIE MODE编译好SAIL，并且通过执行‘sudo make install’或者通过拷贝的方式安装好SAIL的c++库之后,，推荐使用cmake来将SAIL中的库链接到自己的程序中，如果需要使用SAIL多媒体相关的功能，也需要将libsophon,sophon-ffmpeg,sophon-opencv的头文件目录及动态库目录添加到自己的程序中。

在 SoC 设备上安装好libsophon、sophon-ffmpeg、sophon-opencv，及SAIL之后，可以参考PCIE MODE的开发方法使用cmake将SAIL中的库链接到自己的程序中，如果需要使用SAIL多媒体相关的功能，也需要将libsophon、sophon-ffmpeg、sophon-opencv的头文件目录及动态库目录添加到自己的程序中。  可以在程序的CMakeLists.txt中添加如下段落:

```cmake
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
# 添加libsophon的头文件目录

set(SAIL_DIR  /opt/sophon/sophon-sail/lib/cmake)
find_package(SAIL REQUIRED)
include_directories(${SAIL_INCLUDE_DIRS})
link_directories(${SAIL_LIB_DIRS})
# 添加SAIL的头文件及动态库目录

set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 添加sophon-opencv的头文件目录

set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
find_package(FFMPEG REQUIRED)
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIB_DIRS})
# 添加sophon-ffmpeg的头文件及动态库目录

add_executable(${YOUR_TARGET_NAME} ${YOUR_SOURCE_FILES})
target_link_libraries(${YOUR_TARGET_NAME} sail)
```

  在代码中即可以调用sail中的函数：

```cpp
#define USE_FFMPEG  1
#define USE_OPENCV  1
#define USE_BMCV    1

#include <stdio.h>
#include <sail/cvwrapper.h>
#include <iostream>
#include <string>

using namespace std;

int main()
{
    int device_id = 0;
    std::string video_path = "test.avi";
    sail::Decoder decoder(video_path,true,device_id);
    if(!decoder.is_opened()){
        printf("Video[%s] read failed!\n",video_path.c_str());
        exit(1) ;
    }

    sail::Handle handle(device_id);
    sail::Bmcv bmcv(handle);

    while(true){
        sail::BMImage ost_image = decoder.read(handle);
        bmcv.imwrite("test.jpg", ost_image);
        break;
    }

    return 0;
}
```


## 二、模型编译与量化
进入前文创建的模型编译容器 `BModel_Compile`。下述 Qwen3-VL、YOLOv5 命令均在该容器执行。

### 2.1 Qwen3-VL 转换
`llm_convert.py` 的直接构图路径读取模型权重与配置，按语言层及视觉等模块构建 MLIR 子图，再编译并合并为 BModel，同时导出运行所需的 config。ONNX 导出路径则先生成逐层 ONNX，再通过 `model_transform.py` 转成 MLIR。具体流程由转换入口和工具版本决定。

支持的主要参数如下:

| **参数名** | **简写** | **必选？** | **说明** |
| --- | --- | --- | --- |
| model_path | m | 是 | 指定权重路径 |
| seq_length | s | 是 | 指定序列最大长度 |
| max_input_length | - | 否 | 指定单次最大输入长度；默认采用seq_length |
| quantize | q | 是 | 指定量化类型, w4bf16/w4f16/bf16/f16等等 |
| chip | c | 是 | 指定平台, 如bm1684x/bm1688/cv186x |
| q_group_size | g | 否 | 指定每组量化的组大小, 默认64 |
| max_pixels | - | 否 | qwen vl参数, 指定最大图片尺寸, 可以是`672,896`,也可以是`602112` |
| do_sample | - | 否 | 指定输出是否包含采样模型，默认关闭 |
| out_dir | o | 是 | 指定输出目录 |


还有更多参数可以参考进阶应用，其中 `w4bf16` / `w4f16` 表示四位权重配合 BF16 / F16 激活，需同时检查模型 dtype、目标芯片和转换器支持范围；`config.json` 中的 dtype 是选择依据之一。

`llm_convert.py`执行完成后在指定目录会生成对应的bmodel和配置目录config。

未被该转换器覆盖的模型，需要参考对应 demo 的导出与部署流程。CV 模型常走 ONNX → MLIR → BModel；大模型则可能按 embedding、prefill、decode/cache 和视觉模块分别导出，不能默认所有模型都采用同一条整图 ONNX 路线。

下载模型：

```bash
modelscope download --model Qwen/Qwen3-VL-2B-Instruct --local_dir ./Qwen3-VL-2B-Instruct
```

编译命令如下：

```bash
llm_convert.py -m /workspace/Qwen3-VL-2B-Instruct -s 4096 --max_input_length 2048 --quantize w4bf16 -c bm1684x --out_dir qwen3vl_2b --max_pixels 768,768 --use_block_with_kv

llm_convert.py -m /workspace/Qwen3-VL-2B-Instruct -s 4096 --max_input_length 2048 --quantize w8bf16 -c bm1684x --out_dir qwen3vl_2b --max_pixels 768,768 --use_block_with_kv
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625324589-875c617a-3fa7-4c01-ab9b-df98e95fe540.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625323901-acbfcd88-a322-4fe8-b70c-5e5686a0c1d6.png)


```bash
# 也可以尝试4B和8B的模型
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir ./Qwen3-VL-4B-Instruct
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./Qwen3-VL-8B-Instruct

llm_convert.py -m /workspace/Qwen3-VL-4B-Instruct -s 4096 --max_input_length 2048 --quantize w4bf16 -c bm1684x --out_dir qwen3vl_4b --max_pixels 768,768 --use_block_with_kv

llm_convert.py -m /workspace/Qwen3-VL-4B-Instruct -s 4096 --max_input_length 2048 --quantize w8bf16 -c bm1684x --out_dir qwen3vl_4b --max_pixels 768,768 --use_block_with_kv

llm_convert.py -m /workspace/Qwen3-VL-8B-Instruct -s 4096 --max_input_length 2048 --quantize w4bf16 -c bm1684x --out_dir qwen3vl_8b --max_pixels 768,768 --use_block_with_kv
```

**YOLOv5 的 ONNX、校准与混合精度流程**

本节依次建立浮点基线、生成 INT8 校准表、搜索混合精度。`350,498,646` 是示例 ONNX 的输出节点名，换模型时先检查真实输出，不能直接照抄。模型转换、校准与正式推理需要使用相同的前处理语义。

### 2.2 YOLOv5 模型准备

在线安装方式对于 `onnx` 或 `torch` 生成的模型文件，可使用下方命令安装额外的依赖环境:

```bash
# 同时安装onnx, torch依赖
pip install tpu_mlir[onnx,torch] -i https://mirrors.aliyun.com/pypi/simple/
```

从Github的 Assets 处下载 `tpu-mlir-resource.tar` 并解压，解压后将文件夹重命名为 `tpu-mlir-resource` ：

```bash
tar -xvf tpu-mlir-resource.tar
mv regression/ tpu-mlir-resource/
```

建立 `model_yolov5s` 目录，并把模型文件和图片文件都放入 `model_yolov5s` 目录中，操作如下:

```bash
mkdir model_yolov5s && cd model_yolov5s
cp ../tpu-mlir-resource/model/yolov5s.onnx ./
cp -rf ../tpu-mlir-resource/dataset/COCO2017 ./
cp -rf ../tpu-mlir-resource/image ./
mkdir workspace && cd workspace
```

如果模型是图片输入，在转模型之前我们需要了解模型的预处理。如果模型用预处理后的npz文件做输入，则不需要考虑预处理。

官网yolov5的图片是rgb格式，每个值会乘以 `1/255` ，转换成mean和scale对应为 `0.0,0.0,0.0` 和 `0.0039216,0.0039216,0.0039216` 。

```bash
model_transform \
    --model_name yolov5s \
    --model_def ../yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names 350,498,646 \
    --test_input ../image/dog.jpg \
    --test_result yolov5s_top_outputs.npz \
    --mlir yolov5s.mlir
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625342813-7ce9408d-281e-47fe-8674-777167d314d4.png)

### 2.3 YOLOv5 浮点基线

以下每组命令结束后返回 `workspace`，避免继续在上一精度目录中嵌套建目录。将 MLIR 编译成 F32 BModel：

```bash
mkdir -p fp32
cp -rf yolov5s* fp32/
cd fp32

model_deploy.py \
    --mlir yolov5s.mlir \
    --quantize F32 \
    --processor bm1684x \
    --model yolov5s_1684x_f32.bmodel
cd ..
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625351817-bf6de628-f4a8-43db-9f09-2c8bb906955e.png)

将mlir文件转换成`f16`的bmodel，操作方法如下:

```bash
mkdir -p fp16
cp -rf yolov5s* fp16/
cd fp16

model_deploy \
    --mlir yolov5s.mlir \
    --quantize F16 \
    --processor bm1684x \
    --test_input yolov5s_in_f32.npz \
    --test_reference yolov5s_top_outputs.npz \
    --model yolov5s_1684x_f16.bmodel
cd ..
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625364778-be9976cb-6f35-452a-a8a3-fb137939af6a.png)

### 2.4 YOLOv5 INT8 校准

此处采用 INT8 对称量化流程，先通过 `run_calibration` 生成校准表。所谓“全 INT8”仍可能包含编译器保留的浮点算子；实际类型以生成的 TPU MLIR 和模型信息为准。下表保留原流程的校准选型参考，最终用目标数据验证。

| **场景** | **描述** | **量化速度** | **校准方法** | **推荐方法** |
| --- | --- | --- | --- | --- |
| case1 | 模型初次量化 | 不敏感 | 不清楚 | search_threshold |
| case2 | 模型初次量化 | / | 清楚 | cali_method直接选择对应校准方法 |
| case3 | 模型初次量化 | 敏感 | 不清楚 | cali_method选择固定校准方法,具体校准方法选择细节可看后续章节 |
| case4 | 模型量化后在bm1684处理器上部署精度无法满足需求 | / | / | 开启we和bc方法 |


对于校准方法的选择，可以参考以下几点经验性的结论：

 当模型是不带有attention结构的非transformer模型，可以选择 `use_mse` 校准方法。具体操作如下:

```bash
run_calibration yolov5s.mlir \
    --dataset ../COCO2017 \
    --input_num 100 \
    --cali_method use_mse \
    -o yolov5s_cali_table
```

或者也可选择默认的KLD校准方法。具体操作如下:

```bash
run_calibration yolov5s.mlir \
    --dataset ../COCO2017 \
    --input_num 100 \
    -o yolov5s_cali_table
```

编译 INT8 对称量化模型：

```bash
mkdir -p int8_sym
cp -rf yolov5s* int8_sym/
cd int8_sym

model_deploy.py \
    --mlir yolov5s.mlir \
    --quantize INT8 \
    --processor bm1684x \
    --calibration_table yolov5s_cali_table \
    --model yolov5s_bm1684x_int8_sym.bmodel
cd ..
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625445132-3c15ab35-83d6-499a-bd64-999568b50e55.png)

### 2.5 YOLOv5 混合精度

KL / MSE 校准仍不能满足精度要求时，再考虑混合精度。这里使用 `search_qtable`：比较候选阈值与层的量化敏感性，将部分层保留为浮点，并验证整体输出。`search_threshold` 处理阈值选择，`weight_equalization` / `bias_correction` 处理权重均衡和偏置修正，它们解决的问题不同。

`expected_cos=0.99` 是搜索时的输出比较目标；搜索具有候选范围和浮点层数约束，不能保证找到数学意义上的最少浮点层，也不能替代检测任务指标。需要比较特定后处理输出时，可通过版本支持的 `--post_process` 指定含 `PostProcess` 函数的脚本，下面命令未启用自定义后处理。

```bash
mkdir -p int8_mix
cp -rf yolov5s* int8_mix/
cd int8_mix

run_calibration.py yolov5s.mlir \
    --dataset ../../COCO2017 \
    --input_num 100 \
    --inference_num 30 \
    --expected_cos 0.99 \
    --quantize_method_list KL,MSE \
    --search search_qtable \
    --processor bm1684x \
    --quantize_table yolov5s_qtable \
    --calibration_table yolov5s_cali_table
```

日志记录不同阈值方法、层敏感性和最终整体输出比较结果。默认 cosine 搜索路径中的 loss 为 `1 - cosine`；也应记录是否使用其他比较方式。可依据日志调整浮点层表，再重新验证完整模型。

搜索会生成或更新校准表与混合精度表。该敏感层搜索实现将更新的阈值保存为 `new_cali_table.txt`；不同版本及提前退出路径可能不同，部署前检查实际产物。若搜索日志提示全 INT8 已达标、无需混精，则使用前面的 INT8 产物，不继续执行以下混精部署命令。需要混精且两个文件均已生成时，使用 `new_cali_table.txt` 与前面指定的 `yolov5s_qtable`：

```bash
model_deploy \
    --mlir yolov5s.mlir \
    --quantize INT8 \
    --processor bm1684x \
    --calibration_table new_cali_table.txt \
    --quantize_table yolov5s_qtable \
    --model yolov5s_bm1684x_int8_mix.bmodel
cd ..
```

## 三、例程构建与运行
C++ SoC 交叉编译在 `Sophon_Cross_Dev` 中进行，生成的程序放到目标设备运行。Qwen3-VL 的下述 LLM-TPU pybind11/C++ 命令沿用原记录的设备端构建方式，不要直接拿 x86 构建结果到 ARM64 设备运行。Python 脚本及设备测试也在目标设备执行。

### 3.1 Qwen3-VL 例程
+ **sophon-demo**  `sample/Qwen3-VL`的python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

```bash
usage: qwen3_vl.py [-h] -m MODEL_PATH [-c CONFIG_PATH] [-vr VIDEO_RATIO] [-d DEVID] [-ll {DEBUG,INFO,WARNING,ERROR}] [--do_sample]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        path to the processor file
  -vr VIDEO_RATIO, --video_ratio VIDEO_RATIO
                        Set video ratio, default is 0.25
  -d DEVID, --devid DEVID
                        device ID to use
  -ll {DEBUG,INFO,WARNING,ERROR}, --log_level {DEBUG,INFO,WARNING,ERROR}
                        log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]
  --do_sample           if set, generate tokens by sample parameters
```

  安装依赖：

```bash
pip3 install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m dfss --url=open@sophgo.com:tools/silk2/silk2.tools.logger-1.0.2-py3-none-any.whl
pip3 install silk2.tools.logger-1.0.2-py3-none-any.whl --force-reinstall
rm -f silk2.tools.logger-1.0.2-py3-none-any.whl
```

  输入`../datasets/images/test.jpg`测试图片，测试问题为："请描述图片中的内容"，测试命令如下:

```bash
python qwen3_vl.py -m /data/qwen3vl_4b/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251208_103216.bmodel -c /data/qwen3vl_4b/config/ -d 0

python qwen3_vl.py -m /data/qwen3vl_2b/qwen3-vl-2b-instruct_w8bf16_seq2048_bm1684x_1dev_20251208_145614.bmodel -c /data/qwen3vl_2b/config/ -d 0
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625480287-5be5c2bd-a78d-46da-b20b-6b6bf1101325.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625479207-7fc84fd5-0900-4064-bdcf-a77ae136fdf1.png)

+ **LLM-TPU**  `models/Qwen3_VL/python_demo`需要编译：

```bash
pip3 install torchvision transformers qwen_vl_utils
conda install -c conda-forge pybind11=2.12.0 -y

cd python_demo
mkdir build && cd build
# 用第一步得到的路径，手动指定pybind11_DIR
cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

# 编译并拷贝文件
make && cp *cpython* .. && cd ..

# run demo
python3 pipeline.py -m /data/qwen3vl_4b/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251208_103216.bmodel -c /data/qwen3vl_4b/config
python3 pipeline.py -m /data/qwen3vl_2b/qwen3-vl-2b-instruct_w8bf16_seq2048_bm1684x_1dev_20251208_145614.bmodel -c /data/qwen3vl_2b/config
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625561150-f76f9606-cc5a-45fa-b847-4f7af6d285a1.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625560866-a03272b9-207a-48ab-b3b8-1c84f52e3fe6.png)



  `models/Qwen3_VL/cpp_demo`需要编译：

```bash
conda install -c conda-forge opencv=4.5.5 -y
# 如自带/opt/sophon/sophon-opencv-latest，需要修改CMakeLists.txt中的这一行如下（需要补全相关sdk）：
set(SOPHON_OPENCV TRUE)

cd cpp_demo
mkdir build && cd build

# 显式指定 C/C++ 编译器、标准库路径、链接器参数
cmake .. \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,--no-as-needed -lstdc++ -Wl,-rpath=$CONDA_PREFIX/lib" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath=$CONDA_PREFIX/lib -lstdc++"
make && cp pipeline .. && cd ..

# run demo
./pipeline -m /data/qwen3vl_4b/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251208_103216.bmodel -c /data/qwen3vl_4b/config
./pipeline -m /data/qwen3vl_2b/qwen3-vl-2b-instruct_w8bf16_seq2048_bm1684x_1dev_20251208_145614.bmodel -c /data/qwen3vl_2b/config
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625622863-786338de-71b2-4d05-9901-a5055ba29ff7.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625622088-f4e89d48-fe2a-4aa4-a2e8-1c39f2397ae0.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625621846-c13ac20e-f0a5-426f-91cb-334b5acdeb31.png)

**封装为 HTTP 服务**

在这套 SoC 盒子的 Qwen3-VL 服务中，模型只加载一份，请求按顺序使用同一实例。HTTP 入口可以接收多个请求，但设备侧的生成过程串行执行，线程池大小不代表模型并行度。排队需要设置等待上限与超时，测试时区分排队时间和实际生成时间。

每次请求开始前重置会话状态，完整生成结束后再让下一个请求进入。流式响应要让锁覆盖生成器的整个消费过程；如果创建 `StreamingResponse` 后就释放锁，后续请求仍可能清空前一请求正在使用的缓存。多请求测试应检查输出隔离和等待时间，不能据此宣称设备支持并行推理。

### 3.2 YOLOv5 例程
+ **python**  `sample/YOLOv5/python_demo`不需要编译：  如果使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。还需要交叉编译安装sophon-sail，具体可参考交叉编译安装sophon-sail。  此外还需要配置opencv等其他第三方库：

```bash
pip install opencv-python-headless
```

> 注意上述命令安装的opencv是公版opencv，如果希望使用sophon-opencv，可以设置如下环境变量：
>
> 原记录中的 sophon-opencv Python 绑定要求 Python ≤3.8；这是当时 SDK 包的限制，换版本时检查所提供的绑定 ABI。前面的 Python 3.10 环境可以使用公版 OpenCV，不能默认共用该旧绑定。
>

```bash
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
```

  python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。yolov5_opencv.py和yolov5_bmcv.py的参数一致，以yolov5_opencv.py为例：

```bash
usage: yolov5_opencv.py [-h] [--input INPUT] [--bmodel BMODEL] [--dev_id DEV_ID] [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH] [--use_cpu_opt]

optional arguments:
  -h, --help            打印这个帮助日志然后退出
  --input INPUT         测试数据路径，可输入整个图片文件夹的路径或者视频路径
  --bmodel BMODEL       用于推理的bmodel路径，默认使用stage 0的网络进行推理
  --dev_id DEV_ID       用于推理的tpu设备id
  --conf_thresh CONF_THRESH
                        置信度阈值
  --nms_thresh NMS_THRESH
                        nms阈值
  --use_cpu_opt         开启cpu后处理优化
```

> 注意： CPP和python目前都默认关闭nms优化，python调用的优化接口，依赖3.7.0版本之后的sophon-sail，如果sophon-sail版本有该接口，可以添加参数--use_cpu_opt来开启该接口优化，use_cpu_opt仅限输出维度为5的模型(一般是3输出，别的输出个数可能需要用户自行修改后处理代码)。
>

  图片测试实例如下，支持对整个图片文件夹进行测试。

```bash
python python/yolov5_opencv.py --input datasets/test --bmodel models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5 --use_cpu_opt

python python/yolov5_bmcv.py --input datasets/test --bmodel models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5 --use_cpu_opt
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625666099-a20153fa-7a2f-4b07-bef7-02ea4b8d2485.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625666206-fab29566-2538-4bc8-9628-f6febe0f6b65.png)



  测试结束后，会将预测的图片保存在`results/images`下，预测的结果保存在`results/yolov5s_v6.1_3output_fp32_1b.bmodel_test_opencv_python_result.json`下，同时会打印预测结果、推理时间等信息。  视频测试实例如下，支持对视频流进行测试。

```bash
python3 python/yolov5_opencv.py --input datasets/test_car_person_1080P.mp4 --bmodel models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5 --use_cpu_opt

python3 python/yolov5_bmcv.py --input datasets/test_car_person_1080P.mp4 --bmodel models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5 --use_cpu_opt

python3 pythonv8/yolov8_bmcv.py --input datasets/test_car_person_1080P.mp4 --bmodel yolov8s-congestion_fp16_1b.bmodel --dev_id 0 --conf_thresh 0.5 --nms_thresh 0.5
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625700260-b0faf029-eb96-492f-9c58-3d460aa0c9e9.png)
![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625700011-a4430b18-dd97-45bc-b79e-abcbaf8a919b.png)



  测试结束后，会将预测的结果画在`results/test_car_person_1080P.avi`中，同时会打印预测结果、推理时间等信息。  `yolov5_bmcv.py`不会保存视频，而是会将预测结果画在图片上并保存在`results/images`中。  注意，riscv平台暂不支持用opencv进行视频测试，但是可以选择`yolov5_bmcv.py`测试。  `sample/YOLOv5/cpp_demo`需要编译：  如果使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。  通常在x86主机上交叉编译程序，需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体参考交叉编译环境搭建。主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。  交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

+ **cpp**  `sample/YOLOv5/cpp/yolov5_bmcv`需要编译：

```plain
# 1. 降级libdpkg-perl到匹配版本（允许降级）
apt install -y --allow-downgrades libdpkg-perl=1.19.7ubuntu3.2
# 2. 安装基础编译工具（允许降级）
apt install -y --allow-downgrades build-essential gcc=4:9.3.0-1ubuntu2 g++=4:9.3.0-1ubuntu
# 3. 降级binutils-common（允许降级）
apt install -y --allow-downgrades binutils-common=2.34-6ubuntu1.11
# 4. 安装ARM64交叉编译工具链
apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

cd cpp/yolov5_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/workspace/soc-sdk ..
make
```

  编译完成后，会在yolov5_bmcv目录下生成yolov5_bmcv.soc。

```bash
chmod +x yolov5_bmcv.soc
./yolov5_bmcv.soc --input=../../datasets/test --bmodel=../../models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel --dev_id=0 --conf_thresh=0.5 --nms_thresh=0.5 --classnames=../../datasets/coco.names --use_cpu_opt
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625729314-befd2583-5551-42d2-8df4-7052e2d4be85.png)


  `sample/YOLOv5/cpp/yolov5_sail`需要编译：  如果使用sophon-sail接口，需要参考交叉编译安装sophon-sail，给soc环境配置sophon-sail，然后进行如下步骤。

```plain
cd cpp/yolov5_sail
mkdir build && cd build
#请根据实际情况修改-DSDK和-DSAIL_PATH的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/workspace/soc-sdk -DSAIL_PATH=/workspace/sophon-sail/build_soc/sophon-sail ..
make
```

  编译完成后，会在yolov5_sail目录下生成yolov5_sail.soc。

```bash
chmod +x yolov5_sail.soc
./yolov5_sail.soc --input=../../datasets/test --bmodel=../../models/BM1684X/yolov5s_v6.1_3output_fp16_1b.bmodel --dev_id=0 --conf_thresh=0.5 --nms_thresh=0.5 --classnames=../../datasets/coco.names --use_cpu_opt
```


![](https://cdn.nlark.com/yuque/0/2026/png/65292342/1770625736329-5c2807e3-aa82-4988-9be7-c3a32350ffef.png)

## 四、验收

先确认 BModel 与大模型 config 都已生成，再在目标设备验证运行库和模型加载，最后用真实图片或视频检查有效输出与耗时。YOLO 需检查输出顺序、阈值、NMS 与坐标还原；Qwen3-VL 需检查 tokenizer/processor、视觉输入和序列长度是否匹配。

INT8 / 混合精度模型除编译器张量比较外，还应在固定评测数据上验证任务指标。将模型 hash、前处理、软件版本和测试口径与结果一起保存，避免把不同精度或不同输入条件下的数字放在同一张性能表中。


## 五、参考资料

- [TPU-MLIR：模型转换、校准与编译](https://github.com/sophgo/tpu-mlir)
- [LLM-TPU：大模型运行例程](https://github.com/sophgo/LLM-TPU)
- [Sophon Demo：CV 与多模态部署例程](https://github.com/sophgo/sophon-demo)
- [Sophon SAIL：Python/C++ 推理接口](https://github.com/sophgo/sophon-sail)
- [Qwen3-VL Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
