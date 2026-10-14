---
title: Qualcomm QCS8550 固件编译与烧录
created: 2026-03-24
updated: 2026-03-25
status: draft
blog: false
tags: [Qualcomm, QCS8550, 固件, Yocto]
---

# Qualcomm QCS8550 固件编译与烧录

QCS8550 的 Linux 固件通常来自设备厂商提供的 BSP，内核、Yocto 配置、预编译组件和烧录分区必须属于同一发布组合。本文整理厂商 QCS8550.LE.1.0、kalama 平台的构建与 QFIL 烧录流程。源码 manifest、预编译组件、打包脚本和基础固件包需要从设备厂商取得。

以下命令与厂商编译说明及打包脚本对应，但尚未完成从源码构建到目标设备烧录的完整实测。它们适用于这套 BSP，换设备或 release 时需要重新核对目录、分区和工具版本。

# 一、确认主机与 BSP

这套 BSP 使用 x86_64 构建主机，厂商要求 Ubuntu 20.04、Python 3.6 及以上、64 GB 以上内存，并安装指定的交叉编译工具和依赖。DTC 按厂商说明从源码通过 `make` 构建；额外引入 Meson 时需先确认对应 DTC 版本的要求。

使用容器时，锁定基础镜像和依赖，源码、下载缓存和产物分别挂载。安装系统依赖可以使用 root，Yocto 构建使用普通用户，并保证该用户可写源码和构建目录。

```bash
uname -m
df -h .
free -h
git --version
repo version
```

主机和工具二进制架构必须一致。ARM64 训练服务器能运行 Linux，但无法直接执行 BSP 中的 x86_64 预编译工具；容器不会改变 CPU 指令集。构建并行度则依据实际内存调整，内存不足时应先降低并行度。

# 二、同步源码与固定版本

使用厂商授权的 manifest 和独立 SSH 凭据同步 BSP。命令参数中的三个变量代表用户自行配置的地址、分支和 manifest，不是公开下载入口：

```bash
repo init -u "$BSP_MANIFEST_URL" \
  -b "$BSP_BRANCH" -m "$BSP_MANIFEST_FILE"
repo sync -j8
repo manifest -r -o manifest.lock.xml
```

归档 `manifest.lock.xml`、厂商 release 标识、预编译包哈希和构建镜像 digest。不要把整个个人 SSH 目录放进发布包；构建产物也不需要携带源码访问凭据。

# 三、构建内核与系统

以下路径对应归档中的 kalama BSP，其他 release 应以其说明为准：

```bash
export BSP_ROOT="$PWD"
cd "$BSP_ROOT/src/kernel-5.15/kernel_platform"
bash qcom/proprietary/prebuilt_HY11/vendorsetup.sh

export DTS_VENDOR="$BSP_ROOT/src/kernel-5.15/kernel_platform/msm-kernel/arch/arm64/boot/dts/vendor"
if [ ! -e "$DTS_VENDOR" ] && [ ! -L "$DTS_VENDOR" ]; then
  ln -s "$BSP_ROOT/src/kernel-5.15/kernel_platform/qcom/proprietary/devicetree" "$DTS_VENDOR"
fi
ls -ld "$DTS_VENDOR"

BUILD_CONFIG=msm-kernel/build.config.msm.kalama.le \
VARIANT=consolidate \
OUT_DIR=../out/msm-kernel-kalama_le-consolidate \
./build/build.sh -j16

cd "$BSP_ROOT"
export SHELL=/bin/bash
export MACHINE=kalama
export DISTRO=qti-distro-rb-debug
source poky/qti-conf/set_bb_env.sh
export PREBUILT_SRC_DIR="$BSP_ROOT/prebuilt_HY11"
bitbake linux-msm
bitbake qti-robotics-image
```

内核构建依赖设备树与 vendor prebuilt。这里的设备树链接对应该 BSP 的目录布局，已有目录或链接时先检查其内容与指向。原指南用 `-f` 判断 `vendor` 是否存在，但它可能是目录或目录链接，因此示例改为分别检查路径与符号链接，避免反复创建链接。

`source set_bb_env.sh` 会设置 BitBake 环境并切换构建目录，后续的 `bitbake` 命令在初始化后的目录执行。系统构建失败时查看具体 recipe 的 `log.do_*`，总 console 日志用于找到失败任务。

# 四、生成镜像并组合烧录包

将同版本的 `multi_image_out_script.zip` 解压到 BSP 根目录，从该目录运行：

```bash
cd "$BSP_ROOT"
unzip multi_image_out_script.zip
./multi_image_out.sh qfil
```

脚本从 `build-qti-distro-rb-debug/tmp-glibc/deploy/images/kalama/qti-robotics-image` 读取构建产物，生成 `image_out_qfil`。其中包含应用处理器侧镜像、稀疏镜像展开结果和部分分区 XML，**它不是独立的完整烧录包**。

按厂商流程，先复制一份与该 BSP 匹配的完整基础固件包，再将 `image_out_qfil` 中的文件按原文件名覆盖进去，保留基础包中的 programmer、其他分区镜像和 patch XML。烧录时选择这个组合后的目录。仅拿脚本输出目录进入 QFIL，会缺少基础包提供的文件。

组合后检查 `rawprogram*.xml` 引用的镜像都存在，并核对 programmer 与存储类型。不要仅凭文件名相同，把其他 release 的基础包与当前构建产物混用。脚本会重建 `image_out_qfil` 目录，需保留的历史产物先另存。

# 五、QFIL 烧录与启动检查

在 Windows 上安装厂商随包提供的 QPST/QFIL 与设备驱动。该 SOP 的配置是 Flat Build、UFS，并启用 `Reset After Download`；截图中的 `Erase All Before Download` 未勾选，普通更新不额外开启全擦除。烧录前备份需要保留的数据。

设备能通过 ADB 访问时，可以用 `adb reboot edl` 进入 EDL，再确认设备管理器中的 QDLoader 9008 端口。SOP 也列出了 901D 端口选择方式，使用时需符合对应驱动和工具配置；无法启动的设备按板卡手册进入下载模式。

从组合后的完整固件目录选择 `xbl_s_devprg_ns.melf`，再通过 `Load XML` 加载该包对应的 rawprogram 与 patch 文件。检查端口、文件目录和 UFS 配置后点击 `Download`，以工具的成功日志判断下载结果。

烧录成功后，重启并检查：

```bash
cat /etc/os-release
uname -a
lsblk
dmesg | tail -80
```

版本检查使用 Linux 系统信息与 BSP 自身的版本标识。Android 的 `getprop ro.build.*` 不能作为这套 Linux 镜像的通用检查命令。随后按设备用途测试网络、存储和推理所需的 FastRPC/CDSP、OpenCL 路径，确认用户态库与固件能够配合运行。

# 六、参考资料

- [Yocto Project 文档](https://docs.yoctoproject.org/)
- [repo 工具](https://gerrit.googlesource.com/git-repo/)
- [Qualcomm QCS8550.LE.1.0 构建与 SDK 示例](https://github.com/qualcomm/qidk/blob/master/Tools/snpe-helper/README_LE_1_0.md)

