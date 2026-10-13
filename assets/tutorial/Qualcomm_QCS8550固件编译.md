---
title: Qualcomm QCS8550 固件编译与烧录
created: 2026-03-24
updated: 2026-03-25
status: draft
blog: false
tags: [Qualcomm, QCS8550, 固件, Yocto]
---

# Qualcomm QCS8550 固件编译与烧录

QCS8550 的 Linux 固件通常来自设备厂商提供的 BSP，内核、Yocto 配置、预编译组件和烧录分区必须属于同一发布组合。本文保留归档 BSP 的构建顺序，去掉私有源码入口与账号；厂商 manifest、预编译包及烧录 SOP 需要自行取得。

# 一、确认主机与 BSP

使用 x86_64 Linux 构建主机，准备 Bash、Git、repo、编译工具和 BSP 指定的 Python、Meson、DTC。构建镜像锁定基础系统和依赖，源码、下载缓存和产物分别挂载。

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

内核构建依赖设备树与 vendor prebuilt。符号链接缺失时先核对 BSP 目录布局，不应通过猜测路径补齐。系统构建失败时查看具体 recipe 的 `log.do_*`，总 console 日志只用于找到失败任务。

# 四、生成烧录包

使用同版本厂商打包脚本生成 QFIL 包。归档材料中的入口为：

```bash
./multi_image_out.sh qfil
```

检查产物包含匹配的 programmer、分区 XML 和镜像。不能把不同 BSP 的分区配置与新内核随意组合。烧录前备份需要保留的数据，并确认设备型号、存储类型及恢复方式。

# 五、烧录与验收

在 Windows 上安装设备 SOP 指定的 QPST/QFIL 与诊断驱动，将设备进入 EDL，确认正确的 QDLoader 端口。选择 Flat Build，加载本包 programmer、rawprogram 与 patch XML；QCS8550 归档设备使用 UFS，具体选择仍以硬件说明为准。

烧录成功后，重启并检查：

```bash
cat /etc/os-release
uname -a
lsblk
dmesg | tail -80
```

再分别验证网络、存储、FastRPC/CDSP 和 OpenCL。系统可启动只完成基础验收；推理后端用户态库与固件是否匹配，需要运行实际设备探针。

# 六、参考资料

- [Yocto Project 文档](https://docs.yoctoproject.org/)
- [repo 工具](https://gerrit.googlesource.com/git-repo/)

