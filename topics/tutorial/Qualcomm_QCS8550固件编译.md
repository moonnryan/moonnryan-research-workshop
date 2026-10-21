---
title: Qualcomm QCS8550 固件编译与烧录
created: 2026-03-24
updated: 2026-03-25
status: ready
blog: true
tags: [Qualcomm, QCS8550, 固件, Yocto]
aliases: [/assets/tutorial/Qualcomm_QCS8550固件编译.html]
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

这里需要一起固定的还有基础固件包和打包工具。manifest 只记录源码仓库的提交，不能说明 `prebuilt_HY11`、programmer 和分区 XML 来自哪个版本。源码相同而这些文件不同，最后得到的烧录包也可能不同。

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

## 确认系统镜像用了哪份内核

手工内核构建成功，与内核被装入系统镜像是两件事。前面的输出目录和 Yocto 的部署目录不同，需要检查 `linux-msm` recipe 如何引用源码、预编译内核和设备树，不能仅凭两次命令都成功就认为它们已经衔接。

在初始化后的 BitBake 环境中导出最终配置：

```bash
bitbake -e linux-msm > linux-msm.env.txt
grep -E '^(S|B|WORKDIR|DEPLOY_DIR_IMAGE|PREBUILT_SRC_DIR)=' linux-msm.env.txt
```

根据实际 `WORKDIR` 查找 `temp/log.do_compile`、`log.do_install` 和 `log.do_deploy`，核对使用的输入与复制目标。部署目录中的镜像也可能是符号链接，要检查最终指向的文件。改过内核却仍启动旧版本时，先沿这条路径追踪，再决定是否重新执行相关任务。

Yocto 的下载缓存、共享状态缓存与 recipe 工作目录用途不同。`clean`、`cleansstate`、`cleanall` 的清理范围也不同；后者还会删除下载文件。排查单个 recipe 时先保留日志和配置，不要一开始就清空整个构建目录，否则既增加重建时间，也丢失了判断旧产物来自哪里的依据。

# 四、生成镜像并组合烧录包

将同版本的 `multi_image_out_script.zip` 解压到 BSP 根目录，从该目录运行：

```bash
cd "$BSP_ROOT"
unzip multi_image_out_script.zip
./multi_image_out.sh qfil
```

脚本从 `build-qti-distro-rb-debug/tmp-glibc/deploy/images/kalama/qti-robotics-image` 读取构建产物，生成 `image_out_qfil`。其中包含应用处理器侧镜像、稀疏镜像展开结果和部分分区 XML，**它不是独立的完整烧录包**。

这个输入路径在归档脚本中是写死的。修改过 `DISTRO`、构建目录或镜像名称后，需要同步核对脚本；否则可能读不到产物，也可能读到此前留在旧目录中的镜像。打包前应确认输入文件的实际路径、时间和哈希与本次构建对应。

## 打包工具还有单独的 Python 依赖

归档中的 `multi_image_out.sh` 通过 `python` 调用 `checksparse.py`，而这份辅助脚本使用 Python 2 的 `print` 和异常捕获语法。厂商编译说明中的 Python 3.6 及以上要求，不能直接套到这份旧打包工具上。

```bash
command -v python
python --version
```

如果 `python` 指向 Python 3，这份脚本会在解析语法时失败。应优先向厂商取得与 BSP 配套的更新工具；必须使用旧工具时，在隔离环境中明确指定兼容解释器，不要为了打包去替换整台主机的默认 Python。若自行迁移到 Python 3，还要核对二进制读写、整数除法和生成的 XML，不能只改到没有语法错误。

归档的外层 shell 脚本没有启用失败即退出，若干 Python、复制操作后也没有检查返回值。终端最后出现完成提示，仍可能留下缺失镜像的目录。应检查稀疏镜像转换日志和实际输出，而不只看外层脚本是否正常结束。

## 镜像与分区 XML 必须一起使用

按厂商流程，先复制一份与该 BSP 匹配的完整基础固件包，再将 `image_out_qfil` 中的文件按原文件名覆盖进去，保留基础包中的 programmer、其他分区镜像和 patch XML。烧录时选择这个组合后的目录。仅拿脚本输出目录进入 QFIL，会缺少基础包提供的文件。

组合后检查 `rawprogram*.xml` 引用的镜像都存在，并核对 programmer 与存储类型。不要仅凭文件名相同，把其他 release 的基础包与当前构建产物混用。脚本会重建 `image_out_qfil` 目录，需保留的历史产物先另存。

稀疏镜像转换会生成分块文件及新的 rawprogram XML，XML 记录这些文件对应的扇区位置。只复制分块镜像而沿用旧 XML，或只更新 XML 而漏掉分块文件，都不构成可用的烧录输入。patch XML 也要保留原包的配套关系，不能从另一个版本借来补齐文件数量。

下面的只读检查用于发现 XML 引用的缺失文件，参数替换为组合后的固件目录：

```bash
python3 - /path/to/firmware-package <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1]).resolve()
xmls = sorted(root.glob('rawprogram*.xml'))
if not xmls:
    raise SystemExit('未找到 rawprogram XML')
missing = []
for xml in xmls:
    for entry in ET.parse(xml).iter('program'):
        name = entry.get('filename', '').strip()
        if name and not (root / name.replace('\\', '/')).is_file():
            missing.append(f'{xml.name}: {name}')
if missing:
    raise SystemExit('\n'.join(missing))
print('rawprogram 引用文件存在；仍需核对分区布局与版本')
PY
```

文件存在只是第一步，还需确认板型、存储配置及分区布局一致。归档脚本会把 `userStorge.img`、`oem_config.img`、`factory_1.img` 等部分条目的 `filename` 置空。这能说明脚本有意跳过这些条目的镜像写入，但不能据此保证保留所有用户数据：其他 rawprogram 条目、patch 和工具的擦除设置仍会影响最终结果。

# 五、QFIL 烧录与启动检查

在 Windows 上安装厂商随包提供的 QPST/QFIL 与设备驱动。该 SOP 的配置是 Flat Build、UFS，并启用 `Reset After Download`；截图中的 `Erase All Before Download` 未勾选，普通更新不额外开启全擦除。烧录前备份需要保留的数据。

设备能通过 ADB 访问时，可以用 `adb reboot edl` 进入 EDL，再确认设备管理器中的 QDLoader 9008 端口。SOP 也列出了 901D 端口选择方式，使用时需符合对应驱动和工具配置；无法启动的设备按板卡手册进入下载模式。

从组合后的完整固件目录选择 `xbl_s_devprg_ns.melf`，再通过 `Load XML` 加载该包对应的 rawprogram 与 patch 文件。检查端口、文件目录和 UFS 配置后点击 `Download`，以工具的成功日志判断下载结果。

发生失败时，按下载阶段缩小范围：

| 失败位置 | 优先核对 |
| --- | --- |
| 没有可用下载端口 | 板卡下载模式、USB 连接和驱动 |
| programmer 传输或握手失败 | programmer 与设备、安全配置是否匹配，以及 USB 是否稳定 |
| 已进入分区写入后失败 | 日志指向的镜像是否缺失、XML 与存储布局是否匹配 |
| 下载成功但无法正常启动 | 启动日志、内核与设备树、rootfs 及基础固件的版本组合 |

这些是检查顺序，具体原因仍以失败日志为准。Qualcomm 公开的 QCS8550.LE.1.0 示例还提供了 fastboot 更新流程；它与板厂的 QFIL 整包流程不是同一套操作，不能直接混用其中的分区命令和文件。

烧录成功后，重启并检查：

```bash
cat /etc/os-release
uname -a
lsblk
dmesg | tail -80
```

版本检查使用 Linux 系统信息与 BSP 自身的版本标识。Android 的 `getprop ro.build.*` 不能作为这套 Linux 镜像的通用检查命令。随后按设备用途测试网络、存储和推理所需的 FastRPC/CDSP、OpenCL 路径，确认用户态库与固件能够配合运行。

保存实际烧录包的哈希、QFIL 日志和启动后的版本信息。能进入系统说明启动链路基本可用，仍不能替代加速器调用测试；系统镜像更新后，用户态 SDK、驱动与 DSP 固件是否匹配，需要通过真实调用确认。

系统启动后若要继续部署 HTP 模型，先检查 FastRPC、DSP 库与用户态运行库，依赖分层和排查方法见 [Qualcomm QNN 环境部署与 ABI 排查](Qualcomm_QNN依赖与ABI.md)。

# 六、参考资料

- [Yocto 内核构建与调试](https://docs.yoctoproject.org/4.0.25/kernel-dev/common.html)
- [Yocto 任务与缓存清理](https://docs.yoctoproject.org/4.0.25/ref-manual/tasks.html)
- [repo 工具](https://gerrit.googlesource.com/git-repo/)
- [Qualcomm QCS8550.LE.1.0 构建与 SDK 示例](https://github.com/qualcomm/qidk/blob/master/Tools/snpe-helper/README_LE_1_0.md)

板卡专属的目录、QFIL 配置与打包行为依据本地归档的厂商编译说明、下载 SOP 和配套脚本整理；公开文档用于解释通用机制，不替代该板卡对应 release 的说明。

