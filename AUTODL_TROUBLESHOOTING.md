# AutoDL 部署 LingBot-VA 踩坑实战记录

> 这份文档记录了在 AutoDL（RTX 4090 单卡）上从零部署 LingBot-VA RoboTwin 评测的**真实踩坑过程**，配合 [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) 和 [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) 使用。
>
> 官方文档只讲了「正常路径」，这里补上「哪里会翻车 + 怎么救」。

---

## 0. 最重要的结论先放前面

**AutoDL 部分区域/镜像不带 NVIDIA Vulkan 驱动，会导致 sapien 渲染直接失败（Render Error）。** 这是整个部署最容易翻车、且换镜像/换依赖都救不了的地方，务必在租卡后**第一时间验证**（见 §6）。

---

## 1. 租卡与磁盘：数据盘要够大、换区会丢数据

| 坑 | 现象 | 解法 |
|---|---|---|
| 数据盘太小 | 权重 23G + RoboTwin 资产 + 结果视频，50G 会爆盘 | 数据盘选 **≥100G**（实测 100G 装完剩 ~47G） |
| 系统盘太小 | conda 环境装系统盘（默认 30G）会溢出 | 把 conda env 和 pip 缓存指到数据盘（见下） |
| **换区丢数据盘** | bjb2 → bjb1 后数据盘被清空，权重/环境全没 | 换区前确认；数据盘**不跨区迁移** |

```bash
# 数据盘扩容后，把 conda 环境 + pip 缓存都放到数据盘
mkdir -p /root/autodl-tmp/conda_envs /root/autodl-tmp/pip_cache
conda config --add envs_dirs /root/autodl-tmp/conda_envs
pip config set global.cache-dir /root/autodl-tmp/pip_cache
```

> ⚠️ 注意：`conda activate` 报 `Run 'conda init' before 'conda activate'` 时，先 `conda init bash && source ~/.bashrc`。

---

## 2. Python 版本：必须建 3.10 环境

AutoDL 镜像默认 Python 是 3.12，但 RoboTwin 依赖（`sapien==3.0.0b1` / `open3d==0.18.0` / `scipy==1.10.1`）**只有 py3.10 轮子**，3.12 直接装不上。

```bash
conda create -n lingbot python=3.10 -y
conda activate lingbot
python -V   # 必须是 3.10.x
```

---

## 3. 学术加速代理：开关时机是最大的坑

`source /etc/network_turbo`（AutoDL 学术加速）**只加速 GitHub/HuggingFace，对 pip/pytorch 源反而更慢**，甚至会让 pip 从 aliyun 镜像拉到空页面报 `from versions: none`。

**核心原则：装 PyPI 包时关代理，拉 GitHub 时开代理。**

| 操作 | 代理状态 |
|---|---|
| 装 torch / pip 依赖 / ModelScope 下权重 | **关**（`unset http_proxy https_proxy all_proxy`）|
| git clone GitHub / pytorch3d 从 git 装 / curobo 从 git 装 | **开**（`source /etc/network_turbo`）|

---

## 4. torch 安装：用国内镜像，别用官方源

官方 `download.pytorch.org` 在国内慢到卡死。**阿里云镜像有 torch cu126 轮子**：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu126/
```

> 注意：aliyun 的 `pytorch-wheels` 页面是 JS 渲染的，**浏览器能看但 pip 解析不了**，报 `from versions: none` 是正常的 —— 真正可用的是**官方 `download.pytorch.org` 的 nginx 目录**（pip 能解析），只是慢。实测关代理后官方源能到 ~5.7MB/s，可接受。
>
> 修正：cu126 的 torch 用官方 `--index-url https://download.pytorch.org/whl/cu126`（关代理后速度尚可），其余小依赖走清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

验证：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.9.0+cu126 True
```

---

## 5. RoboTwin 依赖：一堆隐形坑

### 5.1 `_install.sh` 假成功

RoboTwin 的 `_install.sh` **没有 `set -e`**，里面 `pip install -r requirements.txt` 失败后脚本会继续跑、最后以 0 退出，显示「环境装好」其实是**假成功**。务必装完手动验证每个包。

### 5.2 `sapien==3.0.0b1` 是预发布版

```bash
pip install "sapien==3.0.0b1" --pre   # 必须加 --pre，否则找不到
```

### 5.3 sapien import 报 `No module named 'pkg_resources'`

`sapien 3.0.0b1` 是老包，硬 `import pkg_resources`。新 setuptools（83+）删了这个模块，**降级 setuptools**：

```bash
pip install "setuptools==69.5.1"   # 还带 pkg_resources 的版本
```

### 5.4 pytorch3d 编译：`--no-deps` 绕依赖解析

pytorch3d 从 git 装时，即使 iopath 已装，pip 仍会查镜像验证依赖（代理开着→空页面→报错）。**手动装依赖 + `--no-deps` 跳过**：

```bash
# 1. 关代理装依赖
pip install iopath fvcore
# 2. 手动 clone（比 pip 的 git 子进程更稳、可重试）
git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && git checkout stable
# 3. 本地装，跳过依赖检查
pip install -e . --no-build-isolation --no-deps
```

> pytorch3d 编译 15–30 分钟，**必须用 screen 跑**，否则 SSH 一断就白编。

### 5.5 curobo 依赖经 `nvidia-curobo`

curobo 的 `pip install -e .` 会拉一堆依赖（含 `nvidia-curobo`），同样要**关代理**装。

---

## 6. 🔴 致命坑：NVIDIA Vulkan 驱动缺失（Render Error）

这是整个部署最核心、最影响结果的问题。

### 现象

起客户端跑评测时，sapien 渲染初始化报：

```
Render Error
```

或手动测渲染时：

```
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent
```

### 根因

AutoDL 部分区域/镜像的容器里，NVIDIA 驱动**只挂载了 CUDA/GL/EGL 库，没有挂载 Vulkan ICD 库**（`libvulkan_nvidia.so`）。

验证（**租卡后第一件事就跑这个**）：

```bash
find / -name "libvulkan_nvidia*" 2>/dev/null
vulkaninfo --summary 2>&1 | grep -iE "deviceName|deviceType"
```

- 有 `libvulkan_nvidia.so` + vulkaninfo 列出 `NVIDIA GeForce RTX 4090` → ✅ GPU Vulkan 可用
- 只有 `llvmpipe (LLVM 15.0.7)` + 找不到 `libvulkan_nvidia.so` → ❌ 无 GPU Vulkan

### 走过的弯路（都没解决，记录供参考）

1. **换镜像**：驱动是宿主机统一挂载的，换镜像（换系统盘）不改变驱动挂载，没用
2. **换区域**：实测 bjb1、bjb2 两个区都无 NVIDIA Vulkan
3. **补装 `libnvidia-gl-580`**：包里有 `libGLX_nvidia.so`（含 `vk_icdGetInstanceProcAddr` 符号），但版本 580.178.04 与宿主内核 580.105.08 不匹配，报 `ERROR_INCOMPATIBLE_DRIVER`；且 dpkg 解压时报 `Invalid cross-device link`（`/usr/share/egl` 跨设备挂载）
4. **软件渲染 llvmpipe**：LLVM 15.0.7 太老，缺 sapien 需要的 Vulkan 扩展，报 `ErrorExtensionNotPresent`；kisak PPA 对 jammy 也没提供新版 Mesa

### 正解

**联系 AutoDL 客服，明确要求「带 NVIDIA Vulkan 驱动（`libvulkan_nvidia.so`）的实例/区域/镜像」**，用于跑 SAPIEN 渲染。这是唯一治本的路。

---

## 7. 服务端 `flash_attn` 硬导入（修改源码）

服务端 `wan_va/modules/model.py` 顶层硬导入 flash_attn，即使 `attn_mode="torch"` 也会执行：

```python
try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func   # ← 这里会 ModuleNotFoundError
```

但看代码逻辑，`flash_attn_func` **只在 `attn_mode='flashattn'` 时被用到**，`attn_mode='torch'` 走 `custom_sdpa`。所以改 import 为「找不到就置 None」即可绕过（省去编译 flash-attn 的几十分钟）：

```python
try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        flash_attn_func = None
```

---

## 8. 部署脚本的 sudo 问题

AutoDL 容器是 root 用户，**没有 `sudo`**。`cloud_robotwin_eval.sh` 里的 `sudo apt ...` 会报 `sudo: command not found`：

```bash
sed -i 's/sudo //g' cloud_robotwin_eval.sh
```

---

## 9. 长任务务必用 screen

pytorch3d 编译、模型加载、100 trial 评测都是长任务，SSH 一断前台命令就死。养成习惯：

```bash
screen -S <name>      # 进去跑长任务
# Ctrl+A 然后 D 切出来（detach）
screen -ls            # 看有哪些 screen
screen -r <name>      # 重新接回
```

> 教训：中途多次 SSH 断线（`Broken pipe`），前台跑的 pip install 反复被杀，浪费了大量时间。

---

## 10. 一键验证清单（部署完必跑）

```bash
conda activate lingbot
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
python -c "import sapien; print('sapien', sapien.__version__)"
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"
python -c "import curobo; print('curobo ok')"
python -c "import mplib; print('mplib ok')"
python -c "import open3d; print('open3d ok')"
python -c "import cv2; print('cv2', cv2.__version__)"
find / -name "libvulkan_nvidia*" 2>/dev/null   # 必须非空，否则 Vulkan 渲染不可用
```

---

## 附：本次实战环境快照

- 平台：AutoDL，RTX 4090（24GB），128 核 CPU，1TB 内存
- 驱动：NVIDIA 580.105.08 / CUDA 13.0
- Python：3.10.20（conda `lingbot` 环境，位于数据盘）
- torch：2.9.0+cu126
- 关键依赖版本：sapien 3.0.0b1 / pytorch3d 0.7.8 / open3d 0.18.0 / scipy 1.10.1 / curobo v0.7.8 / setuptools 69.5.1
- 结果：环境与依赖全部装通，服务端正常监听 29056，客户端因缺 NVIDIA Vulkan 驱动报 Render Error（待换带 Vulkan 的实例解决）
