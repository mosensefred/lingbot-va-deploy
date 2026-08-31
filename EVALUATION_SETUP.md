# RoboTwin 评测环境搭建记录

> 状态：**进行中**（依赖已装，assets 下载中，待编译 pytorch3d/curobo + 跑评测）
> 目的：跑 LingBot-VA 阶段二训练结果在 RoboTwin 上的 success rate 评测

## 一、评测架构（server-client 分离）

```
server（lingbot 环境，torch 2.9.0 cu128）  ←──websocket──  client（robotwin 环境，torch 2.4.1）
  跑 LingBot-VA 推理，输出动作                              跑 RoboTwin 仿真（sapien Vulkan 渲染）
```

- server 端：`wan_va_server.py --config-name robotwin`（已就绪，用 lingbot 环境）
- client 端：`eval_polict_client_openpi.py`（需 robotwin 环境）

## 二、环境搭建步骤

### 已完成 ✅

1. **创建独立 conda 环境**（避免污染 lingbot）：
   ```bash
   conda create -n robotwin python=3.10 -y
   ```

2. **装 torch 2.4.1**（RoboTwin 要求，清华镜像）：
   ```bash
   pip install torch==2.4.1 torchvision --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **装 sapien 3.0.0b1**（预发布版，需 `--pre`）：
   ```bash
   pip install sapien==3.0.0b1 --pre --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **装其余依赖**（mplib 0.2.1 / open3d 0.18.0 / trimesh 4.4.3 / gymnasium 0.29.1 / scipy 1.10.1 等）：
   ```bash
   pip install -r script/requirements.txt --pre --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

5. **下载 assets**（后台进行中，13.9GB）：
   - `background_texture.zip` 10.22GB + `objects.zip` 3.48GB + `embodiments.zip` 0.20GB
   - 来源：HuggingFace `TianxingChen/RoboTwin2.0`

### 待完成 ⬜

6. 编译 pytorch3d（`pip install "git+...pytorch3d.git@stable"`）
7. 编译 curobo（git clone + `pip install -e . --no-build-isolation`）
8. 改 sapien/mplib 代码（`_install.sh` 里的 sed 命令）
9. assets 下载完成后 unzip + 配置路径
10. 跑评测（server + client）

## 三、关键版本

| 包 | 版本 |
|---|---|
| torch | 2.4.1（cu121） |
| sapien | 3.0.0b1 |
| mplib | 0.2.1 |
| open3d | 0.18.0 |
| numpy | 1.26.4 |

## 四、风险点

- **torch 2.4.1（cu121）在 Blackwell（sm_120）上 CUDA 不可用**——但 client 端主要做 Vulkan 渲染仿真，模型推理在 server 端，预计不影响，需实测确认。
