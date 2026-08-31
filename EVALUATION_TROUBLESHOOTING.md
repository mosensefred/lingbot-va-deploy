# RoboTwin 评测环境搭建踩坑实录（阶段性总结）

> 状态：**进行中** —— 依赖已装齐、assets 已就位、client 参数链路已打通，卡在 curobo v0.7.7 编译链（已推进到最后一步：链接）
> 定位：这是从「跑通一个项目」到「掌握这个技术栈」的关键工程积累。每一个坑都记录了现象→根因→修复，形成可复用的排查方法论。

---

## 一、总体架构认知（为什么这么复杂）

```
┌─────────────────────────────────┐          ┌─────────────────────────────────┐
│  Server（lingbot 环境）           │          │  Client（robotwin 环境）          │
│  torch 2.9.0 + cu128            │ websocket│  torch 2.4.1 + cu121            │
│  LingBot-VA 推理（GPU sm_120）    │◄────────►│  RoboTwin 仿真 + Vulkan 渲染     │
│  wan_va_server.py 监听 :29056    │  msgpack │  eval_polict_client_openpi.py   │
└─────────────────────────────────┘          └─────────────────────────────────┘
```

**根本矛盾**：一台 Blackwell（sm_120）GPU 的机器上，要同时跑两套互不兼容的 CUDA 生态——
- Server 要 torch 2.9.0+cu128（唯一支持 sm_120 的组合）
- Client 要 torch 2.4.1+cu121（RoboTwin 官方锁定版本，最高只支持 sm_90）

所以必须双环境隔离，且 client 环境的 CUDA 编译链在 Blackwell 机器上处处别扭。

---

## 二、踩坑全记录（按时间线）

### 坑 1：`multiprocessing.Pool` 多进程 hang（训练阶段，已解决 ✅）

- **现象**：训练在数据集构建阶段永久卡死，128 个 worker 全部 `futex_do_wait`
- **错误诊断路径**（这段弯路本身是教训）：
  - 第一次错误结论：「机械盘 IO 慢」→ iostat 实测 %util=1%，推翻
  - 第二次错误结论：「CUDA fork 后上下文损坏」→ 小规模复现全成功，推翻
  - **正确路径**：搜到官方 issue #32（`construct_lerobot_multi_processor hang`），多人复现
- **修复**：`multiprocessing.Pool` → `multiprocessing.pool.ThreadPool`（一行改动）
- **方法论**：遇到已知开源项目的问题，**先搜 issue 再自己复现**；下结论前必须拿证据

### 坑 2：pytorch3d 编译失败 `No module named 'torch'`

- **现象**：`pip install git+...pytorch3d` 报构建时找不到 torch
- **根因**：pip 默认 build isolation，隔离环境里没有已装好的 torch
- **修复**：加 `--no-build-isolation`
- **注意**：nohup 后台跑 pip 会丢 conda 环境，需 `conda run -n robotwin` 或显式 source

### 坑 3：sapien 3.0.0b1 报 `No module named 'pkg_resources'`

- **根因**：setuptools ≥81 移除了 `pkg_resources`，老代码还在用
- **修复**：`pip install "setuptools<81"`（装 80.10.2）
- **通用性**：这是 2025 年后所有老 Python 项目的通病，见到即降级 setuptools

### 坑 4：HF 下载 10GB 大文件 xet 协议 I/O error

- **现象**：`OSError: error decoding response body`，下载中断
- **根因**：huggingface_hub 新版默认走 xet 协议，大文件解码出错
- **修复**：`HF_HUB_DISABLE_XET=1` 强制传统 HTTP
- **附加教训**：tqdm 进度条显示 0% ≠ 没下载，要看进程 `/proc/<pid>/io` 的 write_bytes

### 坑 5：client 评测脚本参数全被 `error: unrecognized arguments` 拒绝

- **现象**：`--task_name` 等参数报 unrecognized
- **根因**：`eval_polict_client_openpi.py` 的 argparse 用 `nargs=argparse.REMAINDER` 定义 `--overrides`，**官方 launch_client.sh 把所有业务参数放在 `--overrides` 之后**，由脚本自己解析——直接传给 argparse 会被拒
- **修复**：按官方脚本顺序：`--config ... --port ... --test_num ... --overrides --task_name X --policy_name Y ...`
- **深层教训**：`--overrides` 之后的所有参数（含 save_root、guidance_scale）都从 config dict 读取，argparse 的同名参数是摆设

### 坑 6：curobo `CuroboPlanner` 导入失败（API 大版本不兼容）

- **现象**：`cannot import name 'CuroboPlanner'`，`curobo.types` is not a package
- **根因**：**git clone 默认拿到 main 分支 = curobov2 重构版**（`types.py` 单文件、`curobo/` 平铺），而 RoboTwin 用旧 API（`curobo.types.math` 子包、`src/curobo` 布局）
- **修复**：`git checkout v0.7.7` 后重装
- **方法论**：开源项目安装脚本里 `git clone` 不带版本号 = 隐雷。装完先 `python -c "from curobo.types.math import Pose"` 验证 API 布局

### 坑 7：CUDA 编译链五连坑（Blackwell 机器装 CUDA 12.1 工具链）

这是最密集的一串，本质都是「conda 生态装 CUDA toolkit 的布局漂移」：

| # | 报错 | 根因 | 修复 |
|---|---|---|---|
| 7a | `CUDA_HOME environment variable is not set` | 系统无 CUDA toolkit（只有驱动） | `conda install cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 -c nvidia` |
| 7b | `nv/target: 没有那个文件` | conda 把头文件放 `targets/x86_64-linux/include/`，nvcc 不知道 | `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include` |
| 7c | `thrust/complex.h: 没有那个文件` | thrust 藏在 `include/cccl/` 子目录 | CPATH 再加 `.../include/cccl` |
| 7d | `gcc versions later than 12 are not supported` | 系统 gcc 13.3，CUDA 12.1 的 nvcc 最高支持 gcc 12 | `conda install gcc_linux-64=12 gxx_linux-64=12 -c conda-forge`，配 `CC/CXX/CUDAHOSTCXX` |
| 7e | `cuda_fp4.hpp(1033): error: expected a ";"` | **cuda-cudart_linux-64=13.3 混进了 12.1 的环境**，fp4/fp6 头是 12.8+ 的产物，语法 12.1 编译器不认 | 确认 curobo 不引用后**直接删除** `cuda_fp4/fp6.h(pp)`（`_linux-64` 变体在 nvidia channel 无 12.1 版本可降） |
| 7f | `ld: cannot find -lcudart` | libcudart.so 在 `targets/x86_64-linux/lib/`，链接器没搜到 | `LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib`（当前正在验证） |

**核心认知**：conda 的 CUDA 包版本约束不严格——`cuda-nvcc=12.1` 会和 `cuda-cudart_linux-64=13.3` 共存，后者把新版头文件塞进 targets 目录，编译时才炸。**装完必须 `conda list | grep cuda` 全家核对版本一致性**。

### 坑 8（预警，未爆发）：torch 2.4.1 不支持 sm_120

- 编译日志已警告：`RTX PRO 6000 Blackwell (sm_120) is not compatible with the current PyTorch installation`
- client 端 torch 2.4.1+cu121 最高支持 sm_90
- **影响评估**：client 的 GPU 主要给 curobo 运动规划用。若跑评测时 curobo 报 CUDA 内核错误，需要应对（curobo 有 CPU 回退路径，或改在 server 端跑规划）
- 这是双环境架构的**结构性风险**，暂列观察项

---

## 三、已确认可行的完整搭建命令序列

```bash
# ========== 环境创建 ==========
conda create -n robotwin python=3.10 -y
conda activate robotwin

# ========== Python 依赖 ==========
pip install torch==2.4.1 torchvision --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install sapien==3.0.0b1 --pre --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r script/requirements.txt --pre --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "setuptools<81"          # 坑3：pkg_resources
pip install msgpack msgpack-numpy    # client websocket 通信

# ========== pytorch3d（坑2）==========
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# ========== curobo v0.7.7（坑6/7）==========
cd envs && git clone https://github.com/NVlabs/curobo.git
cd curobo && git checkout v0.7.7     # 关键！main 是不兼容的 curobov2

# CUDA 工具链（conda 版，无需 sudo）
conda install cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 -c nvidia -y
conda install gcc_linux-64=12 gxx_linux-64=12 -c conda-forge -y
conda install "cuda-cccl=12.1.*" -c nvidia -y   # cccl 降级（坑7c 部分）

# 混入的 13.3 头文件清理（坑7e，curobo 不用 fp4/fp6）
rm $CONDA_PREFIX/targets/x86_64-linux/include/cuda_fp4.{h,hpp}
rm $CONDA_PREFIX/targets/x86_64-linux/include/cuda_fp6.{h,hpp}

# 编译环境变量全家桶（坑7b/7c/7d/7f）
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl:$CPATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH

pip install -e . --no-build-isolation

# ========== sapien/mplib 补丁（官方 _install.sh 的 sed）==========
SAPIEN_LOCATION=$(pip show sapien | grep Location | awk '{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAPIEN_LOCATION/wrapper/urdf_loader.py
MPLIB_LOCATION=$(pip show mplib | grep Location | awk '{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPLIB_LOCATION/planner.py

# ========== assets（坑4）==========
cd assets && HF_HUB_DISABLE_XET=1 python _download.py
unzip background_texture.zip embodiments.zip objects.zip && rm *.zip
cd .. && python ./script/update_embodiment_config_path.py

# ========== 启动评测（坑5 的参数顺序）==========
# server（lingbot 环境）
python -m torch.distributed.run --nproc_per_node 1 --master_port 29061 \
    wan_va/wan_va_server.py --config-name robotwin --port 29056 --save_root visualization/
# client（robotwin 环境）——注意所有业务参数在 --overrides 之后
python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml --port 29056 --test_num 5 \
    --overrides --task_name adjust_bottle --task_config demo_clean \
    --policy_name ACT --save_root ./results --video_guidance_scale 5 --action_guidance_scale 1
```

---

## 四、方法论沉淀（专家视角的排查框架）

1. **先搜后试**：开源项目异常，官方 issue > 自己复现。本次 issue #32 直接省掉数小时
2. **证据先行**：每个结论要有测量支撑（iostat、/proc/io、conda list），「进度条 0%」这类 UI 信号不可信
3. **版本矩阵意识**：GPU 代际（sm_120）× torch 版本 × CUDA toolkit 版本 × gcc 版本，四者必须交叉核对，conda 不会替你保证一致性
4. **git clone 即版本**：安装脚本不 pin 版本 = 隐雷，装完先验证 API 布局
5. **报错位置即进度**：编译链报错从 preprocessing → compile → link 逐段推进，说明修复在收敛；同样报错反复出现说明没找对根因
6. **双环境隔离**：互斥依赖（cu128 vs cu121）必须环境级隔离，靠 websocket 等进程间通信解耦

---

## 五、当前状态与下一步

- [x] robotwin 环境创建 + 全部 Python 依赖
- [x] pytorch3d 0.7.8 编译通过
- [x] assets 14.9GB 下载解压 + 路径配置
- [x] sapien/mplib 官方补丁
- [x] client 参数链路打通（Config 已正确打印）
- [ ] **curobo v0.7.7 编译**（已到最后链接步，LIBRARY_PATH 修复验证中）
- [ ] client 冒烟测试（adjust_bottle × 5）
- [ ] 正式评测（50 任务 × 100 episodes）

**风险登记**：
- torch 2.4.1 与 sm_120 不兼容（坑8）：curobo 若需 GPU 可能要走 CPU 回退
