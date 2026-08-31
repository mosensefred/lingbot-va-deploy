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

---

## 六、追加：坑 8 完整爆发与最终解法（2026-08-31）

### 现象
CPU 回退补丁（改 `TensorDeviceType` 默认 device + planner.py 去 `.cuda()`）**失败**——curobo 的运动学/碰撞内核是纯 CUDA 扩展（`kinematics_fused_kernel.cu`），`KinematicsFusedFunction` 硬性断言 `joint_vec.is_cuda()`，**没有 CPU 代码路径**。CPU 张量直接 INTERNAL ASSERT 崩溃，且该异常发生在 `CuroboPlanner.__init__` 的 warmup 里，导致 `left_planner` 属性从未赋值，后续 reset 报出误导性的 `AttributeError: no attribute 'left_planner'`（真实根因被异常链掩盖）。

### 结构性认知（重要）
```
torch 2.4.1+cu121（Python 层运行时）──能加载──► 自定义 CUDA 扩展（.so 自带 fatbin）
        │                                              │
        └─ 自身内核只编到 sm_90                          └─ 只要重编译时编入 sm_120，
           → torch.XXX CUDA 算子会炸                      就能在 Blackwell 上跑
```
**关键洞察**：`torch 2.4.1` 的限制只覆盖它**自带**的 CUDA 算子；pip 装的自定义扩展（curobo 的 kinematics/collision kernel）是**独立编译的 .so**，用什么 `-gencode` 编就在什么 GPU 上跑。所以不需要换 torch 版本，只需要**用支持 sm_120 的 nvcc 重编译 curobo**。

### 最终解法
```bash
# 1. 升级 nvcc 到 12.8（支持 compute_120；注意先前的 12.1 连 sm_100 都不认识）
conda install -n robotwin cuda-nvcc=12.8.93 -c nvidia -y

# 2. 显式指定编译架构（否则 torch cpp_extension 按可见 GPU 自动探测，torch2.4 不认识 sm_120 会编错/漏编）
export TORCH_CUDA_ARCH_LIST="12.0;9.0"

# 3. 重编译（其余环境变量同坑 7 全家桶）
pip uninstall -y nvidia_curobo
pip install -e . --no-build-isolation
```

### 已撤销的中间方案
- `curobo/types/base.py` 的 TensorDeviceType CPU 自动回退 → **已还原**（curobo 无 CPU 路径，回退必炸）
- `planner.py` 传 `tensor_args=CPU` → **已还原**（同上）
- `planner.py` 的 3 处 `.cuda()` 改为 `.to(tensor_args.device)` → **保留**（无害且更通用）

### 教训
1. 报错位置 ≠ 根因位置：`AttributeError left_planner` 是 warmup 崩溃的**下游**，必须读完整异常链（第一个 Traceback 才是根因）
2. 评估「绕过」方案前先确认目标库有没有对应代码路径（grep `is_cuda` 断言/`device='cpu'` 支持与否，五分钟的事）
3. torch 版本限制的是自己的算子，不是第三方扩展的 fatbin——这个边界想清楚，解法自然出现

### 坑 8 追加：TORCH_CUDA_ARCH_LIST="12.0" 也会被 torch 2.4 拒绝

- **现象**：`ValueError: Unknown CUDA arch (12.0) or GPU not supported`——arch 字符串到 `-gencode` 的映射表在 `torch/utils/cpp_extension.py` 里，torch 2.4 的表只到 9.0
- **修复**：绕过 torch 的映射，直接在 **curobo 的 setup.py** 的 `extra_cuda_args["nvcc"]` 里加 `"-gencode=arch=compute_120,code=sm_120"`（nvcc 12.8 原生认识），同时 `TORCH_CUDA_ARCH_LIST="9.0"` 让 torch 自己那部分照常生成
- **认知**：`TORCH_CUDA_ARCH_LIST` 不是透传字符串，是查表——表太老就拒绝新 arch；gencode 直传 setup.py 才是透传

### 坑 8 终局：curobo 编过 sm_120 后，torch 自带算子仍然炸——必须升 torch

- **现象**：curobo 以 gencode sm_120 重编译成功（`nvidia_curobo-0.7.7.post1.dev0`），但导入时仍报 `no kernel image available`——这次炸点在 **torch 自带的 `torch.sign`/`torch.where`**（`normalize_quaternion` TorchScript 里调用）
- **实测确认**：`torch.cuda.get_arch_list()` = `['sm_50'...'sm_90']`——torch 2.4.1+cu121 二进制里根本没有 sm_120 内核，**这部分无法靠重编译第三方扩展解决**
- **修复**：升级 torch 到 **2.7.1+cu128**（官方 PyTorch cu128 源，原生含 sm_120；RoboTwin requirements 的 `torch==2.4.1` 锁定被打破，属必要偏离，需回归验证 sapien/mplib/open3d 兼容性）
- **完整架构结论（三层各管各的）**：
  1. torch 自带算子 → 由 torch 二进制的 arch 列表决定 → 必须 cu128 版 torch
  2. curobo 自定义 CUDA 扩展 → 由编译时 gencode 决定 → nvcc 12.8 + `-gencode compute_120`
  3. 两者可以错开：curobo 编 sm_120 + torch cu128 才是完整解；只做其一都会在某一层炸

### 坑 9：torch 升级 2.7.1+cu128 引发的连锁反应（2026-08-31）

| # | 报错 | 根因 | 修复 |
|---|---|---|---|
| 9a | 官方源 150KB/s，2GB 要 4 小时 | PyTorch 官方境外源慢 | **切阿里云镜像** `--find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/`（注意：`--index-url` 指向该目录会报"找不到版本"，因为无 PEP 503 索引，必须用 find-links） |
| 9b | curobo 扩展 `kinematics_fused_cu not found, JIT compiling` → `Ninja is required` | torch 大版本变了 → C++ 扩展 ABI 变 → 旧的 .so 不被加载，触发 JIT 重编译，而 ninja 没装 | `pip install ninja` |
| 9c | JIT 编译 `crt/host_defines.h: No such file` | nvcc 12.8 的 conda 布局把 crt 头放 targets 下，g++ 编译器找不到 | 把 `targets/x86_64-linux/include/crt/*.h` 复制到 `$CONDA_PREFIX/include/crt/` |
| 9d | JIT 链接 `cannot find -lcudart` | libcudart.so 真身在 targets 下，conda lib 里只有相对链接且缺开发链接 | `LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib`（或在 conda lib 里补软链） |

**核心认知**：torch 大版本升级后，所有 `pip install -e .` 装的 C++/CUDA 扩展都按新 ABI JIT 重编译一次；这是一次性的（编译产物会缓存到 `~/.cache/torch_extensions/`），但要把整条编译链（ninja/头文件/库路径）再喂饱一遍。

### 坑 10：`AttributeError: module 'warp' has no attribute 'torch'` + `ffmpeg not found`（2026-08-31）

| # | 报错 | 根因 | 修复 |
|---|---|---|---|
| 10a | `module 'warp' has no attribute 'torch'` | curobo v0.7.7 用 `warp.torch`（旧 warp-lang 1.0 API），但装到了 warp-lang 1.16（新 API 把 `warp.torch` 拆走了） | `pip install warp-lang==1.0.2` |
| 10b | `FileNotFoundError: 'ffmpeg'` | 评测脚本把 episode 存视频，调系统 `ffmpeg`，机器没装 | 软链 imageio_ffmpeg 自带的二进制到 `~/bin/ffmpeg`（`imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`），无需 sudo 装系统 ffmpeg |

**核心认知**：curobo v0.7.7 这套老代码的依赖链上全是「版本被上游大改」的雷——warp 的 torch 子模块、ffmpeg 缺失，都是同一类「老代码 × 新依赖」的版本漂移，和坑 3（setuptools pkg_resources）、坑 6（curobov2）同源。

---

## 七、冒烟测试成功（2026-08-31 18:22）——评测链路全线打通 🎉

### 里程碑
评测 client 成功跑通 adjust_bottle 任务，episode 逐个执行、判定、保存视频：

```
results/stseed-10000/visualization/adjust_bottle/
  0_Grab_the_plastic_drink_bottle..._True.mp4   ← episode 0 成功
  1_Grab_the_smooth_bottle..._True.mp4          ← episode 1 成功
  2_Lift_the_bottle_with_white_printed..._True.mp4  ← episode 2 成功
```

文件名末尾的 `True` = 该 episode 判定成功。视频实际路径在 **`~/RoboTwin/results/`**（脚本内部 `os.chdir(robowin_root)`，所以日志里的相对路径 `results/` 是相对 RoboTwin 目录）。

### 完整的坑谱（共 10 个，按时间线）

| # | 一句话 |
|---|---|
| 1 | 训练 Pool 多进程 hang → ThreadPool（issue #32）|
| 2 | pytorch3d 编译缺 torch → `--no-build-isolation` |
| 3 | sapien 缺 pkg_resources → setuptools<81 |
| 4 | HF 下载 xet I/O error → HF_HUB_DISABLE_XET=1 |
| 5 | client 参数 unrecognized → 全放 `--overrides` 后 |
| 6 | curobo main 分支 API 不兼容 → checkout v0.7.7 |
| 7 | CUDA 编译链五连坑（nvcc 12.1/头文件/gcc 13/fp4fp6/libcudart）|
| 8 | Blackwell sm_120 无 CUDA 内核 → 升 torch 2.7.1+cu128 + gencode 直传 |
| 9 | torch 升级连锁（镜像/ninja/crt 头/链接路径）|
| 10 | warp.torch 版本漂移 + ffmpeg 缺失 |

### 最终可复现的环境（robotwin 环境）

- torch **2.7.1+cu128**（不是官方的 2.4.1，Blackwell 必需）
- curobo **v0.7.7**（不是 main 的 curobov2），nvcc **12.8** 编译 sm_120
- warp-lang **1.0.2**（不是 1.16）
- setuptools **<81**、ninja、ffmpeg（软链 imageio_ffmpeg）
- 关键环境变量：`CUDA_HOME=$CONDA_PREFIX`、`CPATH`（targets include + cccl）、`LIBRARY_PATH`（targets lib）、`LD_LIBRARY_PATH=/usr/lib64:/usr/lib`

### 下一步
- 冒烟 5 episode 完成后 → 正式评测 50 任务 × 100 episodes（后台跑，1-2 天）
- 对比论文基准：Easy 92.9 / Hard 91.6
