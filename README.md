# lingbot-va-deploy

LingBot-VA 部署、训练与二次开发记录。包含：从零部署、本地后训练、踩坑实战、以及（进行中的）触觉模块二次开发分析。

## 文档体系（按使用场景）

### 1. 部署主线（从零跑起来）

| 文件 | 说明 |
|---|---|
| [AUTODL_TROUBLESHOOTING.md](AUTODL_TROUBLESHOOTING.md) | 🔥 **部署踩坑实战记录**（每个坑含报错原文 + 根因 + 解决命令，部署前必读）|
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | 部署与运行清单（硬性门槛 / 安装 / 权重 / 配置 / 训练）|
| [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) | 云上从零跑通 RoboTwin 评测指南 |
| [cloud_robotwin_eval.sh](cloud_robotwin_eval.sh) | 一键云上评测脚本 |
| [EVALUATION_SETUP.md](EVALUATION_SETUP.md) | 评测环境搭建记录（双环境架构 + 步骤清单）|
| [EVALUATION_TROUBLESHOOTING.md](EVALUATION_TROUBLESHOOTING.md) | 🔥 **评测环境踩坑实录**（8 大坑：Pool hang / curobo 版本 / CUDA 编译链五连坑 + 方法论）|

### 2. 本地训练主线（RTX PRO 6000 单卡）

| 文件 | 说明 |
|---|---|
| [LOCAL_DEPLOY_SUMMARY.md](LOCAL_DEPLOY_SUMMARY.md) | 本地部署总结：阶段一推理复现 + 阶段二后训练（含流程图与推理结果图）|
| [PHASE2_TRAINING.md](PHASE2_TRAINING.md) | 阶段二后训练详细记录：配置、数据集、`Pool→ThreadPool` 诊断与修复、checkpoint 说明 |
| [TRAINING_REPORT.md](TRAINING_REPORT.md) | 训练报告：loss 分析、推理验证、动作对比、工程细节（训练完成后）|
| [TRAINING_LOG.md](TRAINING_LOG.md) | 训练运行日志（进度 / loss / GPU 温度）|

### 3. 二次开发主线（加触觉模块）

| 文件 | 说明 |
|---|---|
| [TACTILE_ANALYSIS.md](TACTILE_ANALYSIS.md) | 触觉模块插入点分析（数据流全景 + 模型架构 + 6 处改动点，草稿）|

### 4. 元记录

| 文件 | 说明 |
|---|---|
| [MISTAKES_LOG.md](MISTAKES_LOG.md) | 犯错记录与诊断教训（复盘）|

## 快速开始

- **要部署/评测** → 先读 [AUTODL_TROUBLESHOOTING.md](AUTODL_TROUBLESHOOTING.md)
- **要在本地训练** → 读 [PHASE2_TRAINING.md](PHASE2_TRAINING.md) + [LOCAL_DEPLOY_SUMMARY.md](LOCAL_DEPLOY_SUMMARY.md)
- **要二次开发（加触觉）** → 读 [TACTILE_ANALYSIS.md](TACTILE_ANALYSIS.md)
