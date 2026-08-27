# lingbot-va-deploy

LingBot-VA deployment checklist and cloud evaluation scripts

## 文档索引

| 文件 | 说明 |
|---|---|
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | 部署与运行清单（硬性门槛 / 安装 / 权重 / 配置 / 训练）|
| [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) | 云上从零跑通 RoboTwin 评测指南 |
| [AUTODL_TROUBLESHOOTING.md](AUTODL_TROUBLESHOOTING.md) | 🔥 **AutoDL 部署踩坑实战记录**（每个坑含报错原文 + 根因 + 解决命令，强烈建议部署前先读）|
| [cloud_robotwin_eval.sh](cloud_robotwin_eval.sh) | 一键云上评测脚本 |

## 快速开始

部署前先读 [AUTODL_TROUBLESHOOTING.md](AUTODL_TROUBLESHOOTING.md)，里面有：

- 完整的从零部署命令全集（可复制粘贴）
- 最常见的坑：代理开关时机、Vulkan 驱动缺失、sapien 预发布版等
- 一键验证清单

配合 [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) 和 [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) 使用。
