# SegmentationSystem - 遥感地物分割系统

## 项目简介
本项目是《软件工程综合实训》课程项目，旨在开发一个基于深度学习的遥感影像地物语义分割Web系统。系统使用U-Net模型对来自LoveDA数据集的遥感图像进行像素级分类，识别建筑、道路、水体等7类地物，并通过Web界面提供直观的上传、分割和结果可视化功能。

## 项目需求
本项目需满足以下核心需求，以确保符合课程要求并具备完整功能：
1.可视化界面：必须包含Web或桌面端界面，而非纯算法项目。
2.完整流程：实现从“用户上传图像”到“展示分割结果”的端到端闭环。
3.核心功能：
（1）支持上传单张遥感图像（JPG/PNG格式）。
（2）后端调用预训练的U-Net模型进行语义分割。
（3）前端并排展示“原图”与“分割结果图”。
（4）清晰展示7类地物（建筑、道路、水体、贫瘠土地、森林、农业用地、背景）的颜色图例。
（5）展示模型在验证集上的定量评估指标（如mIoU）。
4.进阶功能（可选）：
（1）批量上传与处理多张图像。
（2）分割结果图下载功能。
（3）简单的地物面积占比统计图表。

## 运行方式
### 环境准备
1.  确保已安装 [Python 3.8+](https://www.python.org/) 和 [Git](https://git-scm.com/)。
2.  克隆本项目到本地：
    ```bash
    git clone https://github.com/<你的用户名>/SegmentationSystem.git
    cd SegmentationSystem
    ```
3.  创建并激活虚拟环境（推荐）：
    ```bash
    conda create -n segmentation python=3.11
    conda activate segmentation
    ```
4.  安装项目依赖：
    ```bash
    pip install -r requirements.txt
    ```

### 启动系统
1.  确保已按照 `scripts/README.md` 的说明训练好模型，并将模型文件置于 `models/` 目录。
2.  启动Flask后端服务器：
    ```bash
    cd backend
    python app.py
    ```
3.  在浏览器中访问 `http://127.0.0.1:5000` 即可使用系统。

SegmentationSystem/
├── README.md                           # 项目总说明文档（本文档）
├── requirements.txt                    # Python项目依赖包列表
├── run.py                              # 项目统一启动入口（可选）
├── 2021LoveDA/                         # 【数据集目录】原始LoveDA数据集，按原始结构存放
│   ├── Train/
│   │   ├── Rural/
│   │   │   ├── images_png/             # 原始训练影像
│   │   │   ├── masks_png/              # 原始训练标签
│   │   │   ├── images_256/             # 离线 resize 后的训练影像（preprocess_data.py生成）
│   │   │   └── masks_256/              # 离线 resize 后的训练标签
│   │   └── Urban/
│   │       ├── images_png/
│   │       ├── masks_png/
│   │       ├── images_256/
│   │       └── masks_256/
│   ├── Val/
│   │   ├── Rural/
│   │   │   ├── images_png/
│   │   │   ├── masks_png/
│   │   │   ├── images_256/
│   │   │   └── masks_256/
│   │   └── Urban/
│   │       ├── images_png/
│   │       ├── masks_png/
│   │       ├── images_256/
│   │       └── masks_256/
│   └── Test/
│       ├── Rural/
│       │   └── images_png/
│       └── Urban/
│           └── images_png/
├── backend/                            # 【后端目录】Flask服务器核心代码
│   ├── app.py                          # 后端主程序：Flask应用实例、API路由定义（如/upload, /predict）
│   ├── config.py                       # 后端配置文件：模型路径、上传文件夹路径等常量
│   ├── image_processor.py              # 图像处理模块：负责图片尺寸调整、归一化等预处理
│   ├── model_inference.py              # 模型推理模块：加载训练好的模型，执行预测
│   ├── static/images/                  # 静态文件夹：用于临时存储用户上传的图片
│   └── templates/                      # Jinja2模板文件夹（本项目可能直接使用前端静态页面）
├── config/                             # 配置文件目录
│   └── config.yaml                     # 全局配置文件（如超参数、路径）
├── docs/                               # 项目文档目录（用于存放报告、设计文档等）
├── frontend/                           # 【前端目录】静态网页文件
│   ├── index.html                      # 前端主页面：包含上传表单、图片展示区、图例等HTML结构
│   └── static/
│       ├── css/style.css               # 前端样式表：定义页面布局、颜色、字体等样式
│       ├── js/main.js                  # 前端交互逻辑：处理图片上传、调用后端API、动态展示结果
│       └── images/                     # 前端用到的静态图片资源
├── models/                             # 模型目录：用于存放所有训练好的模型文件（.pth格式）
│   ├── unet_latest.pth                 # 当前训练得到的U-Net模型权重
│   ├── unet_best.pth                   # 最佳模型权重（完整检查点）
│   └── unet_best_cpu.pth               # CPU优化训练的最佳模型
├── scripts/                            # 【算法脚本目录】用于数据与模型相关的Python脚本
│   ├── preprocess_data.py              # 数据预处理脚本：离线resize LoveDA数据，加速训练
│   ├── train.py                        # 模型训练脚本：加载数据集，训练U-Net并保存模型（已优化）
│   ├── evaluate.py                     # 模型评估脚本：计算mIoU并生成可视化结果
│   ├── verify_labels.py                # 标签验证脚本：分析数据集标签分布
│   ├── optimized_train.py              # 优化版训练脚本（CPU友好）
│   ├── datasets/                       # 数据集定义模块
│   │   └── loveda_dataset.py           # LoveDA Dataset类（已增强数据增强）
│   ├── models/                         # 模型结构定义模块
│   │   └── unet.py                     # U-Net网络结构定义
│   └── samplers/                       # 数据采样器模块
│       └── balanced_sampler.py         # 平衡采样器（用于类别不平衡数据）
├── outputs/                            # 模型预测结果输出目录
│   ├── training_20260104_2240/         # 最新训练结果
│   │   ├── training_curves.png         # 训练曲线图
│   │   ├── config.json                 # 训练配置
│   │   └── training_stats.json         # 训练统计
│   ├── best/                           # mIoU较高的预测样例（评估后生成）
│   ├── mid/                            # 中等效果预测样例
│   └── worst/                          # 效果较差的预测样例
└── tests/                              # 测试代码目录
    ├── backend/                        # 后端API测试
    └── frontend/                       # 前端交互测试



## 团队成员与分工
- **[高文楷]** (项目经理): 负责项目协调与系统集成。
- **[高文楷]** (算法开发): 负责U-Net模型的训练、调优与评估脚本。
- **[郑林山]** (前端开发/后端开发): 负责Web界面的设计与交互实现、后端API开发。
- **[高文楷]** (测试/文档): 负责系统测试、数据预处理及项目文档撰写。

## 开发协作流程
1.开发基于分支：所有新功能必须在 develop 分支上进行开发。禁止直接向 main 分支提交代码。
2.创建功能分支：每位成员从最新的 develop 分支创建个人功能分支，命名格式为 feature/姓名-功能简述（如 feature/zhangsan-flask-upload-api）。
3.提交与合并：
（1）在个人分支上完成开发、自测后，提交代码并推送至远程仓库。
（2）在GitHub上向 develop 分支发起 Pull Request (PR)。
（3）必须在PR描述中说明改动内容，并至少邀请一位队友进行代码审查。
（4）审查通过后，由项目经理（高文楷） 执行合并操作。
4.同步更新：每日开始工作前，请务必从远程仓库拉取最新的 develop 分支代码，保持本地同步。
5.冲突解决：如遇合并冲突，通常由PR发起者负责解决。请在本地整合代码后，重新提交。