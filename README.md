# SegmentationSystem - 遥感地物分割系统

## 项目简介
本项目是《软件工程综合实训》课程项目，旨在开发一个基于深度学习的遥感影像地物语义分割Web系统。系统使用U-Net模型对来自LoveDA数据集的遥感图像进行像素级分类，识别建筑、道路、水体等7类地物，并通过Web界面提供直观的上传、分割和结果可视化功能。

**核心功能：**
- 用户上传单张遥感图像（JPG/PNG格式）。
- 后端调用预训练的U-Net模型进行语义分割。
- 前端并排展示原图与分割结果图，并附颜色图例说明。
- 展示模型在验证集上的定量评估指标（如mIoU）。

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

## 项目结构
SegmentationSystem/
├── backend/ # 后端 Flask 应用
│ ├── app.py # 应用主入口和路由
│ ├── model_inference.py # 模型加载与推理函数
│ ├── image_processor.py # 图像预处理与后处理
│ └── static/images/ # 用户上传的临时图像存储
├── frontend/ # 前端静态页面
│ ├── index.html # 主页面
│ └── static/ # CSS, JavaScript 文件
├── scripts/ # 模型训练与评估脚本
│ ├── train.py # 训练 U-Net 模型
│ └── evaluate.py # 评估模型性能
├── data/ # 数据集（通常通过 .gitignore 忽略，需自行下载 LoveDA 数据集）
├── models/ # 训练好的模型文件存储位置
├── docs/ # 项目文档、报告
├── requirements.txt # Python 依赖包列表
└── README.md # 项目说明文件（本文件）


## 团队成员与分工
- **[高文楷]** (项目经理): 负责项目协调与系统集成。
- **[高文楷]** (算法开发): 负责U-Net模型的训练、调优与评估脚本。
- **[郑林山]** (前端开发/后端开发): 负责Web界面的设计与交互实现、后端API开发。
- **[高文楷]** (测试/文档): 负责系统测试、数据预处理及项目文档撰写。

## 开发协作流程
1.  所有新功能应在 `develop` 分支上进行开发。
2.  每位成员从 `develop` 分支创建自己的功能分支（如 `feature/xxx`）。
3.  开发完成后，向 `develop` 分支发起 Pull Request (PR)，并至少邀请一位队友进行代码审查。
4.  PR 通过审查后，由项目经理合并到 `develop` 分支。
5.  项目里程碑或演示前，将稳定的 `develop` 分支合并至 `main` 分支。