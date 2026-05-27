# 脑瘤检测及医患交流平台

这是一个基于小样本学习的YOLOv11模型和神经网络分割模型nnU-Net（本项目里称为YOLOnl和UNetnl）的脑瘤检测系统，可实现肿瘤的精确识别和分割，使用Mayavi+VTK实现了肿瘤区域3D可视化，打造了医患双端以实现交流功能，用来辅助医生进行诊疗。 
为了防止医疗信息泄露，仅适用于本地电脑部署

## 功能特点

1. **脑瘤检测**: 使用YOLOv11模型进行脑瘤检测，支持图像上传和实时分析
2. **医患交流**: 提供患者与医生之间的交流平台，支持实时消息传递
3. **用户管理**: 支持用户注册和登录功能，区分患者和医生角色
4. **数据存储**: 使用MySQL数据库存储用户信息和消息记录
5. **可视化界面**: 提供直观的用户界面，包含检测结果的概率分布图
6. **检查报告**: 每次检测后可生成详细的检查报告，包含"仅供参考"的重要提示
7. **脑瘤分割**: 使用nnU-Net训练模型对检测到的脑瘤进行分割，生成分割结果图

## 系统架构

- `yolo11_brain_tumor_classifier.py`: 训练YOLOv11模型的脚本
- `backend.py`: Flask后端服务器，提供API接口
- `frontend/index.html`: 前端首页，仅用于登录和注册功能
- `frontend/chat.html`: 患者平台，包含脑瘤检测和与医生交流功能
- `frontend/doctor.html`: 专门的医生平台界面
- `init_database.py`: 数据库初始化脚本
- `copy_model.py`: 模型复制脚本
- `best_brain_tumor_classifier.pt`: 训练好的最佳模型文件
- `requirements.txt`: 项目依赖包列表
- `README.md`: 项目说明文档
- `flask_nnunet.server.py`: 模型服务文件, 用于启动脑瘤分割模型服务

## 安装步骤

1. 克隆项目到本地
2. 安装依赖包：`pip install -r requirements.txt`
3. 确保你有脑瘤数据集（如果需要重新训练模型）
4. 初始化数据库：`python init_database.py`（已经初始化过了，不用再运行这个文件了）
5. 运行后端服务器：`python backend.py`
6. 在浏览器中打开 `http://localhost:5000`

## 使用说明

1. **首页导航**:
   - 访问 `http://localhost:5000` 进入系统首页
   - 通过登录/注册功能管理账户
   - 登录后可根据需要访问患者平台或医生平台

2. **脑瘤检测**:
   - 登录后进入"患者平台" (`http://localhost:5000/chat`)
   - 在脑瘤检测区域上传CT或MRI扫描图像
   - 系统将自动分析图像并显示检测结果，包括:
     - 预测类别（健康/肿瘤）
     - 置信度分数
     - 各类别概率分布

3. **医患交流**:
   - 在患者平台中选择要交流的医生
   - 通过聊天界面发送和接收消息
   - 所有消息都会安全地存储在数据库中

4. **医生平台**:
   - 登录后访问医生平台 (`http://localhost:5000/doctor`)
   - 查看患者提交的检测报告
   - 与患者进行交流
   
5. **脑瘤分割**:
   - 登录后访问医生平台 (`http://localhost:5000/doctor`)
   - 将患者脑部mri文件上传至服务器
   - 系统将自动对上传的mri文件进行分割，并生成分割结果图
## 技术栈

- Python 3.x
- Flask (后端框架)
- YOLOv11 (目标检测模型)
- HTML/CSS/JavaScript (前端技术)
- MySQL (数据库)
- nnU-Net (脑瘤分割模型)

## 错误处理

- 前端实现了健壮的错误处理机制，能够捕获并显示各种异常情况
- 后端包含了详细的错误日志和状态检查
- 支持多种图像格式（JPG, PNG, BMP, TIFF）

## 扩展性

- 系统设计具有良好的扩展性，可以轻松添加新的功能模块
- 数据库结构支持多用户、多角色的复杂交互场景
- 模型可以替换为其他YOLO版本或深度学习模型
- 分割模型可以适配不同的数据集，满足不同需求，未来可以实现不局限于脑部肿瘤的分割，如：肺癌分割、肝癌分割等

## 脑部肿瘤分割模型
- 模型基于nnU-Net，是一个基于U-Net的医学图像分割模型，可以进行 precise brain tumor segmentation.
- 本模型是基于Brats2020的MRI数据集进行训练，模型在测试集上准确率达到0.93
- 模型训练历史见：nnUNet_project/nnUNet_results/Dataset044_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/progress.png
- 模型训练的最佳参数见：nnUNet_project/nnUNet_trained_models/Dataset044_BraTS2020/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/model_best.model.npz
- 支持多种文件格式，如：.nii.gz, .nii等
