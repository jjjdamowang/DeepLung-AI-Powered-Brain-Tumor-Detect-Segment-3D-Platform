import os
import random
import torch
import numpy as np
from pathlib import Path

# 在导入ultralytics之前设置环境变量
os.environ["SSLKEYLOGFILE"] = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import csv
import matplotlib.pyplot as plt

# 设置环境变量以避免多线程冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class BrainTumorClassifier:
    """
    基于YOLOv11的脑肿瘤分类器
    实现健康/肿瘤二分类任务
    """
    
    def __init__(self, data_root=".", target_root="./brain_tumor_dataset"):
        self.data_root = Path(data_root)
        self.target_root = Path(target_root)
        
        # 数据集配置
        self.image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")
        self.image_folders = ["Brain Tumor CT scan Images（CT图）", "Brain Tumor MRI images（MRI图）"]
        self.cls_mapping = {"Healthy": "healthy", "Tumor": "tumor"}
        
        # 训练配置
        # 检查本地是否存在预训练模型
        if os.path.exists("yolo11n-cls.pt"):
            self.model_name = "yolo11n-cls.pt"  # 使用本地模型
        elif os.path.exists("yolov8n-cls.pt"):  # 备选模型
            self.model_name = "yolov8n-cls.pt"
        else:
            self.model_name = "yolo11n-cls.pt"  # 默认模型名，会尝试下载
        self.train_name = "brain_tumor_classifier"
        self.epochs = 50
        self.batch_size = 16  # GPU训练批次大小
        
        # 设备配置
        self.device = self._setup_device()
        
        # 数据划分比例
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        
        print(f"初始化完成，使用设备: {self.device}")
    
    def _setup_device(self):
        """设置训练设备"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"检测到 {gpu_count} 个CUDA设备")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name}, 显存: {memory:.1f} GB")
            
            # 使用第一个GPU
            device = "0"
            print(f"使用GPU {device} 进行训练")
        else:
            device = "cpu"
            print("未检测到CUDA GPU，使用CPU进行训练")
        
        return device
    
    def organize_dataset(self):
        """整理数据集为YOLO格式"""
        print("开始整理数据集...")
        
        # 创建目标目录结构
        for split in ["train", "val", "test"]:
            for cls in self.cls_mapping.values():
                target_path = self.target_root / split / cls
                target_path.mkdir(parents=True, exist_ok=True)
        
        total_images = 0
        
        # 遍历每个原始数据文件夹
        for folder in self.image_folders:
            folder_path = self.data_root / folder
            if not folder_path.exists():
                print(f"警告：原始文件夹 {folder_path} 不存在，跳过！")
                continue
            
            # 遍历每个分类（Healthy/Tumor）
            for orig_cls, target_cls in self.cls_mapping.items():
                cls_path = folder_path / orig_cls
                if not cls_path.exists():
                    print(f"警告：分类文件夹 {cls_path} 不存在，跳过！")
                    continue
                
                # 获取该分类下所有图片
                images = [
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith(self.image_extensions)
                ]
                
                if not images:
                    print(f"警告：{cls_path} 下无图片，跳过！")
                    continue
                
                # 打乱数据（保证划分随机）
                random.seed(42)  # 固定随机种子，结果可复现
                random.shuffle(images)
                total = len(images)
                total_images += total
                
                # 划分训练/验证/测试集
                train_num = int(total * self.train_ratio)
                val_num = int(total * self.val_ratio)
                test_num = total - train_num - val_num
                
                train_imgs = images[:train_num]
                val_imgs = images[train_num:train_num + val_num]
                test_imgs = images[train_num + val_num:]
                
                # 复制图片到目标目录
                def copy_imgs(img_list, split):
                    for img in img_list:
                        src = cls_path / img
                        dst = self.target_root / split / target_cls / img
                        if not dst.exists():  # 避免重复复制
                            with open(src, "rb") as f_src, open(dst, "wb") as f_dst:
                                f_dst.write(f_src.read())
                
                copy_imgs(train_imgs, "train")
                copy_imgs(val_imgs, "val")
                copy_imgs(test_imgs, "test")
                
                print(f"{folder}->{orig_cls}：共{total}张 → 训练{train_num} | 验证{val_num} | 测试{test_num}")
        
        print(f"\n数据集整理完成！总图片数：{total_images}")
        print(f"整理后路径：{self.target_root.absolute()}")
        return total_images
    
    def train_model(self):
        """训练YOLOv11分类模型"""
        print(f"开始训练模型，总共 {self.epochs} 轮...")
        
        # 直接使用本地模型文件
        local_model_path = self.model_name
        if os.path.exists(local_model_path):
            print(f"使用本地模型文件: {local_model_path}")
            try:
                model = YOLO(local_model_path)
            except Exception as e:
                print(f"加载本地模型失败: {e}")
                # 尝试使用其他本地模型
                alternative_models = ["yolov8n-cls.pt", "yolo11n.pt", "best.pt"]
                model_loaded = False
                for alt_model in alternative_models:
                    if os.path.exists(alt_model):
                        print(f"尝试使用替代模型: {alt_model}")
                        try:
                            model = YOLO(alt_model)
                            model_loaded = True
                            break
                        except Exception as alt_e:
                            print(f"加载替代模型 {alt_model} 失败: {alt_e}")
                            continue
                if not model_loaded:
                    print("所有本地模型都无法加载")
                    return None
        else:
            print(f"本地模型文件不存在: {local_model_path}")
            return None
        
        # 开始训练
        try:
            results = model.train(
                data=str(self.target_root),
                epochs=self.epochs,
                batch=self.batch_size,
                device=self.device,
                name=self.train_name,
                pretrained=True,
                patience=15,  # 早停
                save=True,
                val=True,
                imgsz=640,  # 输入尺寸
                weight_decay=0.0005,
                lr0=0.01,  # 初始学习率
                lrf=0.01,  # 最终学习率
                verbose=True,  # 显示详细训练信息
                workers=0,  # 设置DataLoader工作进程数为0，避免多进程问题
                # 数据增强
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                flipud=0.0,  # 医疗图像禁用上下翻转
                fliplr=0.5,  # 左右翻转增强
                mosaic=1.0,
            )
            
            # 将训练好的模型复制到项目根目录
            trained_model_path = self.target_root.parent / "runs" / "classify" / self.train_name / "weights" / "best.pt"
            if trained_model_path.exists():
                import shutil
                final_model_path = Path(".") / "best_brain_tumor_classifier.pt"
                shutil.copy(trained_model_path, final_model_path)
                print(f"模型已保存到项目根目录: {final_model_path}")
            
            print("模型训练完成！")
            return results
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            return None
    
    def evaluate_model(self):
        """评估模型性能"""
        model_path = self.target_root.parent / "runs" / "classify" / self.train_name / "weights" / "best.pt"
        
        if not model_path.exists():
            print("警告：未找到训练好的模型！")
            return None
        
        print("开始评估模型...")
        model = YOLO(model_path)
        
        # 在验证集上评估
        val_results = model.val(
            data=str(self.target_root),
            split="val",
            imgsz=640,
            device=self.device
        )
        
        # 在测试集上评估
        test_results = model.val(
            data=str(self.target_root),
            split="test",
            imgsz=640,
            device=self.device
        )
        
        print("\n==================== 验证集评估结果 ====================")
        print(f"Top1准确率: {val_results.top1:.2f}%")
        print(f"Top5准确率: {val_results.top5:.2f}%")
        print(f"验证损失: {val_results.loss.data['train_loss']:.4f}")
        
        print("\n==================== 测试集评估结果 ====================")
        print(f"Top1准确率: {test_results.top1:.2f}%")
        print(f"Top5准确率: {test_results.top5:.2f}%")
        print(f"测试损失: {test_results.loss.data['train_loss']:.4f}")
        
        return val_results, test_results
    
    def predict_single_image(self, image_path):
        """对单张图片进行预测"""
        model_path = self.target_root.parent / "runs" / "classify" / self.train_name / "weights" / "best.pt"
        
        if not model_path.exists():
            print("警告：未找到训练好的模型！")
            return None
        
        model = YOLO(model_path)
        results = model(image_path, device=self.device)
        
        # 获取预测结果
        pred_class = results[0].names[results[0].probs.top1]
        pred_confidence = results[0].probs.top1conf.item()
        
        print(f"预测类别: {pred_class}")
        print(f"置信度: {pred_confidence:.4f}")
        
        return pred_class, pred_confidence
    
    def plot_training_history(self):
        """绘制训练历史图表"""
        runs_path = self.target_root.parent / "runs" / "classify" / self.train_name
        
        if not runs_path.exists():
            print("警告：未找到训练记录！")
            return
        
        # 读取训练结果
        history_path = runs_path / "results.csv"
        if not history_path.exists():
            print("警告：未找到训练历史记录！")
            return
        
        import pandas as pd
        df = pd.read_csv(history_path)
        
        # 绘制训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLOv11 Brain Tumor Classification Training History', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(df['epoch'], df['val/loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'], label='Top1 Acc', color='green')
        axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top5'], label='Top5 Acc', color='orange')
        axes[0, 1].set_title('Top-1 and Top-5 Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='purple')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 参数数量
        axes[1, 1].axis('off')  # 关闭坐标轴
        info_text = f"""Model Information:
        
Model: YOLOv11n-cls
Dataset: Brain Tumor Images
Classes: 2 (healthy, tumor)
Input Size: 640x640
Device: {self.device}
Batch Size: {self.batch_size}
Epochs: {self.epochs}

Final Metrics:
Top-1 Accuracy: {df['metrics/accuracy_top1'].iloc[-1]:.2f}%
Top-5 Accuracy: {df['metrics/accuracy_top5'].iloc[-1]:.2f}%
Final Loss: {df['val/loss'].iloc[-1]:.4f}
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(runs_path / "training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练历史图表已保存至: {runs_path / 'training_history.png'}")
    
    def run_complete_pipeline(self):
        """运行完整的训练流程"""
        print("=" * 60)
        print("开始YOLOv11脑肿瘤分类模型训练流程")
        print("=" * 60)
        
        # 1. 整理数据集
        print("\n步骤 1: 整理数据集")
        self.organize_dataset()
        
        # 2. 训练模型
        print("\n步骤 2: 训练模型")
        train_results = self.train_model()
        if train_results is None:
            print("训练失败，退出流程")
            return
        
        # 3. 评估模型
        print("\n步骤 3: 评估模型")
        eval_results = self.evaluate_model()
        
        # 4. 绘制训练历史
        print("\n步骤 4: 绘制训练历史")
        self.plot_training_history()
        
        print("\n" + "=" * 60)
        print("YOLOv11脑肿瘤分类模型训练流程完成！")
        print("=" * 60)


if __name__ == "__main__":
    # 创建分类器实例
    classifier = BrainTumorClassifier()
    
    # 运行完整流程
    classifier.run_complete_pipeline()
