import shutil
import os

# 源文件路径
source_path = "runs/classify/brain_tumor_classifier/weights/best.pt"
# 目标文件路径
target_path = "best_brain_tumor_classifier.pt"

# 检查源文件是否存在
if os.path.exists(source_path):
    # 复制文件
    shutil.copy2(source_path, target_path)
    print(f"模型已从 {source_path} 复制到 {target_path}")
else:
    print(f"源文件不存在: {source_path}")
    
    # 检查其他可能的路径
    alternative_paths = [
        "runs/classify/brain_tumor_classifier2/weights/best.pt",
        "runs/classify/brain_tumor_classifier/weights/last.pt",
        "runs/classify/brain_tumor_classifier2/weights/last.pt"
    ]
    
    for path in alternative_paths:
        if os.path.exists(path):
            shutil.copy2(path, target_path)
            print(f"模型已从 {path} 复制到 {target_path}")
            break
    else:
        print("未找到训练好的模型文件")
        print("现有模型文件:")
        for root, dirs, files in os.walk("runs"):
            for file in files:
                if file.endswith(".pt"):
                    print(os.path.join(root, file))
