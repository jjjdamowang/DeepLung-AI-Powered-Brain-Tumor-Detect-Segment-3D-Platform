# 独立的轮廓生成脚本
import os
import sys
import argparse

# 设置matplotlib为非交互式后端
import matplotlib
matplotlib.use('Agg')

# 导入必要的模块
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure

def generate_contour(img_path, seg_path, output_path):
    """
    生成带肿瘤轮廓的图像
    :param img_path: 原始影像路径 (.nii.gz)
    :param seg_path: 分割掩码路径 (.nii.gz)
    :param output_path: 输出图像路径 (.png)
    """
    print(f"加载原始影像: {img_path}")
    try:
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)  # (Z, H, W)
        print(f"原始影像尺寸: {img_array.shape}")
    except Exception as e:
        raise Exception(f"原始影像加载失败：{e}")

    print(f"加载分割掩码: {seg_path}")
    try:
        seg = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(seg)
        print(f"分割掩码尺寸: {seg_array.shape}")
        # 校验尺寸
        assert img_array.shape == seg_array.shape, "原始影像和分割掩码尺寸不匹配！"
    except Exception as e:
        raise Exception(f"分割掩码加载失败：{e}")

    # 选择中间切片
    current_slice = img_array.shape[0] // 2
    print(f"选择切片: {current_slice}")

    # 创建绘图窗口
    fig, ax = plt.subplots(figsize=(10, 8))
    # 显示原始灰度影像
    im = ax.imshow(img_array[current_slice], cmap='gray')
    # 提取当前切片的轮廓并绘制红色线条
    print("提取轮廓...")
    contours_measure = measure.find_contours(seg_array[current_slice], level=0.5)
    print(f"找到 {len(contours_measure)} 个轮廓")
    for contour in contours_measure:
        # 转换为整数坐标（适配像素）
        y_coords = contour[:, 0].astype(int)
        x_coords = contour[:, 1].astype(int)
        ax.plot(x_coords, y_coords, color='red', linewidth=2)

    # 设置标题
    ax.set_title("Tumor Segmentation Contour (Red)")
    ax.axis('off')

    # 保存图像
    print(f"保存图像: {output_path}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print("轮廓生成成功！")
    return output_path

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成带肿瘤轮廓的图像")
    parser.add_argument("--img", required=True, help="原始影像路径 (.nii.gz)")
    parser.add_argument("--seg", required=True, help="分割掩码路径 (.nii.gz)")
    parser.add_argument("--output", required=True, help="输出图像路径 (.png)")
    args = parser.parse_args()
    
    try:
        # 生成轮廓图像
        generate_contour(args.img, args.seg, args.output)
        print(f"轮廓图像生成成功: {args.output}")
    except Exception as e:
        print(f"轮廓生成失败: {e}")
        sys.exit(1)
