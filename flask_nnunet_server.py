# flask_nnunet_server.py - 基于Flask的后端服务
import os
import sys
import json
import uuid
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import threading
import time
from datetime import datetime

# 任务状态存储
tasks = {}

# ====================== 核心配置 ======================
import os
NNUNET_ROOT = os.getenv("NNUNET_ROOT", r"D:\Medical Imaging\nnUNet-master")
MODEL_FOLDER = os.getenv("MODEL_FOLDER", r"D:\Medical Imaging\nnUNet_project\nnUNet_results\Dataset044_BraTS2020\nnUNetTrainer__nnUNetPlans__3d_fullres")
TASK_ID = int(os.getenv("TASK_ID", "44"))
TEMP_DIR = os.getenv("TEMP_DIR", r"D:\Medical Imaging\nnUNet_backend\temp")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", r"D:\Medical Imaging\nnUNet_backend\output")
# =========================================================
NNUNET_RAW = os.getenv("nnUNet_raw", r"D:\Medical Imaging\nnUNet_project\nnUNet_raw")
# 创建临时目录
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 支持最大1GB文件

CORS(app, origins="*", supports_credentials=True)
# 启用CORS支持

# 设置matplotlib为非交互式后端
import matplotlib
matplotlib.use('Agg')

# 导入其他模块
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
import gzip
from pathlib import Path

def convert_nii_to_nii_gz(input_path, output_path=None):
    """
    将.nii文件转换为.nii.gz格式
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径（可选）
    :return: 转换后的文件路径
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise Exception(f"输入文件不存在：{input_path}")
    if input_path.suffix != '.nii':
        return str(input_path)  # 已经是.nii.gz格式
    
    if output_path is None:
        output_path = input_path.with_suffix('.nii.gz')
    else:
        output_path = Path(output_path)
        if not output_path.name.endswith('.nii.gz'):
            output_path = output_path.with_name(f"{output_path.stem}.nii.gz")
    
    try:
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        return str(output_path)
    except Exception as e:
        raise Exception(f"转换失败：{str(e)}")

def get_contour(mask_slice):
    """
    从分割掩码切片中提取轮廓坐标
    :param mask_slice: 二维分割掩码数组 (H, W)
    :return: 轮廓的x、y坐标列表
    """
    contours = []
    # 提取所有连通区域的轮廓
    contours_measure = measure.find_contours(mask_slice, level=0.5)
    for contour in contours_measure:
        # 转换为整数坐标（适配像素）
        y_coords = contour[:, 0].astype(int)
        x_coords = contour[:, 1].astype(int)
        contours.append((x_coords, y_coords))
    return contours

def generate_contour_image(img_path, seg_path, output_path):
    """
    生成带肿瘤轮廓的图像
    :param img_path: 原始影像路径 (.nii.gz)
    :param seg_path: 分割掩码路径 (.nii.gz)
    :param output_path: 输出图像路径 (.png)
    :return: 生成的图像路径
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
    contours = get_contour(seg_array[current_slice])
    print(f"找到 {len(contours)} 个轮廓")
    for (x, y) in contours:
        ax.plot(x, y, color='red', linewidth=2)

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


def run_prediction(input_dir, output_dir, task_id):
    """
    运行nnU-Net预测
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param task_id: 任务ID
    """
    # 更新任务状态
    tasks[task_id]["status"] = "running"
    tasks[task_id]["progress"] = 10
    tasks[task_id]["message"] = "初始化预测..."
    
    # 设置nnU-Net环境变量
    NNUNET_RAW = os.getenv("nnUNet_raw", r"D:\train\nnUNet_project\nnUNet_raw")
    NNUNET_PREPROCESSED = os.getenv("nnUNet_preprocessed", r"D:\train\nnUNet_project\nnUNet_preprocessed")
    NNUNET_RESULTS = os.getenv("nnUNet_results", r"D:\train\nnUNet_project\nnUNet_results")
    
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS
    
    # 明确设置使用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    # 添加nnU-Net路径
    sys.path.append(NNUNET_ROOT)

    # 动态导入nnU-Net相关模块
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # 尝试使用GPU
    print("尝试使用GPU处理...")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # 获取最佳GPU设备
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量: {gpu_count}")
        
        # 选择内存最大的GPU
        best_gpu = 0
        max_memory = 0
        for i in range(gpu_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 内存: {gpu_memory / 1024**3:.2f} GB")
            if gpu_memory > max_memory:
                max_memory = gpu_memory
                best_gpu = i
        
        device = torch.device(f'cuda:{best_gpu}')
        print(f"选择GPU设备: {torch.cuda.get_device_name(best_gpu)}")
        print(f"GPU内存: {max_memory / 1024**3:.2f} GB")
        perform_everything_on_device = True
    else:
        print("警告: 没有可用的GPU，将使用CPU处理")
        device = torch.device('cpu')
        perform_everything_on_device = False
    
    print(f"最终使用设备: {device}")
    print(f"perform_everything_on_device: {perform_everything_on_device}")

    # 更新进度
    tasks[task_id]["progress"] = 20
    tasks[task_id]["message"] = "初始化预测器..."

    # 初始化预测器
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_everything_on_device,
        device=device,
        verbose=True,  # 增加详细程度以便调试
        verbose_preprocessing=True,  # 增加详细程度以便调试
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_FOLDER,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    # 更新进度
    tasks[task_id]["progress"] = 40
    tasks[task_id]["message"] = "开始预测..."

    # 执行预测
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_dir,
        output_folder_or_list_of_truncated_output_files=output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    
    # 更新进度
    tasks[task_id]["progress"] = 90
    tasks[task_id]["message"] = "生成轮廓图像..."


def process_single_file(file_path, output_dir=None, task_id=None):
    """
    处理单个文件：执行预测并生成带轮廓的图像
    :param file_path: 输入文件路径 (.nii 或 .nii.gz)
    :param output_dir: 输出目录（可选）
    :param task_id: 任务ID（可选）
    :return: 预测结果文件路径和轮廓图像路径
    """
    try:
        # 生成唯一ID
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        if output_dir is None:
            output_dir = os.path.join(OUTPUT_DIR, task_id)
        input_dir = os.path.join(TEMP_DIR, task_id, "input")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 复制文件到输入目录
        file_name = os.path.basename(file_path)
        input_file_path = os.path.join(input_dir, file_name)
        shutil.copy2(file_path, input_file_path)
        
        # 判断文件类型并进行预处理
        print(f"检查文件类型: {file_name}")
        if not (file_name.lower().endswith('.nii.gz') or file_name.lower().endswith('.nii')):
            raise Exception(f"不支持的文件格式: {file_name}，仅支持.nii或.nii.gz格式")
        
        # 如果是.nii文件，转换为.nii.gz格式
        if file_name.lower().endswith('.nii'):
            print(f"转换文件格式: {file_name} -> .nii.gz")
            input_file_path = convert_nii_to_nii_gz(input_file_path)
            print("格式转换成功")
        elif file_name.lower().endswith('.nii.gz'):
            print("文件已是.nii.gz格式，无需转换")
        
        # 运行预测
        print(f"开始预测: {input_file_path}")
        run_prediction(input_dir, output_dir, task_id)
        
        # 更新进度
        if task_id in tasks:
            tasks[task_id]["progress"] = 95
            tasks[task_id]["message"] = "处理完成..."
        
        # 获取预测结果
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        print(f"在输出目录中找到 {len(result_files)} 个结果文件: {result_files}")
        if not result_files:
            raise Exception("预测失败，未生成结果文件")
        
        result_file = os.path.join(output_dir, result_files[0])
        print(f"预测完成: {result_file}")
        
        # 生成带肿瘤轮廓的图像
        contour_output_dir = os.path.join(output_dir, "contour")
        os.makedirs(contour_output_dir, exist_ok=True)
        contour_image_path = os.path.join(contour_output_dir, f"contour_{result_files[0].replace('.nii.gz', '.png')}")
        
        try:
            # 生成轮廓图像
            generate_contour_image(input_file_path, result_file, contour_image_path)
            print(f"轮廓生成完成: {contour_image_path}")
        except Exception as e:
            print(f"生成轮廓图像失败：{str(e)}")
            contour_image_path = None
        
        # 更新进度
        if task_id in tasks:
            tasks[task_id]["progress"] = 100
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "任务完成"
            tasks[task_id]["result_file"] = result_file
            tasks[task_id]["contour_image"] = contour_image_path
        
        return result_file, contour_image_path
        
    except Exception as e:
        if task_id in tasks:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = f"处理失败: {str(e)}"
        raise Exception(f"处理文件失败：{str(e)}")


def run_prediction_async(input_dir, output_dir, task_id):
    """
    异步运行预测
    """
    try:
        process_single_file(None, output_dir, task_id)
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"预测失败: {str(e)}"


def run_prediction_folder_async(input_dir, output_dir, task_id, uploaded_files):
    """
    异步运行多文件预测
    """
    try:
        # 更新任务状态
        tasks[task_id]["status"] = "running"
        tasks[task_id]["progress"] = 10
        tasks[task_id]["message"] = "初始化预测..."
        
        # 设置nnU-Net环境变量
        NNUNET_RAW = os.getenv("nnUNet_raw", r"D:\train\nnUNet_project\nnUNet_raw")
        NNUNET_PREPROCESSED = os.getenv("nnUNet_preprocessed", r"D:\train\nnUNet_project\nnUNet_preprocessed")
        NNUNET_RESULTS = os.getenv("nnUNet_results", r"D:\train\nnUNet_project\nnUNet_results")
        
        os.environ["nnUNet_raw"] = NNUNET_RAW
        os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
        os.environ["nnUNet_results"] = NNUNET_RESULTS
        
        # 明确设置使用GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        # 添加nnU-Net路径
        sys.path.append(NNUNET_ROOT)

        # 动态导入nnU-Net相关模块
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        # 尝试使用GPU
        print("尝试使用GPU处理...")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            # 获取最佳GPU设备
            gpu_count = torch.cuda.device_count()
            print(f"可用GPU数量: {gpu_count}")
            
            # 选择内存最大的GPU
            best_gpu = 0
            max_memory = 0
            for i in range(gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 内存: {gpu_memory / 1024**3:.2f} GB")
                if gpu_memory > max_memory:
                    max_memory = gpu_memory
                    best_gpu = i
            
            device = torch.device(f'cuda:{best_gpu}')
            print(f"选择GPU设备: {torch.cuda.get_device_name(best_gpu)}")
            print(f"GPU内存: {max_memory / 1024**3:.2f} GB")
            perform_everything_on_device = True
        else:
            print("警告: 没有可用的GPU，将使用CPU处理")
            device = torch.device('cpu')
            perform_everything_on_device = False
        
        print(f"最终使用设备: {device}")
        print(f"perform_everything_on_device: {perform_everything_on_device}")

        # 更新进度
        tasks[task_id]["progress"] = 20
        tasks[task_id]["message"] = "初始化预测器..."

        # 初始化预测器
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=True,  # 增加详细程度以便调试
            verbose_preprocessing=True,  # 增加详细程度以便调试
            allow_tqdm=True
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=MODEL_FOLDER,
            use_folds=(0,),
            checkpoint_name='checkpoint_best.pth'
        )

        # 更新进度
        tasks[task_id]["progress"] = 40
        tasks[task_id]["message"] = "开始预测..."

        # 执行预测
        predictor.predict_from_files(
            list_of_lists_or_source_folder=input_dir,
            output_folder_or_list_of_truncated_output_files=output_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        
        # 更新进度
        tasks[task_id]["progress"] = 90
        tasks[task_id]["message"] = "生成轮廓图像..."
        
        # 获取预测结果
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        print(f"在输出目录中找到 {len(result_files)} 个结果文件: {result_files}")
        if not result_files:
            raise Exception("预测失败，未生成结果文件")
        
        result_file = os.path.join(output_dir, result_files[0])
        print(f"预测完成: {result_file}")
        
        # 根据dataset.json中的channel_names信息重命名文件
        print("根据dataset.json中的channel_names信息重命名文件...")
        renamed_files = rename_files_according_to_dataset(input_dir)
        
        # 生成带肿瘤轮廓的图像
        contour_output_dir = os.path.join(output_dir, "contour")
        os.makedirs(contour_output_dir, exist_ok=True)
        contour_image_path = None
        
        # 尝试使用重命名后的第一个文件作为原始影像生成轮廓
        if renamed_files:
            try:
                # 生成轮廓图像
                contour_image_path = os.path.join(contour_output_dir, f"contour_{result_files[0].replace('.nii.gz', '.png')}")
                print(f"生成轮廓图像: {contour_image_path}")
                generate_contour_image(renamed_files[0], result_file, contour_image_path)
                print(f"轮廓生成完成: {contour_image_path}")
            except Exception as e:
                print(f"生成轮廓图像失败：{str(e)}")
                import traceback
                traceback.print_exc()
                contour_image_path = None
        
        # 更新进度
        tasks[task_id]["progress"] = 100
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "任务完成"
        tasks[task_id]["result_file"] = result_file
        tasks[task_id]["contour_image"] = contour_image_path
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"预测失败: {str(e)}"

def rename_files_according_to_dataset(input_dir):
    """
    根据dataset.json中的channel_names信息重命名文件
    """
    # 加载dataset.json文件
    dataset_json_path = os.path.join(NNUNET_RAW, "Dataset044_BraTS2020", "dataset.json")
    if os.path.exists(dataset_json_path):
        import json
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        # 获取通道名称映射
        channel_names = dataset_info.get("channel_names", {})
        
        # 获取输入目录中的所有.nii和.nii.gz文件
        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.nii', '.nii.gz'))]
        
        print(f"所有文件: {all_files}")
        print(f"通道名称: {channel_names}")
        
        # 检查文件数量是否匹配预期的通道数
        expected_channels = len(channel_names)
        print(f"文件数量: {len(all_files)}, 期望通道数: {expected_channels}")
        
        if len(all_files) == expected_channels:
            print(f"检测到 {len(all_files)} 个文件，将根据dataset.json中的channel_names进行重命名")
            
            # 按照文件名排序，以便与channel_names的索引对应
            sorted_files = sorted(all_files)
            
            # 生成新的文件名列表，用于返回给调用者
            renamed_files = []
            
            # 重命名文件以符合nnU-Net的要求
            for idx, filename in enumerate(sorted_files):
                if str(idx) in channel_names:
                    channel_name = channel_names[str(idx)]
                    old_path = os.path.join(input_dir, filename)
                    
                    # 生成新文件名，格式为 caseId_0000.nii.gz, caseId_0001.nii.gz 等
                    # 使用原始文件名的前缀作为case_id，或者使用通用的case_id
                    base_name = os.path.splitext(filename)[0]
                    if '_' in base_name:
                        case_id = base_name.split('_')[0] + '_' + base_name.split('_')[1]  # 如 BraTS20_Training_001
                    else:
                        case_id = "BraTS2020_001"  # 默认case ID
                    
                    new_filename = f"{case_id}_{idx:04d}.nii.gz"
                    new_path = os.path.join(input_dir, new_filename)
                    
                    # 如果原文件不是.nii.gz格式，先转换
                    if filename.lower().endswith('.nii'):
                        new_path = convert_nii_to_nii_gz(old_path, new_path.replace('.nii.gz', '.nii'))
                    else:
                        # 确保目标文件名正确
                        if old_path != new_path:
                            os.rename(old_path, new_path)
                    
                    print(f"重命名文件: {filename} -> {new_filename} (对应通道: {channel_name})")
                    renamed_files.append(new_path)
                else:
                    print(f"警告: 索引 {idx} 在channel_names中未找到对应名称")
                    # 如果索引不在channel_names中，仍然添加到列表中但不重命名
                    old_path = os.path.join(input_dir, filename)
                    renamed_files.append(old_path)
            
            return renamed_files
        else:
            print(f"文件数量 ({len(all_files)}) 与期望的通道数 ({expected_channels}) 不匹配，跳过重命名")
            # 返回原始文件路径列表
            return [os.path.join(input_dir, f) for f in all_files]
    else:
        print(f"警告: 未找到dataset.json文件: {dataset_json_path}")
        # 返回原始文件路径列表
        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.nii', '.nii.gz'))]
        return [os.path.join(input_dir, f) for f in all_files]


# 根路径
@app.route('/')
def index():
    """根路径返回服务状态信息"""
    return jsonify({
        "code": 200,
        "msg": "nnU-Net后端服务运行正常",
        "data": {
            "version": "2.0.0",
            "endpoints": {
                "/": "服务状态检查",
                "/predict/file": "上传单个文件并执行预测（会自动生成带肿瘤轮廓的图像）",
                "/predict/file_async": "异步上传单个文件并执行预测",
                "/predict/folder_async": "异步上传多个文件并执行预测",
                "/task_status/<task_id>": "查询任务状态",
                "/download/<task_id>/<file_name>": "下载预测结果文件或轮廓图像"
            },
            "features": [
                "支持.nii和.nii.gz格式文件",
                "自动检测并转换文件格式",
                "自动生成带红色肿瘤轮廓的图像",
                "提供完整的RESTful API接口",
                "支持异步任务处理，避免超时问题"
            ]
        }
    })

# 前端页面
@app.route('/ui')
def ui():
    """返回前端UI页面"""
    return send_file('index.html')

# 查询任务状态
@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """查询任务状态"""
    if task_id not in tasks:
        return jsonify({
            "code": 404,
            "msg": "任务不存在"
        }), 404
    
    return jsonify({
        "code": 200,
        "msg": "查询成功",
        "data": tasks[task_id]
    })

# 预测端点 - 支持单个文件上传（异步）
@app.route('/predict/file_async', methods=['POST'])
def predict_file_async():
    """异步上传单个文件并执行预测"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({
                "code": 400,
                "msg": "缺少上传文件"
            }), 400
        
        # 生成唯一ID
        task_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化任务状态
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "等待处理...",
            "created_at": datetime.now().isoformat(),
            "result_file": None,
            "contour_image": None
        }
        
        # 保存上传的文件
        file = request.files['file']
        file_path = os.path.join(output_dir, file.filename)
        file.save(file_path)
        
        # 更新任务状态
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 5
        tasks[task_id]["message"] = "文件已接收，准备处理..."
        
        # 在后台线程中处理文件
        thread = threading.Thread(target=process_single_file, args=(file_path, output_dir, task_id))
        thread.start()
        
        # 返回任务ID
        return jsonify({
            "code": 200,
            "msg": "文件已接收，正在后台处理",
            "data": {
                "task_id": task_id,
                "status_url": f"/task_status/{task_id}",
                "uploaded_files": 1
            }
        })
        
    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": f"预测失败：{str(e)}"
        }), 500

# 预测端点 - 支持单个文件上传（同步）
@app.route('/predict/file', methods=['POST'])
def predict_file():
    """上传单个文件并执行预测"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({
                "code": 400,
                "msg": "缺少上传文件"
            }), 400
        
        # 生成唯一ID
        task_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存上传的文件
        file = request.files['file']
        file_path = os.path.join(output_dir, file.filename)
        file.save(file_path)
        
        # 处理文件，执行预测并生成轮廓图像
        result_file, contour_image = process_single_file(file_path, output_dir)
        
        # 构建返回结果
        result_file_name = os.path.basename(result_file)
        contour_file_name = os.path.basename(contour_image) if contour_image else None
        
        # 返回结果
        return jsonify({
            "code": 200,
            "msg": "预测成功",
            "data": {
                "task_id": task_id,
                "result_url": f"/download/{task_id}/{result_file_name}",
                "result_file": result_file_name,
                "contour_url": f"/download/{task_id}/contour/{contour_file_name}" if contour_file_name else None,
                "contour_file": contour_file_name,
                "uploaded_files": 1  # 添加这个字段以保持与多文件上传响应的一致性
            }
        })
        
    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": f"预测失败：{str(e)}"
        }), 500

# 预测端点 - 支持多文件（文件夹）上传（异步）
@app.route('/predict/folder_async', methods=['POST'])
def predict_folder_async():
    """异步上传多个文件（文件夹）并执行预测，支持四模态图处理"""
    try:
        print("\n=== 接收到文件夹上传请求 ===")
        
        # 检查是否有文件上传
        if 'files' not in request.files:
            print("错误：缺少上传文件")
            return jsonify({
                "code": 400,
                "msg": "缺少上传文件"
            }), 400
        
        files = request.files.getlist('files')
        print(f"接收到 {len(files)} 个文件")
        
        if not files:
            print("错误：未选择任何文件")
            return jsonify({
                "code": 400,
                "msg": "请选择要上传的文件"
            }), 400
        
        # 生成唯一ID
        task_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(OUTPUT_DIR, task_id)
        input_dir = os.path.join(TEMP_DIR, task_id, "input")
        
        print(f"任务ID: {task_id}")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        
        # 初始化任务状态
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "等待处理...",
            "created_at": datetime.now().isoformat(),
            "result_file": None,
            "contour_image": None
        }
        
        # 创建目录
        try:
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            print("目录创建成功")
        except Exception as e:
            print(f"目录创建失败: {str(e)}")
            return jsonify({
                "code": 500,
                "msg": f"目录创建失败：{str(e)}"
            }), 500
        
        # 保存上传的所有文件
        uploaded_files = []
        for i, file in enumerate(files):
            if file.filename:
                print(f"保存文件 {i+1}/{len(files)}: {file.filename}")
                
                # 处理文件名中的路径部分，只使用纯文件名
                pure_filename = os.path.basename(file.filename)
                file_path = os.path.join(input_dir, pure_filename)
                
                try:
                    file.save(file_path)
                    uploaded_files.append(file_path)
                    print(f"文件保存成功: {file_path}")
                    
                    # 判断文件类型并进行预处理
                    print(f"检查文件类型: {pure_filename}")
                    if not (pure_filename.lower().endswith('.nii.gz') or pure_filename.lower().endswith('.nii')):
                        raise Exception(f"不支持的文件格式: {pure_filename}，仅支持.nii或.nii.gz格式")
                    
                    # 如果是.nii文件，转换为.nii.gz格式
                    if pure_filename.lower().endswith('.nii'):
                        print(f"转换文件格式: {pure_filename} -> .nii.gz")
                        convert_nii_to_nii_gz(file_path)
                        print("格式转换成功")
                    elif pure_filename.lower().endswith('.nii.gz'):
                        print("文件已是.nii.gz格式，无需转换")
                        
                except Exception as e:
                    print(f"保存文件失败: {str(e)}")
                    return jsonify({
                        "code": 500,
                        "msg": f"保存文件失败：{str(e)}"
                    }), 500
        
        if not uploaded_files:
            print("错误：未成功上传任何文件")
            return jsonify({
                "code": 400,
                "msg": "未成功上传任何文件"
            }), 400
        
        print(f"成功上传 {len(uploaded_files)} 个文件")
        
        # 根据dataset.json中的channel_names信息重命名文件
        print("根据dataset.json中的channel_names信息重命名文件...")
        renamed_files = rename_files_according_to_dataset(input_dir)
        
        # 更新任务状态
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 5
        tasks[task_id]["message"] = f"已接收 {len(uploaded_files)} 个文件，准备处理..."
        
        # 在后台线程中运行预测
        thread = threading.Thread(target=run_prediction_folder_async, args=(input_dir, output_dir, task_id, uploaded_files))
        thread.start()
        
        # 返回任务ID
        return jsonify({
            "code": 200,
            "msg": "文件已接收，正在后台处理",
            "data": {
                "task_id": task_id,
                "status_url": f"/task_status/{task_id}",
                "uploaded_files": len(uploaded_files)
            }
        })
        
    except Exception as e:
        print(f"发生未捕获的异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "code": 500,
            "msg": f"预测失败：{str(e)}"
        }), 500

# 预测端点 - 支持多文件（文件夹）上传（同步）
@app.route('/predict/folder', methods=['POST'])
def predict_folder():
    """上传多个文件（文件夹）并执行预测，支持四模态图处理"""
    try:
        print("\n=== 接收到文件夹上传请求 ===")
        
        # 检查是否有文件上传
        if 'files' not in request.files:
            print("错误：缺少上传文件")
            return jsonify({
                "code": 400,
                "msg": "缺少上传文件"
            }), 400
        
        files = request.files.getlist('files')
        print(f"接收到 {len(files)} 个文件")
        
        if not files:
            print("错误：未选择任何文件")
            return jsonify({
                "code": 400,
                "msg": "请选择要上传的文件"
            }), 400
        
        # 生成唯一ID
        task_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join(OUTPUT_DIR, task_id)
        input_dir = os.path.join(TEMP_DIR, task_id, "input")
        
        print(f"任务ID: {task_id}")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        
        # 初始化任务状态（对于同步处理，我们也需要初始化状态以避免错误）
        tasks[task_id] = {
            "task_id": task_id,
            "status": "running",
            "progress": 0,
            "message": "初始化预测...",
            "created_at": datetime.now().isoformat(),
            "result_file": None,
            "contour_image": None
        }
        
        # 创建目录
        try:
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            print("目录创建成功")
        except Exception as e:
            print(f"目录创建失败: {str(e)}")
            return jsonify({
                "code": 500,
                "msg": f"目录创建失败：{str(e)}"
            }), 500
        
        # 保存上传的所有文件
        uploaded_files = []
        for i, file in enumerate(files):
            if file.filename:
                print(f"保存文件 {i+1}/{len(files)}: {file.filename}")
                
                # 处理文件名中的路径部分，只使用纯文件名
                pure_filename = os.path.basename(file.filename)
                file_path = os.path.join(input_dir, pure_filename)
                
                try:
                    file.save(file_path)
                    uploaded_files.append(file_path)
                    print(f"文件保存成功: {file_path}")
                    
                    # 判断文件类型并进行预处理
                    print(f"检查文件类型: {pure_filename}")
                    if not (pure_filename.lower().endswith('.nii.gz') or pure_filename.lower().endswith('.nii')):
                        raise Exception(f"不支持的文件格式: {pure_filename}，仅支持.nii或.nii.gz格式")
                    
                    # 如果是.nii文件，转换为.nii.gz格式
                    if pure_filename.lower().endswith('.nii'):
                        print(f"转换文件格式: {pure_filename} -> .nii.gz")
                        convert_nii_to_nii_gz(file_path)
                        print("格式转换成功")
                    elif pure_filename.lower().endswith('.nii.gz'):
                        print("文件已是.nii.gz格式，无需转换")
                        
                except Exception as e:
                    print(f"保存文件失败: {str(e)}")
                    return jsonify({
                        "code": 500,
                        "msg": f"保存文件失败：{str(e)}"
                    }), 500
        
        if not uploaded_files:
            print("错误：未成功上传任何文件")
            return jsonify({
                "code": 400,
                "msg": "未成功上传任何文件"
            }), 400
        
        print(f"成功上传 {len(uploaded_files)} 个文件")
        
        # 根据dataset.json中的channel_names信息重命名文件
        print("根据dataset.json中的channel_names信息重命名文件...")
        renamed_files = rename_files_according_to_dataset(input_dir)
        
        # 运行预测
        print(f"开始预测多文件: {len(uploaded_files)} 个文件")
        try:
            run_prediction(input_dir, output_dir, task_id)
            print("预测执行成功")
        except Exception as e:
            print(f"预测执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "code": 500,
                "msg": f"预测执行失败：{str(e)}"
            }), 500
        
        # 获取预测结果
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        print(f"生成的结果文件: {result_files}")
        
        if not result_files:
            print("错误：预测失败，未生成结果文件")
            return jsonify({
                "code": 500,
                "msg": "预测失败，未生成结果文件"
            }), 500
        
        result_file = os.path.join(output_dir, result_files[0])
        print(f"预测完成: {result_file}")
        
        # 生成带肿瘤轮廓的图像
        contour_output_dir = os.path.join(output_dir, "contour")
        os.makedirs(contour_output_dir, exist_ok=True)
        contour_image_path = None
        
        # 尝试使用重命名后的第一个文件作为原始影像生成轮廓
        if renamed_files:
            try:
                # 生成轮廓图像
                contour_image_path = os.path.join(contour_output_dir, f"contour_{result_files[0].replace('.nii.gz', '.png')}")
                print(f"生成轮廓图像: {contour_image_path}")
                generate_contour_image(renamed_files[0], result_file, contour_image_path)
                print(f"轮廓生成完成: {contour_image_path}")
            except Exception as e:
                print(f"生成轮廓图像失败：{str(e)}")
                import traceback
                traceback.print_exc()
                contour_image_path = None
        
        # 构建返回结果
        result_file_name = os.path.basename(result_file)
        contour_file_name = os.path.basename(contour_image_path) if contour_image_path else None
        
        print(f"返回结果: 任务ID={task_id}, 结果文件={result_file_name}, 轮廓文件={contour_file_name}")
        
        # 返回结果
        return jsonify({
            "code": 200,
            "msg": "预测成功",
            "data": {
                "task_id": task_id,
                "result_url": f"/download/{task_id}/{result_file_name}",
                "result_file": result_file_name,
                "contour_url": f"/download/{task_id}/contour/{contour_file_name}" if contour_file_name else None,
                "contour_file": contour_file_name,
                "uploaded_files": len(uploaded_files)
            }
        })
        
    except Exception as e:
        print(f"发生未捕获的异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "code": 500,
            "msg": f"预测失败：{str(e)}"
        }), 500

# 下载端点
@app.route('/download/<task_id>/<path:file_name>')
def download_result(task_id, file_name):
    """下载预测结果文件或轮廓图像"""
    try:
        # 检查是否是轮廓图像
        if file_name.startswith("contour_") and file_name.endswith(".png"):
            file_path = os.path.join(OUTPUT_DIR, task_id, "contour", file_name)
        else:
            file_path = os.path.join(OUTPUT_DIR, task_id, file_name)
        
        if not os.path.exists(file_path):
            return jsonify({
                "code": 404,
                "msg": "文件不存在"
            }), 404
        
        return send_file(file_path, as_attachment=True, download_name=file_name)
    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": f"下载失败：{str(e)}"
        }), 500

if __name__ == "__main__":
    try:
        print("正在启动nnU-Net后端服务...")
        print(f"模型路径: {MODEL_FOLDER}")
        print(f"临时目录: {TEMP_DIR}")
        print(f"输出目录: {OUTPUT_DIR}")
        print("服务将在 http://localhost:8080 启动")
        app.run(host='0.0.0.0', port=8080, debug=False)
    except Exception as e:
        print(f"服务启动失败：{str(e)}")
        import traceback
        traceback.print_exc()
