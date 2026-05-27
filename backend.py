import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pathlib import Path
import base64
import json
from datetime import datetime
import pymysql
import time
# 设置环境变量以避免SSL密钥日志文件权限问题
os.environ['SSLKEYLOGFILE'] = ''

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Mysql数据库配置
DB_CONFIG = {
    'host': 'localhost',      # MySQL 服务器地址
    'user': 'root',           # MySQL 用户名（建议用专用用户）
    'password': '123456',  # MySQL 密码
    'database': 'medical_platform',  # 数据库名
    'charset': 'utf8mb4'      # 字符集，支持所有中文
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保上传目录存在
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

def init_db():
    """初始化数据库表"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 创建用户表（MySQL语法）
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    user_type ENUM('patient', 'doctor') NOT NULL,
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

        # 创建消息表
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    sender_id INT NOT NULL,
                    receiver_id INT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_read TINYINT(1) DEFAULT 0,  # MySQL用TINYINT(1)替代BOOLEAN
                    FOREIGN KEY (sender_id) REFERENCES users (id) ON DELETE CASCADE,
                    FOREIGN KEY (receiver_id) REFERENCES users (id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

        # 创建报告表
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    patient VARCHAR(50) NOT NULL,
                    doctor VARCHAR(50) NOT NULL,
                    result_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

        # 创建肿瘤分割记录表
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS segmentation_records (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    doctor VARCHAR(50) NOT NULL,
                    patient_name VARCHAR(100) NOT NULL,
                    image_url TEXT NOT NULL,
                    file_count INT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

        conn.commit()
        print("MySQL数据库表初始化完成")
    except pymysql.Error as e:
        print(f"初始化数据库表失败: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_db_connection():
    """获取数据库连接"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        # 设置游标为字典类型，使结果可以通过列名访问（替代 sqlite3.Row）
        conn.cursorclass = pymysql.cursors.DictCursor
        return conn
    except pymysql.Error as e:
        print(f"获取数据库连接失败: {e}")
        raise

# 加载训练好的模型
model_path = "best_brain_tumor_classifier.pt"  # 使用训练好的模型
model = None

# 设置环境变量以避免SSL密钥日志文件权限问题
os.environ['SSLKEYLOGFILE'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print("模型加载成功")
    else:
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 yolo11_brain_tumor_classifier.py 完成模型训练")
        # 尝试加载预训练模型作为备用
        backup_model_path = "yolo11n-cls.pt"
        if os.path.exists(backup_model_path):
            model = YOLO(backup_model_path)
            print("使用备份模型进行演示")
        else:
            print("未找到任何可用模型")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("尝试使用CPU模式加载模型...")
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path, device='cpu')
            print("模型在CPU模式下加载成功")
        elif os.path.exists(backup_model_path):
            model = YOLO(backup_model_path, device='cpu')
            print("备份模型在CPU模式下加载成功")
    except Exception as e2:
        print(f"CPU模式下模型加载也失败: {e2}")

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """预处理图像"""
    image = Image.open(image_path)
    return image

def predict_image(image_path):
    """对图像进行预测"""
    global model
    if model is None:
        return {"error": "模型未加载，请检查模型文件是否存在"}

    try:
        # 使用模型进行预测
        results = model(image_path)

        # 获取预测结果
        probs = results[0].probs  # 获取概率
        top1_idx = probs.top1     # 最可能的类别索引
        top1_conf = probs.top1conf.item()  # 最可能类别的置信度
        class_names = results[0].names  # 类别名称字典
        top1_class = class_names[top1_idx]  # 最可能的类别名称

        # 获取所有类别的概率
        all_probs = probs.data.cpu().numpy()
        class_probabilities = {}
        for idx, prob in enumerate(all_probs):
            class_name = class_names[idx]
            class_probabilities[class_name] = float(prob)

        return {
            "predicted_class": top1_class,
            "confidence": round(top1_conf, 4),
            "all_probabilities": class_probabilities,
            "top_prediction": {
                "class": top1_class,
                "confidence": round(top1_conf, 4)
            }
        }
    except Exception as e:
        return {"error": f"预测过程中出现错误: {str(e)}"}

# 医患交流平台相关路由
@app.route('/')
def index():
    """返回前端页面"""
    with open('frontend/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/chat')
def chat_page():
    """返回聊天页面"""
    with open('frontend/chat.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/doctor')
def doctor_page():
    """返回医生平台页面"""
    with open('frontend/doctor.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    sender_username = data.get('sender')
    receiver_username = data.get('receiver')
    message = data.get('message')
    timestamp = data.get('timestamp')  # 前端可能会传时间戳

    if not sender_username or not receiver_username or not message:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 获取发送者ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (sender_username,))
        sender_row = cursor.fetchone()
        if not sender_row:
            return jsonify({'success': False, 'error': '发送者用户不存在'})
        sender_id = sender_row['id']

        # 获取接收者ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (receiver_username,))
        receiver_row = cursor.fetchone()
        if not receiver_row:
            return jsonify({'success': False, 'error': '接收者用户不存在'})
        receiver_id = receiver_row['id']

        # 处理时间戳
        if timestamp:
            # 如果是 ISO 格式 (2026-02-14T01:48:25.950Z)
            if 'T' in timestamp:
                # 转换为 MySQL 格式: 2026-02-14 01:48:25
                mysql_timestamp = timestamp.replace('T', ' ').replace('Z', '').split('.')[0]
            else:
                mysql_timestamp = timestamp
            cursor.execute('''INSERT INTO messages (sender_id, receiver_id, message, timestamp)
                             VALUES (%s, %s, %s, %s)''',
                           (sender_id, receiver_id, message, mysql_timestamp))
        else:
            # 如果没有传时间戳，使用 NOW()
            cursor.execute('''INSERT INTO messages (sender_id, receiver_id, message, timestamp)
                             VALUES (%s, %s, %s, NOW())''',
                           (sender_id, receiver_id, message))

        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"发送消息错误: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()


@app.route('/send_report_to_doctor', methods=['POST'])
def send_report_to_doctor():
    data = request.json
    patient = data.get('patient')
    doctor = data.get('doctor')
    result = data.get('result')
    timestamp = data.get('timestamp')

    if not patient or not doctor or not result:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    # 处理时间戳格式问题
    mysql_timestamp = None
    if timestamp:
        try:
            # 如果是 ISO 格式 (2026-02-14T01:48:25.950Z)
            if 'T' in timestamp:
                # 转换为 MySQL 格式: 2026-02-14 01:48:25
                mysql_timestamp = timestamp.replace('T', ' ').replace('Z', '').split('.')[0]
            else:
                mysql_timestamp = timestamp
        except:
            mysql_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        mysql_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 将检测报告插入数据库
    conn = get_db_connection()
    c = conn.cursor()
    try:
        # 插入报告记录
        c.execute('''INSERT INTO reports (patient, doctor, result_data, timestamp)
                     VALUES (%s, %s, %s, %s)''',
                  (patient, doctor, json.dumps(result), mysql_timestamp))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"发送报告错误: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/get_messages', methods=['POST'])
def get_messages():
    """获取指定用户的聊天消息"""
    try:
        data = request.get_json()
        user1_username = data.get('user1', '')  # 当前用户
        user2_username = data.get('user2', '')  # 对话对象

        if not user1_username or not user2_username:
            return jsonify({"success": False, "error": "缺少必要参数"}), 400

        # 从数据库获取用户ID
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取两个用户的ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (user1_username,))
        user1_row = cursor.fetchone()
        if not user1_row:
            conn.close()
            return jsonify({"success": False, "error": "用户1不存在"}), 400
        user1_id = user1_row['id']

        cursor.execute("SELECT id FROM users WHERE username = %s", (user2_username,))
        user2_row = cursor.fetchone()
        if not user2_row:
            conn.close()
            return jsonify({"success": False, "error": "用户2不存在"}), 400
        user2_id = user2_row['id']

        # 获取两个用户之间的消息
        cursor.execute('''
            SELECT m.*, u1.username as sender_username, u2.username as receiver_username
            FROM messages m
            JOIN users u1 ON m.sender_id = u1.id
            JOIN users u2 ON m.receiver_id = u2.id
            WHERE (m.sender_id = %s AND m.receiver_id = %s) OR (m.sender_id = %s AND m.receiver_id = %s)
            ORDER BY m.timestamp ASC
        ''', (user1_id, user2_id, user2_id, user1_id))
        messages = cursor.fetchall()
        conn.close()

        # 转换为字典列表
        messages_list = []
        for msg in messages:
            messages_list.append({
                "id": msg['id'],
                "sender_id": msg['sender_id'],
                "receiver_id": msg['receiver_id'],
                "sender_username": msg['sender_username'],
                "receiver_username": msg['receiver_username'],
                "message": msg['message'],
                "timestamp": msg['timestamp'],
                "is_read": bool(msg['is_read'])
            })

        return jsonify({
            "success": True,
            "messages": messages_list
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/register_user', methods=['POST'])
def register_user():
    """注册新用户（患者或医生）"""
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        user_type = data.get('user_type', 'patient')  # 'patient' 或 'doctor'
        email = data.get('email', '')
        phone = data.get('phone', '')
        additional_info = data.get('additional_info', {})

        if not username or not password:
            return jsonify({"success": False, "error": "用户名和密码不能为空"}), 400

        # 检查用户是否已存在
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            conn.close()
            return jsonify({"success": False, "error": "用户名已存在"}), 400

        # 创建新用户
        cursor.execute('''
            INSERT INTO users (username, password, user_type, email, phone)
            VALUES (%s, %s, %s, %s, %s)
        ''', (username, password, user_type, email, phone))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "user_id": user_id,
            "username": username,
            "user_type": user_type
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')

        if not username or not password:
            return jsonify({"success": False, "error": "用户名和密码不能为空"}), 400

        # 查找用户
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, user_type, email, phone
            FROM users WHERE username = %s AND password = %s
        ''', (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({
                "success": True,
                "user_id": user['id'],
                "username": user['username'],
                "user_type": user['user_type'],
                "email": user['email'],
                "phone": user['phone']
            })
        else:
            return jsonify({"success": False, "error": "用户名或密码错误"}), 401

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """处理图像预测请求"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "没有文件被上传"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "不支持的文件格式"}), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 进行预测
        result = predict_image(filepath)

        # 删除临时文件
        try:
            os.remove(filepath)
        except:
            pass  # 如果删除失败，忽略错误

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"处理请求时出现错误: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """健康检查端点"""
    # 检查数据库连接
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = cursor.fetchone()['count']
        cursor.execute("SELECT COUNT(*) as count FROM messages")
        message_count = cursor.fetchone()['count']
        conn.close()
    except Exception as e:
        user_count = 0
        message_count = 0

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message_count": message_count,
        "user_count": user_count,
        "database_connected": True
    })


@app.route('/get_doctor_dashboard', methods=['POST'])
def get_doctor_dashboard():
    """获取医生仪表盘数据（最新报告和待办事项）"""
    try:
        data = request.get_json()
        doctor_username = data.get('doctor')

        if not doctor_username:
            return jsonify({'success': False, 'error': '缺少医生名称'})

        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取该医生的最新报告
        cursor.execute('''
            SELECT patient, result_data, timestamp 
            FROM reports 
            WHERE doctor = %s 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', (doctor_username,))
        reports = cursor.fetchall()

        # 解析报告数据
        parsed_reports = []
        for report in reports:
            try:
                result = json.loads(report['result_data'])
                # 格式化时间
                timestamp = report['timestamp']
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime('%H:%M')
                else:
                    time_str = str(timestamp)

                parsed_reports.append({
                    'patient': report['patient'],
                    'result': result,
                    'timestamp': time_str,
                    'display_text': f"{report['patient']}: 脑瘤检测: {result.get('predicted_class', '未知')} (置信度: {result.get('confidence', 0) * 100:.1f}%)"
                })
            except:
                continue

        conn.close()

        return jsonify({
            'success': True,
            'reports': parsed_reports
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_todo_items', methods=['GET'])
def get_todo_items():
    """获取待办事项"""
    try:
        # 这里可以从数据库获取，现在先返回静态数据
        todo_items = [
            {'id': 1, 'text': '审核患者A的CT扫描结果', 'status': 'pending'},
            {'id': 2, 'text': '安排患者C的进一步检查', 'status': 'pending'},
            {'id': 3, 'text': '与患者B预约复诊时间', 'status': 'pending'},
            {'id': 4, 'text': '更新患者D的病历记录', 'status': 'pending'}
        ]
        return jsonify({'success': True, 'todos': todo_items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/add_todo_item', methods=['POST'])
def add_todo_item():
    """添加待办事项"""
    try:
        data = request.get_json()
        text = data.get('text')
        # 这里可以保存到数据库
        return jsonify({'success': True, 'id': int(time.time())})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/toggle_todo_item', methods=['POST'])
def toggle_todo_item():
    """切换待办事项状态"""
    try:
        data = request.get_json()
        todo_id = data.get('id')
        # 这里可以更新数据库
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete_todo_item', methods=['POST'])
def delete_todo_item():
    """删除待办事项"""
    try:
        data = request.get_json()
        todo_id = data.get('id')
        # 这里可以从数据库删除
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/save_segmentation_record', methods=['POST'])
def save_segmentation_record():
    """保存肿瘤分割记录到数据库"""
    try:
        data = request.get_json()
        doctor = data.get('doctor')
        patient_name = data.get('patient_name')
        image_url = data.get('image_url')
        file_count = data.get('file_count')

        if not doctor or not patient_name or not image_url:
            return jsonify({'success': False, 'error': '缺少必要参数'})

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO segmentation_records (doctor, patient_name, image_url, file_count)
                VALUES (%s, %s, %s, %s)
            ''', (doctor, patient_name, image_url, file_count))
            conn.commit()
            return jsonify({'success': True, 'id': cursor.lastrowid})
        except Exception as e:
            print(f"保存分割记录错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_segmentation_records', methods=['POST'])
def get_segmentation_records():
    """获取医生的肿瘤分割记录"""
    try:
        data = request.get_json()
        doctor = data.get('doctor')

        if not doctor:
            return jsonify({'success': False, 'error': '缺少医生名称'})

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT id, doctor, patient_name, image_url, file_count, timestamp
                FROM segmentation_records
                WHERE doctor = %s
                ORDER BY timestamp DESC
            ''', (doctor,))
            records = cursor.fetchall()
            conn.close()

            return jsonify({
                'success': True,
                'records': records
            })
        except Exception as e:
            print(f"获取分割记录错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/rename_segmentation_record', methods=['POST'])
def rename_segmentation_record():
    """重命名肿瘤分割记录"""
    try:
        data = request.get_json()
        record_id = data.get('id')
        new_patient_name = data.get('new_patient_name')

        if not record_id or not new_patient_name:
            return jsonify({'success': False, 'error': '缺少必要参数'})

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE segmentation_records
                SET patient_name = %s
                WHERE id = %s
            ''', (new_patient_name, record_id))
            conn.commit()
            return jsonify({'success': True})
        except Exception as e:
            print(f"重命名分割记录错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete_segmentation_record', methods=['POST'])
def delete_segmentation_record():
    """删除肿瘤分割记录"""
    try:
        data = request.get_json()
        record_id = data.get('id')

        if not record_id:
            return jsonify({'success': False, 'error': '缺少记录ID'})

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                DELETE FROM segmentation_records
                WHERE id = %s
            ''', (record_id,))
            conn.commit()
            return jsonify({'success': True})
        except Exception as e:
            print(f"删除分割记录错误: {e}")
            return jsonify({'success': False, 'error': str(e)})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # 初始化数据库
    init_db()
    print("启动脑瘤检测及医患交流系统...")
    print("访问 http://localhost:5000 查看登录平台")
    print("访问 http://localhost:5000/chat 查看患者平台")
    print("访问 http://localhost:5000/doctor 查看医生平台")
    app.run(debug=True, host='0.0.0.0', port=5000)
