import pymysql
from datetime import datetime
import os
import json

# MySQL 数据库配置（根据你的实际情况修改）
DB_CONFIG = {
    'host': 'localhost',  # MySQL 服务器地址
    'user': 'root',  # MySQL 用户名
    'password': '123456',  # MySQL 密码
    'database': 'medical_Platform',  # 数据库名（保持你想要的格式）
    'charset': 'utf8mb4'  # 字符集，支持所有中文和特殊字符
}


def init_mysql_db():
    """
    初始化MySQL数据库
    """
    # 1. 先连接MySQL服务器（不指定数据库，用于创建数据库）
    conn = None
    try:
        # 连接MySQL服务器
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG['charset']
        )
        cursor = conn.cursor()

        # 创建数据库（如果不存在）
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} DEFAULT CHARACTER SET {DB_CONFIG['charset']}")
        print(f"数据库 {DB_CONFIG['database']} 创建/检查完成")

        # 2. 重新连接到指定的数据库
        conn.close()
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 3. 创建用户表
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
        print("users 表创建/检查完成")

        # 4. 创建患者详细信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_profiles (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT UNIQUE,
                full_name VARCHAR(100),
                age INT,
                gender VARCHAR(10),
                medical_history TEXT,
                allergies TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        print("patient_profiles 表创建/检查完成")

        # 5. 创建医生详细信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctor_profiles (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT UNIQUE,
                full_name VARCHAR(100),
                specialty VARCHAR(100),
                license_number VARCHAR(50),
                hospital_affiliation VARCHAR(100),
                years_experience INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        print("doctor_profiles 表创建/检查完成")

        # 6. 创建消息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INT PRIMARY KEY AUTO_INCREMENT,
                sender_id INT NOT NULL,
                receiver_id INT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_read TINYINT(1) DEFAULT 0,
                FOREIGN KEY (sender_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (receiver_id) REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        print("messages 表创建/检查完成")

        # 7. 创建报告表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INT PRIMARY KEY AUTO_INCREMENT,
                patient VARCHAR(50) NOT NULL,
                doctor VARCHAR(50) NOT NULL,
                result_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        print("reports 表创建/检查完成")

        # 8. 创建会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INT PRIMARY KEY AUTO_INCREMENT,
                participant1_id INT NOT NULL,
                participant2_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_message_time TIMESTAMP NULL,
                FOREIGN KEY (participant1_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (participant2_id) REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        print("conversations 表创建/检查完成")

        # 9. 创建肿瘤分割记录表
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
        print("segmentation_records 表创建/检查完成")

        # 提交表结构创建的更改
        conn.commit()
        print("\n所有数据表创建/检查完成")

        # 9. 插入示例数据
        print("\n开始插入示例数据...")

        # 插入医生用户
        cursor.execute('''
            INSERT INTO users (username, password, user_type, email, phone)
            VALUES (%s, %s, %s, %s, %s)
        ''', ('zhang_doctor', 'password123', 'doctor', 'zhang@hospital.com', '13800138001'))

        # 获取医生ID
        doctor_id = cursor.lastrowid
        print(f"医生用户创建成功，ID: {doctor_id}")

        # 插入医生档案
        cursor.execute('''
            INSERT INTO doctor_profiles (user_id, full_name, specialty, license_number, hospital_affiliation, years_experience)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (doctor_id, '张医生', '神经外科', 'DOC123456', '第一人民医院', 10))
        print("医生档案创建成功")

        # 插入患者用户（张三）
        cursor.execute('''
            INSERT INTO users (username, password, user_type, email, phone)
            VALUES (%s, %s, %s, %s, %s)
        ''', ('patient_li', 'password123', 'patient', 'li@patient.com', '13800138002'))

        # 获取患者ID
        patient_id = cursor.lastrowid
        print(f"患者用户创建成功，ID: {patient_id}")

        # 插入患者档案
        cursor.execute('''
            INSERT INTO patient_profiles (user_id, full_name, age, gender, medical_history)
            VALUES (%s, %s, %s, %s, %s)
        ''', (patient_id, '张三', 45, '男', '高血压'))
        print("患者档案创建成功")

        # 插入其他测试患者
        test_patients = [
            ('patient_wang', '李四', '13800138003'),
            ('patient_zhao', '王五', '13800138004'),
            ('patient_sun', '赵六', '13800138005'),
            ('patient_qi', '孙七', '13800138006')
        ]

        for i, (username, fullname, phone) in enumerate(test_patients):
            cursor.execute('''
                INSERT INTO users (username, password, user_type, email, phone)
                VALUES (%s, %s, %s, %s, %s)
            ''', (username, 'password123', 'patient', f'{username}@test.com', phone))

            new_patient_id = cursor.lastrowid

            cursor.execute('''
                INSERT INTO patient_profiles (user_id, full_name, age, gender)
                VALUES (%s, %s, %s, %s)
            ''', (new_patient_id, fullname, 35 + i, '男'))
            print(f"患者 {fullname} 创建成功")

        # 插入测试消息
        from datetime import timedelta

        # 张三发送的第一条消息（30分钟前）
        cursor.execute('''
            INSERT INTO messages (sender_id, receiver_id, message, timestamp, is_read)
            VALUES (%s, %s, %s, %s, %s)
        ''', (patient_id, doctor_id, '医生您好，我是张三，想咨询一下我的检查结果。',
              datetime.now() - timedelta(minutes=30), 1))

        # 张医生回复（25分钟前）
        cursor.execute('''
            INSERT INTO messages (sender_id, receiver_id, message, timestamp, is_read)
            VALUES (%s, %s, %s, %s, %s)
        ''', (doctor_id, patient_id, '您好张先生，我已经查看了您的CT扫描结果，初步判断有异常，建议您进一步检查。',
              datetime.now() - timedelta(minutes=25), 1))

        # 张三再次询问（20分钟前）
        cursor.execute('''
            INSERT INTO messages (sender_id, receiver_id, message, timestamp, is_read)
            VALUES (%s, %s, %s, %s, %s)
        ''', (patient_id, doctor_id, '好的医生，那我需要做什么检查呢？',
              datetime.now() - timedelta(minutes=20), 0))

        print("测试消息创建成功")

        # 插入测试报告
        test_report_1 = {
            "predicted_class": "tumor",
            "confidence": 0.925,
            "all_probabilities": {
                "healthy": 0.075,
                "tumor": 0.925
            }
        }

        cursor.execute('''
            INSERT INTO reports (patient, doctor, result_data, timestamp)
            VALUES (%s, %s, %s, %s)
        ''', ('patient_li', 'zhang_doctor', json.dumps(test_report_1), datetime.now() - timedelta(minutes=30)))

        test_report_2 = {
            "predicted_class": "healthy",
            "confidence": 0.872,
            "all_probabilities": {
                "healthy": 0.872,
                "tumor": 0.128
            }
        }

        cursor.execute('''
            INSERT INTO reports (patient, doctor, result_data, timestamp)
            VALUES (%s, %s, %s, %s)
        ''', ('patient_wang', 'zhang_doctor', json.dumps(test_report_2), datetime.now() - timedelta(hours=1)))

        print("测试报告创建成功")

        conn.commit()

        print("\n" + "=" * 60)
        print("✅ 数据库初始化完成！")
        print("=" * 60)
        print(f"\n📝 数据库名称: {DB_CONFIG['database']}")
        print("\n📝 测试账号信息:")
        print("   👨‍⚕️ 医生账号: zhang_doctor")
        print("   🔑 医生密码: password123")
        print("   👤 患者账号: patient_li (张三)")
        print("   🔑 患者密码: password123")
        print("   👥 其他患者: patient_wang (李四), patient_zhao (王五), patient_sun (赵六), patient_qi (孙七)")
        print("   🔑 其他患者密码: password123 (所有患者密码统一)")
        print("=" * 60)

    except pymysql.Error as e:
        print(f"数据库初始化出错: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def get_db_connection():
    """
    获取MySQL数据库连接
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        conn.cursorclass = pymysql.cursors.DictCursor
        return conn
    except pymysql.Error as e:
        print(f"获取数据库连接失败: {e}")
        raise


if __name__ == "__main__":
    print("开始初始化医疗平台MySQL数据库...")
    print(f"目标数据库: {DB_CONFIG['database']}")
    init_mysql_db()