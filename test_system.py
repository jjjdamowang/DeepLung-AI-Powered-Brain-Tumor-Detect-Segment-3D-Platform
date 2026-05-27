import os
import requests
from PIL import Image
import io

# 修复环境变量设置
if 'SSLKEYLOGFILE' in os.environ:
    del os.environ['SSLKEYLOGFILE']

def test_system():
    """
    测试完整的脑瘤检测和医患交流系统
    """
    print("开始测试脑瘤检测和医患交流系统...")
    
    # 测试1: 检查服务器健康状态
    print("\n1. 测试服务器健康状态...")
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ 服务器状态: {health_data['status']}")
            print(f"   ✓ 模型加载状态: {health_data['model_loaded']}")
            print(f"   ✓ 用户数量: {health_data['user_count']}")
            print(f"   ✓ 消息数量: {health_data['message_count']}")
        else:
            print(f"   ✗ 健康检查失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 健康检查出错: {e}")
    
    # 测试2: 尝试上传一张图片进行预测
    print("\n2. 测试图像预测功能...")
    try:
        # 查找一个测试图像
        test_image_path = None
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'healthy' in file.lower():
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path:
                break
        
        if test_image_path:
            print(f"   使用测试图像: {test_image_path}")
            with open(test_image_path, 'rb') as img_file:
                files = {'file': img_file}
                response = requests.post('http://localhost:5000/predict', files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'error' not in result:
                        print(f"   ✓ 预测成功!")
                        print(f"   - 预测类别: {result['predicted_class']}")
                        print(f"   - 置信度: {result['confidence']}")
                        print(f"   - 所有概率: {result['all_probabilities']}")
                    else:
                        print(f"   ⚠ 预测返回错误: {result['error']}")
                else:
                    print(f"   ✗ 预测请求失败，状态码: {response.status_code}")
        else:
            print("   ⚠ 未找到测试图像")
    except Exception as e:
        print(f"   ✗ 图像预测测试出错: {e}")
    
    # 测试3: 测试用户注册
    print("\n3. 测试用户注册功能...")
    try:
        # 注册一个测试用户
        user_data = {
            'username': 'test_patient',
            'password': 'test_password',
            'user_type': 'patient',
            'email': 'test@example.com',
            'phone': '1234567890'
        }
        response = requests.post('http://localhost:5000/register_user', json=user_data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✓ 用户注册成功! 用户ID: {result['user_id']}")
            else:
                print(f"   ⚠ 用户注册返回错误: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ✗ 用户注册请求失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 用户注册测试出错: {e}")
    
    # 测试4: 测试用户登录
    print("\n4. 测试用户登录功能...")
    try:
        # 登录刚才注册的用户
        login_data = {
            'username': 'test_patient',
            'password': 'test_password'
        }
        response = requests.post('http://localhost:5000/login', json=login_data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✓ 用户登录成功! 用户类型: {result['user_type']}")
            else:
                print(f"   ⚠ 用户登录返回错误: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ✗ 用户登录请求失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 用户登录测试出错: {e}")
    
    # 测试5: 测试消息发送
    print("\n5. 测试消息发送功能...")
    try:
        # 发送一条测试消息
        message_data = {
            'sender': 'test_patient',
            'receiver': 'doctor1',
            'message': '这是一条测试消息'
        }
        response = requests.post('http://localhost:5000/send_message', json=message_data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✓ 消息发送成功!")
            else:
                print(f"   ⚠ 消息发送返回错误: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ✗ 消息发送请求失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 消息发送测试出错: {e}")
    
    print("\n系统测试完成!")

if __name__ == "__main__":
    test_system()
