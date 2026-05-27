import urllib.request
import urllib.parse
import json
import os
import ssl

# 修复SSL环境变量问题
if 'SSLKEYLOGFILE' in os.environ:
    del os.environ['SSLKEYLOGFILE']

# 创建无证书验证的上下文
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def test_system():
    """
    使用urllib测试脑瘤检测和医患交流系统
    """
    print("开始测试脑瘤检测和医患交流系统...")
    
    # 测试1: 检查服务器健康状态
    print("\n1. 测试服务器健康状态...")
    try:
        req = urllib.request.Request('http://localhost:5000/health')
        response = urllib.request.urlopen(req, context=ssl_context)
        data = json.loads(response.read().decode())
        print(f"   [SUCCESS] 服务器状态: {data['status']}")
        print(f"   [SUCCESS] 模型加载状态: {data['model_loaded']}")
        print(f"   [SUCCESS] 用户数量: {data['user_count']}")
        print(f"   [SUCCESS] 消息数量: {data['message_count']}")
    except Exception as e:
        print(f"   [ERROR] 健康检查出错: {e}")
    
    # 测试2: 测试用户注册
    print("\n2. 测试用户注册功能...")
    try:
        user_data = {
            'username': 'test_patient',
            'password': 'test_password',
            'user_type': 'patient',
            'email': 'test@example.com',
            'phone': '1234567890'
        }
        req = urllib.request.Request(
            'http://localhost:5000/register_user',
            data=json.dumps(user_data).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, context=ssl_context)
        result = json.loads(response.read().decode())
        if result['success']:
            print(f"   [SUCCESS] 用户注册成功! 用户ID: {result['user_id']}")
        else:
            print(f"   [WARNING] 用户注册返回错误: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   [ERROR] 用户注册测试出错: {e}")
    
    # 测试3: 测试用户登录
    print("\n3. 测试用户登录功能...")
    try:
        login_data = {
            'username': 'test_patient',
            'password': 'test_password'
        }
        req = urllib.request.Request(
            'http://localhost:5000/login',
            data=json.dumps(login_data).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, context=ssl_context)
        result = json.loads(response.read().decode())
        if result['success']:
            print(f"   [SUCCESS] 用户登录成功! 用户类型: {result['user_type']}")
        else:
            print(f"   [WARNING] 用户登录返回错误: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   [ERROR] 用户登录测试出错: {e}")
    
    # 测试4: 测试消息发送
    print("\n4. 测试消息发送功能...")
    try:
        message_data = {
            'sender': 'test_patient',
            'receiver': 'doctor1',
            'message': '这是一条测试消息'
        }
        req = urllib.request.Request(
            'http://localhost:5000/send_message',
            data=json.dumps(message_data).encode(),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, context=ssl_context)
        result = json.loads(response.read().decode())
        if result['success']:
            print(f"   [SUCCESS] 消息发送成功!")
        else:
            print(f"   [WARNING] 消息发送返回错误: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   [ERROR] 消息发送测试出错: {e}")
    
    print("\n系统测试完成!")

if __name__ == "__main__":
    test_system()
