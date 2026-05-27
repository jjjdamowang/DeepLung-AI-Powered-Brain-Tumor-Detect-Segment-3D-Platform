import os
# 设置环境变量以避免SSL密钥日志文件权限问题
os.environ['SSLKEYLOGFILE'] = ''

import requests

# 测试后端预测功能
def test_predict():
    url = 'http://localhost:5000/predict'
    
    # 选择一个测试图像
    test_image_path = "Brain Tumor CT scan Images（CT图）/Tumor/ct_tumor (1).jpg"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        # 尝试另一个路径
        test_image_path = "Brain Tumor CT scan Images（CT图）/Healthy/ct_healthy (1).jpg"
        if not os.path.exists(test_image_path):
            print(f"测试图像也不存在: {test_image_path}")
            return
    
    print(f"使用测试图像: {test_image_path}")
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        # 尝试解析JSON响应
        try:
            json_response = response.json()
            print(f"JSON响应: {json_response}")
        except:
            print("响应不是有效的JSON格式")
            
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_predict()
