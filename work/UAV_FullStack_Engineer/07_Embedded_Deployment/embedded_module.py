#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙大高飞无人机团队 | 嵌入式部署模块
教学目标：掌握Jetson Orin、Pixhawk部署
对应文档：《Jetson Orin Developer Guide》
"""

import numpy as np
import onnxruntime as ort
import time

class JetsonOrinInterface:
    """Jetson Orin接口"""
    
    def __init__(self):
        """初始化Jetson Orin接口"""
        self.device = 'cuda' if ort.get_device() == 'GPU' else 'cpu'
        self.session = None
        self.input_name = None
        self.output_name = None
        
        print(f"Jetson Orin初始化完成，设备: {self.device}")
    
    def load_model(self, model_path):
        """加载ONNX模型"""
        print(f"加载模型: {model_path}")
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"模型加载成功，输入: {self.input_name}, 输出: {self.output_name}")
    
    def infer(self, input_data):
        """执行推理"""
        if self.session is None:
            print("错误：未加载模型")
            return None
        
        # 准备输入数据
        input_data = np.array(input_data, dtype=np.float32)
        
        # 执行推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        inference_time = (time.time() - start_time) * 1000
        
        print(f"推理耗时: {inference_time:.2f} ms")
        
        return outputs[0]
    
    def get_performance_stats(self):
        """获取性能统计"""
        import psutil
        
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        print(f"CPU使用率: {cpu_usage}%")
        print(f"内存使用率: {memory_usage}%")
        
        return {'cpu': cpu_usage, 'memory': memory_usage}

class PixhawkInterface:
    """Pixhawk飞控接口"""
    
    def __init__(self, port='/dev/ttyUSB0', baud_rate=57600):
        """
        初始化Pixhawk接口
        :param port: 串口端口
        :param baud_rate: 波特率
        """
        self.port = port
        self.baud_rate = baud_rate
        self.connected = False
        
        # 飞控状态
        self.armed = False
        self.mode = 'STABILIZE'
        self.battery_voltage = 0.0
        self.gps_fix = False
        
        print(f"Pixhawk接口初始化，端口: {port}")
    
    def connect(self):
        """连接Pixhawk"""
        # 模拟连接
        print(f"正在连接Pixhawk...")
        self.connected = True
        print("Pixhawk连接成功")
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        print("Pixhawk已断开")
    
    def send_rc_channels(self, channels):
        """发送RC通道数据"""
        if not self.connected:
            print("错误：未连接Pixhawk")
            return
        
        # channels: [roll, pitch, throttle, yaw]
        print(f"发送RC通道: {channels}")
    
    def send_mavlink_message(self, message_type, data):
        """发送MAVLink消息"""
        if not self.connected:
            print("错误：未连接Pixhawk")
            return
        
        print(f"发送MAVLink消息: {message_type}, 数据: {data}")
    
    def read_sensors(self):
        """读取传感器数据"""
        if not self.connected:
            return None
        
        # 模拟传感器数据
        sensors = {
            'imu': {'acc': [0.1, 0.2, 9.8], 'gyro': [0.01, 0.02, 0.03]},
            'gps': {'lat': 30.5728, 'lon': 104.0668, 'alt': 500.0},
            'battery': {'voltage': 14.2, 'current': 5.5},
            'baro': {'altitude': 498.5}
        }
        
        return sensors

class TensorRTOptimizer:
    """TensorRT模型优化器"""
    
    def __init__(self):
        """初始化TensorRT优化器"""
        try:
            import tensorrt as trt
            self.trt = trt
            self.logger = trt.Logger(trt.Logger.WARNING)
            print("TensorRT优化器初始化完成")
        except ImportError:
            self.trt = None
            print("警告：未安装TensorRT")
    
    def optimize_model(self, onnx_path, engine_path):
        """
        优化ONNX模型为TensorRT引擎
        :param onnx_path: ONNX模型路径
        :param engine_path: 输出引擎路径
        """
        if self.trt is None:
            print("错误：未安装TensorRT")
            return
        
        print(f"优化ONNX模型: {onnx_path}")
        
        # 创建构建器
        builder = self.trt.Builder(self.logger)
        config = builder.create_builder_config()
        
        # 设置最大工作空间
        config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # 解析ONNX模型
        parser = self.trt.OnnxParser(builder.create_network(), self.logger)
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        
        # 构建引擎
        network = builder.create_network(1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser.parse_from_file(onnx_path)
        
        engine = builder.build_engine(network, config)
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT引擎已保存至: {engine_path}")
    
    def load_engine(self, engine_path):
        """加载TensorRT引擎"""
        if self.trt is None:
            print("错误：未安装TensorRT")
            return None
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = self.trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        return engine

class EdgeDeployment:
    """边缘部署管理器"""
    
    def __init__(self):
        """初始化边缘部署管理器"""
        self.jetson = JetsonOrinInterface()
        self.pixhawk = PixhawkInterface()
        self.trt_optimizer = TensorRTOptimizer()
        
        # 实时性统计
        self.latency_history = []
        self.max_latency = 0
        self.min_latency = float('inf')
    
    def deploy_model(self, onnx_path):
        """部署模型到边缘设备"""
        print("开始模型部署...")
        
        # 优化模型
        engine_path = onnx_path.replace('.onnx', '.engine')
        self.trt_optimizer.optimize_model(onnx_path, engine_path)
        
        # 加载模型
        self.jetson.load_model(engine_path)
        
        print("模型部署完成")
    
    def run_inference(self, input_data):
        """运行推理"""
        start_time = time.time()
        
        # 执行推理
        output = self.jetson.infer(input_data)
        
        # 计算延迟
        latency = (time.time() - start_time) * 1000
        self.latency_history.append(latency)
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
        
        return output
    
    def get_latency_stats(self):
        """获取延迟统计"""
        if not self.latency_history:
            return None
        
        avg_latency = np.mean(self.latency_history)
        std_latency = np.std(self.latency_history)
        
        print(f"平均延迟: {avg_latency:.2f} ms")
        print(f"最大延迟: {self.max_latency:.2f} ms")
        print(f"最小延迟: {self.min_latency:.2f} ms")
        print(f"延迟标准差: {std_latency:.2f} ms")
        
        return {
            'avg': avg_latency,
            'max': self.max_latency,
            'min': self.min_latency,
            'std': std_latency
        }

def test_deployment():
    """测试边缘部署"""
    deployment = EdgeDeployment()
    
    # 连接Pixhawk
    deployment.pixhawk.connect()
    
    # 读取传感器数据
    sensors = deployment.pixhawk.read_sensors()
    print(f"传感器数据: {sensors}")
    
    # 测试推理
    print("\n测试推理...")
    for i in range(10):
        input_data = np.random.randn(1, 32).astype(np.float32)
        output = deployment.jetson.infer(input_data)
        print(f"推理结果形状: {output.shape if output is not None else 'None'}")
    
    # 获取性能统计
    deployment.jetson.get_performance_stats()
    
    # 断开连接
    deployment.pixhawk.disconnect()
    
    print("\n边缘部署测试完成！")

if __name__ == '__main__':
    test_deployment()