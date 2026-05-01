class PID:
    # pid的初始化赋值
    def __init__(self, Kp, Ki, Kd, setpoint=0, sample_time=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self.prev_error = 0
        self.integral = 0

    # pid的cal_process
    def update(self, measured_value):
        error = self.setpoint - measured_value  # 计算误差
        self.integral += error * self.sample_time  # 积分
        derivative = (error - self.prev_error) / self.sample_time  # 微分

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative  # 计算控制输入
        self.prev_error = error  # 保存误差

        return output


pid = PID(Kp=1.0, Ki=0.1, Kd=0.01, setpoint=100)
measured_value = 90  # 假设的当前测量值
control_input = pid.update(measured_value)

print(f"Control Input: {control_input}")
