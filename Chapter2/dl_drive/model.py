# import paddle.nn as nn
# import paddle
#
#
# class AutoDriveNet(nn.Layer):
#     """端到端自动驾驶模型"""
#     def __init__(self):
#         """初始化"""
#         super(AutoDriveNet, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2D(3, 24, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2D(24, 36, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2D(36, 48, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2D(48, 64, 3),
#             nn.ELU(),
#             nn.Conv2D(64, 64, 3),
#             nn.Dropout(0.5),
#         )
#         self.linear_layers = nn.Sequential(
#             nn.Linear(in_features=64 * 8 * 13, out_features=100),
#             nn.ELU(),
#             nn.Linear(in_features=100, out_features=50),
#             nn.ELU(),
#             nn.Linear(in_features=50, out_features=10),
#             nn.Linear(in_features=10, out_features=1),
#         )
#
#     def forward(self, input):
#         """前向推理"""
#         input = paddle.reshape(input, [input.shape[0], 3, 120, 160])
#         output = self.conv_layers(input) # 卷积模块
#         output = paddle.reshape(output, [output.shape[0], -1]) # 展平
#         output = self.linear_layers(output) # 线性变换模块
#         return output
import paddle
import paddle.nn as nn
class AutoDriveNet(nn.Layer):
    """端到端自动驾驶模型"""
    def __init__(self):
        """初始化"""
        super(AutoDriveNet,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2D(3,24,5,stride=2),
            nn.ELU(),
            nn.Conv2D(24,36,5,stride=2),
            nn.ELU(),
            nn.Conv2D(36,48,5,stride=2),
            nn.ELU(),
            nn.Conv2D(48,64,3),
            nn.ELU(),
            nn.Conv2D(64,64,3),
            nn.Dropout(0.5),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*8*13,out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100,out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50,out_features=10),
            nn.Linear(in_features=10,out_features=1),
        )
    def forward(self,input):
        """前向推理"""
        input = paddle.reshape(input,[input.shape[0],3,120,160])
        output = self.conv_layers(input)
        output = paddle.reshape(output,[output.shape[0],-1])
        output = self.linear_layers(output)
        return output
