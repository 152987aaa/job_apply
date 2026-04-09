"""
=============================================================================
LSTM股票价格预测 - 完整示例代码
=============================================================================
本示例演示如何使用PyTorch实现LSTM神经网络进行时间序列预测。

核心知识点：
1. 数据预处理：归一化、滑动窗口构建序列数据
2. LSTM模型构建：理解input_size, hidden_size, num_layers等参数
3. 训练流程：损失函数、优化器、反向传播
4. 模型评估：MSE、MAE、R²等指标
=============================================================================
"""

# =============================================================================
# 第一部分：导入必要的库
# =============================================================================
import numpy as np  # 数值计算库，用于数据处理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块，包含LSTM等层
import matplotlib.pyplot as plt  # 绑图库，用于结果可视化
from sklearn.preprocessing import MinMaxScaler  # 数据归一化工具
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 评估指标

# -----------------------------------------------------------------------------
# 设置matplotlib中文显示
# Windows系统使用SimHei或Microsoft YaHei，其他系统会fallback到DejaVu Sans
# -----------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# =============================================================================
# 第二部分：数据准备
# =============================================================================

def generate_stock_data(n_samples=1000):
    """
    生成模拟股票价格数据

    股票价格通常包含三个组成部分：
    1. 长期趋势（Trend）：线性增长或下降
    2. 季节性波动（Seasonality）：周期性变化
    3. 随机噪声（Noise）：不可预测的波动

    参数：
        n_samples: 数据点数量，默认1000个时间步

    返回：
        price: 生成的价格序列（numpy数组）
    """
    # 创建时间轴，从0到20均匀分布
    t = np.linspace(0, 20, n_samples)

    # 组合三个成分生成价格
    # 100：初始价格基数
    # 10*t：线性趋势，每个时间步增加10
    # 5*sin(...)：季节性波动，振幅为5
    # 2*randn(...)：随机噪声，标准差为2
    price = 100 + 10 * t + 5 * np.sin(2 * np.pi * t / 0.5) + 2 * np.random.randn(n_samples)

    return price


# 生成1000个时间步的模拟股票数据
prices = generate_stock_data(1000)
print(f"生成数据形状: {prices.shape}")  # 输出: (1000,)

# -----------------------------------------------------------------------------
# 数据归一化（非常关键的一步！）
# -----------------------------------------------------------------------------
# LSTM对输入数据的范围非常敏感，如果不归一化：
# - 梯度可能过大或过小，导致训练不稳定
# - sigmoid/tanh激活函数可能饱和
#
# MinMaxScaler将数据缩放到[0, 1]区间，公式为：
# X_scaled = (X - X_min) / (X_max - X_min)
# -----------------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
# reshape(-1, 1)将一维数组转为二维列向量，因为sklearn需要二维输入
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
print(f"归一化后数据范围: [{prices_scaled.min():.4f}, {prices_scaled.max():.4f}]")


# -----------------------------------------------------------------------------
# 滑动窗口构建监督学习数据集
# -----------------------------------------------------------------------------
# 时间序列预测的核心思想：用历史数据预测未来
#
# 例如，SEQ_LENGTH=25表示：
# - 输入X：过去25天的价格序列
# - 输出y：第26天的价格（我们要预测的目标）
#
# 示意图：
# 时间轴:  [t1, t2, t3, ..., t25, t26, t27, ..., t50, ...]
# 样本1:   [X1, X2, X3, ..., X25] -> 预测 X26
# 样本2:       [X2, X3, X4, ..., X26] -> 预测 X27
# ...
# -----------------------------------------------------------------------------
def create_dataset(data, seq_length):
    """
    使用滑动窗口方法创建训练数据集

    参数：
        data: 归一化后的价格数据（形状为[n, 1]）
        seq_length: 序列长度（用多少个历史点预测下一个点）

    返回：
        X: 输入序列，形状为[n_samples, seq_length, 1]
        y: 目标值，形状为[n_samples, 1]
    """
    X, y = [], []

    # 遍历数据，创建滑动窗口样本
    for i in range(len(data) - seq_length):
        # 输入：从位置i开始的seq_length个连续数据点
        X.append(data[i:i + seq_length])
        # 输出：紧接在输入序列后的下一个数据点
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


# 设置序列长度为25，即用过去25天的数据预测下一天
# 这个值需要根据具体任务调整：
# - 太短：可能无法捕捉长期依赖
# - 太长：增加计算量，且可能引入无关噪声
SEQ_LENGTH = 25
X, y = create_dataset(prices_scaled, SEQ_LENGTH)

print(f"总样本数: {len(X)}")  # 1000 - 25 = 975个样本
print(f"输入X形状: {X.shape}")  # (975, 25, 1)
print(f"输出y形状: {y.shape}")  # (975, 1)

# -----------------------------------------------------------------------------
# 划分训练集和测试集
# -----------------------------------------------------------------------------
# 时间序列数据不能随机打乱！必须按时间顺序划分
# 前80%作为训练集，后20%作为测试集
# -----------------------------------------------------------------------------
train_size = int(len(X) * 0.8)  # 80%的数据用于训练

# 转换为PyTorch张量（Tensor）
# FloatTensor是32位浮点数，适合深度学习计算
X_train = torch.FloatTensor(X[:train_size])
y_train = torch.FloatTensor(y[:train_size])
X_test = torch.FloatTensor(X[train_size:])
y_test = torch.FloatTensor(y[train_size:])

print(f"\n数据集划分:")
print(f"  训练集: {X_train.shape} ({train_size} 个样本)")
print(f"  测试集: {X_test.shape} ({len(X) - train_size} 个样本)")


# =============================================================================
# 第三部分：LSTM模型定义
# =============================================================================

class LSTMModel(nn.Module):
    """
    LSTM股票预测模型

    模型架构：
    1. LSTM层：提取序列特征，捕捉时间依赖关系
    2. 全连接层：将LSTM输出映射到预测值

    LSTM内部结构（每个时间步）：
    - 遗忘门(f_t)：决定丢弃哪些旧信息
    - 输入门(i_t)：决定存储哪些新信息
    - 输出门(o_t)：决定输出哪些信息

    关键公式：
    f_t = σ(W_f·[h_{t-1}, x_t] + b_f)    遗忘门
    i_t = σ(W_i·[h_{t-1}, x_t] + b_i)    输入门
    C̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c) 候选细胞状态
    C_t = f_t⊙C_{t-1} + i_t⊙C̃_t          更新细胞状态
    o_t = σ(W_o·[h_{t-1}, x_t] + b_o)    输出门
    h_t = o_t⊙tanh(C_t)                   隐藏状态输出
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        """
        初始化模型参数

        参数说明：
            input_size: 输入特征维度
                - 股票价格是单变量，所以=1
                - 如果使用多个特征（如开盘价、收盘价、成交量），则>1

            hidden_size: LSTM隐藏层神经元数量
                - 决定模型的"记忆容量"
                - 越大能学习越复杂的模式，但也越容易过拟合
                - 常用值：32, 64, 128, 256

            num_layers: LSTM堆叠层数
                - 1层：捕捉简单时序模式
                - 2层：捕捉更复杂的模式（推荐）
                - 3+层：可能需要残差连接，训练难度增加

            output_size: 输出维度
                - 预测单一价格值，所以=1
        """
        super(LSTMModel, self).__init__()

        # 定义LSTM层
        # batch_first=True 表示输入数据形状为(batch, seq, feature)
        # batch_first=False 表示输入数据形状为(seq, batch, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,  # 输入特征维度：1（仅价格）
            hidden_size=hidden_size,  # 隐藏状态维度：64
            num_layers=num_layers,  # LSTM层数：2
            batch_first=True,  # 输入格式：(batch, seq, feature)
            dropout=0.2  # 层间Dropout，防止过拟合
        )

        # 定义全连接层（输出层）
        # 将LSTM的隐藏状态映射到预测值
        # 输入维度 = hidden_size，输出维度 = 1（预测价格）
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播：定义数据如何流过网络

        参数：
            x: 输入张量，形状为(batch_size, seq_length, input_size)

        返回：
            output: 预测值，形状为(batch_size, output_size)

        LSTM输出说明：
            lstm_out: 所有时间步的隐藏状态，形状(batch, seq, hidden)
            (h_n, c_n): 最后时间步的隐藏状态和细胞状态
        """
        # LSTM层处理
        # lstm_out: 所有时间步的输出，形状(batch, seq_length, hidden_size)
        # h_n: 最后时间步的隐藏状态，形状(num_layers, batch, hidden_size)
        # c_n: 最后时间步的细胞状态，形状(num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        # lstm_out[:, -1, :] 表示：
        #   :      - 取所有batch
        #   -1     - 取最后一个时间步
        #   :      - 取所有hidden特征
        # 结果形状：(batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # 通过全连接层映射到预测值
        # 输出形状：(batch_size, output_size)
        output = self.fc(last_output)

        return output


# 实例化模型
model = LSTMModel(
    input_size=1,  # 单变量输入（价格）
    hidden_size=64,  # 64个隐藏单元
    num_layers=2,  # 2层LSTM
    output_size=1  # 单值输出（预测价格）
)

# 打印模型结构
print(f"\n模型结构:")
print(model)
print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 第四部分：模型训练
# =============================================================================

# -----------------------------------------------------------------------------
# 定义损失函数
# -----------------------------------------------------------------------------
# MSELoss（均方误差损失）适用于回归问题
# 公式：MSE = (1/n) * Σ(y_pred - y_true)²
# 特点：对大误差惩罚更重，训练时梯度较大
# -----------------------------------------------------------------------------
criterion = nn.MSELoss()

# -----------------------------------------------------------------------------
# 定义优化器
# -----------------------------------------------------------------------------
# Adam是目前最流行的优化器，结合了：
# - Momentum：加速收敛，减少震荡
# - RMSprop：自适应学习率
#
# 学习率lr=0.001是一个较好的起点
# 如果训练不稳定，可以减小到0.0001
# 如果收敛太慢，可以增大到0.01
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------------------------------------------------
# 创建数据加载器
# -----------------------------------------------------------------------------
# DataLoader的作用：
# 1. 将数据分成小批次(batch)，减少内存占用
# 2. shuffle=True在训练时打乱数据顺序，提高泛化能力
#    注意：时间序列预测时，通常不shuffle以保持时序关系
#    这里为了演示基本流程，使用shuffle
# -----------------------------------------------------------------------------
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # 对于纯序列预测，可以设为False
)

# -----------------------------------------------------------------------------
# 训练循环
# -----------------------------------------------------------------------------
# 训练深度学习的标准流程：
# 1. 前向传播：计算预测值
# 2. 计算损失：比较预测值和真实值
# 3. 反向传播：计算梯度
# 4. 更新参数：优化器根据梯度更新权重
# -----------------------------------------------------------------------------
num_epochs = 100  # 训练轮数
train_losses = []  # 记录每轮的损失，用于绘制损失曲线

print(f"\n开始训练...")
print(f"训练配置: epochs={num_epochs}, batch_size={batch_size}")

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式（启用Dropout等）

    epoch_loss = 0  # 累计本轮损失

    for batch_X, batch_y in train_loader:
        # 步骤1：清零梯度
        # PyTorch会累加梯度，每次迭代前需要清零
        optimizer.zero_grad()

        # 步骤2：前向传播，得到预测值
        predictions = model(batch_X)

        # 步骤3：计算损失
        loss = criterion(predictions, batch_y)

        # 步骤4：反向传播，计算梯度
        loss.backward()

        # 步骤5：更新参数
        optimizer.step()

        # 累计损失
        epoch_loss += loss.item()

    # 计算平均损失并记录
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 每20轮打印一次进度
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

print("训练完成！")

# =============================================================================
# 第五部分：模型预测
# =============================================================================

# 切换到评估模式
# eval()会关闭Dropout等训练专用的层，确保预测结果稳定
model.eval()

# torch.no_grad()：禁用梯度计算
# 预测时不需要梯度，可以节省内存和计算时间
with torch.no_grad():
    # 对训练集和测试集进行预测
    train_pred = model(X_train)
    test_pred = model(X_test)

# -----------------------------------------------------------------------------
# 反归一化：将预测结果还原到原始价格范围
# -----------------------------------------------------------------------------
# 之前用MinMaxScaler将数据缩放到[0,1]
# 现在需要逆变换回原始价格范围
# 公式：X_original = X_scaled * (X_max - X_min) + X_min
# -----------------------------------------------------------------------------
train_pred = scaler.inverse_transform(train_pred.numpy())
y_train_actual = scaler.inverse_transform(y_train.numpy())
test_pred = scaler.inverse_transform(test_pred.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy())

print(f"\n预测完成！")
print(f"预测值范围: [{test_pred.min():.2f}, {test_pred.max():.2f}]")

# =============================================================================
# 第六部分：模型评估与可视化
# =============================================================================

# -----------------------------------------------------------------------------
# 计算评估指标
# -----------------------------------------------------------------------------
# MSE（均方误差）：越小越好，对大误差敏感
# RMSE（均方根误差）：MSE的平方根，与原数据同量纲
# MAE（平均绝对误差）：对异常值更鲁棒
# R²（决定系数）：0-1之间，越接近1说明拟合越好
# -----------------------------------------------------------------------------
mse = mean_squared_error(y_test_actual, test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, test_pred)
r2 = r2_score(y_test_actual, test_pred)

# -----------------------------------------------------------------------------
# 多维度可视化
# -----------------------------------------------------------------------------
# 创建一个大画布，包含6个子图，从不同角度展示模型效果
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 12))

# 子图1：原始股票数据概览
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(prices, 'b-', linewidth=1.5, alpha=0.8)
ax1.set_title('原始股票价格数据', fontsize=12, fontweight='bold')
ax1.set_xlabel('时间步')
ax1.set_ylabel('价格')
ax1.grid(True, alpha=0.3)
# 标记训练集和测试集的分割点
ax1.axvline(x=len(prices) * 0.8, color='red', linestyle='--',
            linewidth=2, label='训练/测试分割点')
ax1.legend()

# 子图2：训练损失曲线
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(train_losses, 'purple', linewidth=2)
ax2.set_title('训练损失曲线', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch（训练轮次）')
ax2.set_ylabel('MSE Loss（均方误差）')
ax2.grid(True, alpha=0.3)
# 填充曲线下方区域，更直观展示损失下降趋势
ax2.fill_between(range(len(train_losses)), train_losses,
                 alpha=0.3, color='purple')

# 子图3：训练集预测对比
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(y_train_actual, 'g-', label='真实值', alpha=0.7, linewidth=1.5)
ax3.plot(train_pred, 'r--', label='预测值', alpha=0.7, linewidth=1.5)
ax3.set_title('训练集预测对比', fontsize=12, fontweight='bold')
ax3.set_xlabel('样本索引')
ax3.set_ylabel('价格')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 子图4：测试集预测对比（最重要的评估）
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(y_test_actual, 'g-', label='真实值', linewidth=1.5)
ax4.plot(test_pred, 'r--', label='预测值', linewidth=1.5)
ax4.set_title('测试集预测对比（核心评估）', fontsize=12, fontweight='bold')
ax4.set_xlabel('样本索引')
ax4.set_ylabel('价格')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

# 子图5：预测误差分布
ax5 = fig.add_subplot(2, 3, 5)
# 计算预测误差
test_errors = y_test_actual.flatten() - test_pred.flatten()
# 绘制误差直方图
ax5.hist(test_errors, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
ax5.axvline(x=np.mean(test_errors), color='orange', linestyle='-', linewidth=2,
            label=f'平均误差: {np.mean(test_errors):.2f}')
ax5.set_title('测试集预测误差分布', fontsize=12, fontweight='bold')
ax5.set_xlabel('预测误差（真实值 - 预测值）')
ax5.set_ylabel('频次')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

# 子图6：真实值 vs 预测值散点图
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(y_test_actual.flatten(), test_pred.flatten(),
            alpha=0.5, c='blue', s=10)
# 理想预测线：如果预测完美，所有点应该在这条线上
min_val = min(y_test_actual.min(), test_pred.min())
max_val = max(y_test_actual.max(), test_pred.max())
ax6.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2, label='理想预测线 (y=x)')
ax6.set_title('真实值 vs 预测值散点图', fontsize=12, fontweight='bold')
ax6.set_xlabel('真实值')
ax6.set_ylabel('预测值')
ax6.legend(loc='upper left')
ax6.grid(True, alpha=0.3)

# 添加总标题，包含主要评估指标
fig.suptitle(
    f'LSTM股票价格预测结果 | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}',
    fontsize=14, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('lstm_stock_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# 打印详细的评估报告
# -----------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("                    模型评估报告")
print(f"{'=' * 60}")
print(f"\n【误差指标】越小越好")
print(f"  均方误差 (MSE):      {mse:.4f}")
print(f"  均方根误差 (RMSE):   {rmse:.4f}")
print(f"  平均绝对误差 (MAE):  {mae:.4f}")
print(f"\n【拟合优度】")
print(f"  决定系数 (R²):       {r2:.4f}  ", end="")
if r2 > 0.9:
    print("(优秀)")
elif r2 > 0.7:
    print("(良好)")
elif r2 > 0.5:
    print("(一般)")
else:
    print("(需改进)")
print(f"\n{'=' * 60}")
print(f"预测结果已保存到: lstm_stock_prediction.png")
print(f"{'=' * 60}")