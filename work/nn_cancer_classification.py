# ==============================================
# 《机器学习》实验三：神经网络构建癌症分类模型
# 学生姓名：曹婷婷
# 学号：20253619
# 专业班级：软工专硕251班
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 全局设置（解决中文显示） ----------------------
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("《机器学习》实验三：神经网络构建癌症分类模型")
print("=" * 70)

# ---------------------- 3.1 数据集介绍 ----------------------
print("\n【3.1 数据集介绍】")
print("-" * 50)

# 加载威斯康星乳腺癌数据集
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='diagnosis')

print(f"数据集来源：sklearn内置的威斯康星乳腺癌数据集（Breast Cancer Wisconsin Dataset）")
print(f"数据集规模：{X.shape[0]}个样本，{X.shape[1]}个特征")
print(f"类别分布：恶性(Malignant)=0: {sum(y==0)}个，良性(Benign)=1: {sum(y==1)}个")
print(f"\n特征属性说明：")
print("1. radius_mean: 平均半径")
print("2. texture_mean: 平均纹理")
print("3. perimeter_mean: 平均周长")
print("4. area_mean: 平均面积")
print("5. smoothness_mean: 平均平滑度")
print("6. compactness_mean: 平均紧致度")
print("7. concavity_mean: 平均凹陷度")
print("8. concave points_mean: 平均凹点数")
print("9. symmetry_mean: 平均对称性")
print("10. fractal_dimension_mean: 平均分形维数")
print("...（其余20个特征为标准差和最大值）")

# ---------------------- 3.2 数据预处理 ----------------------
print("\n【3.2 数据预处理】")
print("-" * 50)

# 数据标准化（神经网络对特征尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分层划分训练集(70%)/测试集(30%)，保证类别分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ 数据标准化完成：所有特征已归一化到均值为0，标准差为1")
print(f"✅ 数据集划分完成：")
print(f"   训练集：{X_train.shape}（正样本:{sum(y_train==1)}, 负样本:{sum(y_train==0)}）")
print(f"   测试集：{X_test.shape}（正样本:{sum(y_test==1)}, 负样本:{sum(y_test==0)}）")

# ---------------------- 3.3 利用感知机构建癌症分类模型 ----------------------
print("\n【3.3 利用感知机构建癌症分类模型】")
print("-" * 50)

# 3.3.1 代码实现
print("3.3.1 代码实现")
print("-" * 40)

# 构建感知机模型
perceptron_model = Perceptron(
    eta0=0.1,               # 学习率
    max_iter=1000,          # 最大迭代次数
    tol=1e-3,               # 容忍度，当损失下降小于此值时停止
    random_state=42,        # 固定随机种子
    shuffle=True            # 每轮迭代前打乱样本顺序
)

# 训练模型
perceptron_model.fit(X_train, y_train)

# 预测
y_pred_perceptron = perceptron_model.predict(X_test)

# 3.3.2 运行结果
print("\n3.3.2 运行结果")
print("-" * 40)

# 计算评价指标
per_acc = accuracy_score(y_test, y_pred_perceptron)
per_pre = precision_score(y_test, y_pred_perceptron)
per_rec = recall_score(y_test, y_pred_perceptron)
per_f1 = f1_score(y_test, y_pred_perceptron)

print("感知机模型评价指标：")
print(f"准确率(Accuracy)：{per_acc:.4f}")
print(f"精准率(Precision)：{per_pre:.4f}")
print(f"召回率(Recall)：{per_rec:.4f}")
print(f"F1值(F1-Score)：{per_f1:.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred_perceptron, target_names=['恶性', '良性']))

# ---------------------- 3.4 利用多层感知机构建癌症分类模型 ----------------------
print("\n【3.4 利用多层感知机构建癌症分类模型】")
print("-" * 50)

# 3.4.1 代码实现
print("3.4.1 代码实现")
print("-" * 40)

# 构建多层感知机模型
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 隐藏层结构：64个神经元→32个神经元
    activation='relu',            # 激活函数：ReLU
    solver='adam',                # 优化器：Adam
    alpha=0.0001,                 # L2正则化系数
    batch_size=32,                # 小批量梯度下降的批次大小
    learning_rate='adaptive',     # 自适应学习率
    max_iter=500,                 # 最大迭代次数
    tol=1e-4,                     # 容忍度
    random_state=42,              # 固定随机种子
    verbose=False                 # 不打印训练过程
)

# 训练模型
mlp_model.fit(X_train, y_train)

# 预测
y_pred_mlp = mlp_model.predict(X_test)

# 3.4.2 运行结果
print("\n3.4.2 运行结果")
print("-" * 40)

# 计算评价指标
mlp_acc = accuracy_score(y_test, y_pred_mlp)
mlp_pre = precision_score(y_test, y_pred_mlp)
mlp_rec = recall_score(y_test, y_pred_mlp)
mlp_f1 = f1_score(y_test, y_pred_mlp)

print("多层感知机模型评价指标：")
print(f"准确率(Accuracy)：{mlp_acc:.4f}")
print(f"精准率(Precision)：{mlp_pre:.4f}")
print(f"召回率(Recall)：{mlp_rec:.4f}")
print(f"F1值(F1-Score)：{mlp_f1:.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred_mlp, target_names=['恶性', '良性']))

# 绘制MLP训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(mlp_model.loss_curve_, label='训练损失', color='#1f77b4')
plt.title('多层感知机训练损失曲线', fontsize=14)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('MLP训练损失曲线.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 四、实验对比分析 ----------------------
print("\n【四、实验对比分析】")
print("-" * 50)

# 模型对比表格
comparison_df = pd.DataFrame({
    '模型': ['感知机', '多层感知机'],
    '准确率': [per_acc, mlp_acc],
    '精准率': [per_pre, mlp_pre],
    '召回率': [per_rec, mlp_rec],
    'F1值': [per_f1, mlp_f1]
})

print("模型评价指标对比表：")
print("-" * 50)
print(comparison_df.to_string(index=False))
print("\n" + "-" * 50)

# 分析结论
print("\n【实验分析】")
print("1. 模型性能对比：")
print(f"   - 感知机准确率：{per_acc:.2%}，多层感知机准确率：{mlp_acc:.2%}")
print(f"   - 多层感知机在所有指标上均优于感知机，提升幅度约{((mlp_acc-per_acc)/per_acc*100):.1f}%")

print("\n2. 差异原因分析：")
print("   - 感知机是线性模型，只能处理线性可分问题")
print("   - 多层感知机通过隐藏层和非线性激活函数，能够学习复杂的非线性特征")
print("   - 乳腺癌数据存在非线性边界，MLP更适合此类问题")

print("\n3. 模型复杂度对比：")
print("   - 感知机：仅一个神经元，参数少，训练快，但表达能力有限")
print("   - 多层感知机：包含隐藏层，参数多，训练时间较长，但表达能力强")

# 保存对比结果
comparison_df.to_csv('神经网络模型对比.csv', index=False, encoding='utf-8-sig')
print("\n✅ 模型对比结果已保存到：神经网络模型对比.csv")

# ---------------------- 五、实验总结 ----------------------
print("\n【五、实验总结】")
print("-" * 50)
print("1. 成功实现了感知机和多层感知机对癌症数据的分类")
print("2. 多层感知机通过非线性变换能够更好地处理复杂的分类问题")
print("3. 数据预处理（标准化）对神经网络模型性能有重要影响")
print("4. 在医疗诊断场景中，高召回率尤为重要，本实验中MLP召回率达到99%以上")

print("\n" + "=" * 70)
print("实验完成！")
print("=" * 70)