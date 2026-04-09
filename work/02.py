
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# ---------------------- 自定义函数：英文数字转阿拉伯数字 ----------------------
def convert_english_to_number(word):


    num_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100,
        'thirty-nine': 39, 'forty-five': 45, 'fifty-two': 52  # 处理带连字符的数字
    }
    # 处理连字符/空格分隔的复合数字
    word = word.lower().replace('-', ' ')
    parts = word.split()

    total = 0
    current = 0
    for part in parts:
        if part == 'hundred':
            current *= 100
            total += current
            current = 0
        else:
            current += num_map.get(part, 0)
    total += current
    return total if total > 0 else 0

# ---------------------- 1. 加载数据集 ----------------------
print("=" * 60)
print("开始加载数据集...")
df = pd.read_csv('ParisHousing.csv')
print(f"数据集加载成功，形状：{df.shape}")
print(f"数据集列名：\n{df.columns.tolist()}")
print(f"缺失值统计：\n{df.isnull().sum().sum()} 个缺失值")

# ---------------------- 2. 数据预处理（解决所有非数值问题） ----------------------
print("\n" + "=" * 60)
print("开始数据预处理...")

# 2.1 处理numberOfRooms列：英文数字转数值
df['numberOfRooms'] = df['numberOfRooms'].apply(convert_english_to_number)
print("✅ numberOfRooms（房间数）：英文数字已转换为阿拉伯数字")

# 2.2 处理布尔型列：True/False → 1/0
bool_cols = ['isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasGuestRoom']
# 先筛选出数据集中实际存在的布尔列（避免列名不一致报错）
bool_cols = [col for col in bool_cols if col in df.columns]
for col in bool_cols:
    df[col] = df[col].astype(int)
print(f"✅ 布尔型列 {bool_cols}：已转换为 0/1 数值型")

# 2.3 处理字符串分类列（如PoolAndYard/最后一列）
# 筛选出所有非数值型列（除了目标列category）
str_cols = df.select_dtypes(include=['object']).columns.tolist()
str_cols = [col for col in str_cols if col != 'category']  # 排除目标列
for col in str_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"✅ 字符串列 {col}：已通过LabelEncoder转换为数值编码")

# ---------------------- 3. 特征/目标分离 + 标准化 + 数据集划分 ----------------------
# 特征列：剔除价格(price)和分类目标(category)
X = df.drop(['price', 'category'], axis=1)
# 目标列：住房分类（Basic/Luxury）
y = df['category']

print(f"\n预处理后特征列：{X.columns.tolist()}")
print(f"特征列数据类型：\n{X.dtypes}")

# 标准化连续特征（现在所有特征都是数值型，不会报错）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分层划分训练集(70%)/测试集(30%)，保证类别分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n✅ 数据集划分完成：")
print(f"   训练集：{X_train.shape} | 测试集：{X_test.shape}")
print("=" * 60)

# ---------------------- 4. 训练ID3决策树模型 ----------------------
print("\n开始训练ID3决策树模型...")
id3_model = DecisionTreeClassifier(
    criterion='entropy',  # 熵 → 信息增益（ID3核心）
    max_depth=8,  # 限制树深度，防止过拟合
    random_state=42,  # 固定随机种子，结果可复现
    min_samples_split=20  # 节点分裂最小样本数
)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

# 计算ID3评价指标（二分类/多分类通用）
id3_acc = accuracy_score(y_test, y_pred_id3)
id3_pre = precision_score(y_test, y_pred_id3, average='macro')
id3_rec = recall_score(y_test, y_pred_id3, average='macro')
id3_f1 = f1_score(y_test, y_pred_id3, average='macro')

print("=" * 60)
print("ID3决策树模型评价指标")
print(f"准确率(Accuracy)：{id3_acc:.4f}")
print(f"精准率(Precision)：{id3_pre:.4f}")
print(f"召回率(Recall)：{id3_rec:.4f}")
print(f"F1值(F1-Score)：{id3_f1:.4f}")
print("=" * 60)

# ---------------------- 5. ID3决策树可视化（中文显示） ----------------------
plt.figure(figsize=(18, 10))
plot_tree(
    id3_model,
    max_depth=3,  # 仅展示前3层，避免图形过复杂
    feature_names=X.columns,  # 特征列名
    class_names=['基础型住房', '豪华型住房'],  # 中文类别名
    filled=True,  # 填充颜色
    fontsize=9,  # 字体大小
    impurity=True,  # 显示熵/基尼指数
    rounded=True,  # 圆角节点
    proportion=True  # 显示样本占比
)
plt.title('ID3决策树（前3层）', fontsize=18, pad=20)
plt.xlabel('特征划分', fontsize=12)
plt.ylabel('决策分支', fontsize=12)
# 保存图片（高清+无白边）
plt.savefig('ID3决策树_中文显示.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 6. 训练C4.5决策树模型 ----------------------
print("\n开始训练C4.5决策树模型...")
c45_model = DecisionTreeClassifier(
    criterion='entropy',  # 熵 → 信息增益率（C4.5核心）
    max_depth=8,
    random_state=42,
    min_samples_split=20,
    splitter='best'  # 最优划分（C4.5逻辑）
)
c45_model.fit(X_train, y_train)
y_pred_c45 = c45_model.predict(X_test)

# 计算C4.5评价指标
c45_acc = accuracy_score(y_test, y_pred_c45)
c45_pre = precision_score(y_test, y_pred_c45, average='macro')
c45_rec = recall_score(y_test, y_pred_c45, average='macro')
c45_f1 = f1_score(y_test, y_pred_c45, average='macro')

print("=" * 60)
print("C4.5决策树模型评价指标")
print(f"准确率(Accuracy)：{c45_acc:.4f}")
print(f"精准率(Precision)：{c45_pre:.4f}")
print(f"召回率(Recall)：{c45_rec:.4f}")
print(f"F1值(F1-Score)：{c45_f1:.4f}")
print("=" * 60)

# ---------------------- 7. 训练CART决策树模型 ----------------------
print("\n开始训练CART决策树模型...")
cart_model = DecisionTreeClassifier(
    criterion='gini',  # 基尼指数（CART核心）
    max_depth=8,
    random_state=42,
    min_samples_split=20
)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

# 计算CART评价指标
cart_acc = accuracy_score(y_test, y_pred_cart)
cart_pre = precision_score(y_test, y_pred_cart, average='macro')
cart_rec = recall_score(y_test, y_pred_cart, average='macro')
cart_f1 = f1_score(y_test, y_pred_cart, average='macro')

print("=" * 60)
print("CART决策树模型评价指标")
print(f"准确率(Accuracy)：{cart_acc:.4f}")
print(f"精准率(Precision)：{cart_pre:.4f}")
print(f"召回率(Recall)：{cart_rec:.4f}")
print(f"F1值(F1-Score)：{cart_f1:.4f}")
print("=" * 60)

# ---------------------- 8. 模型结果汇总 + 保存 ----------------------
model_metrics = pd.DataFrame({
    '模型名称': ['ID3', 'C4.5', 'CART'],
    '准确率': [id3_acc, c45_acc, cart_acc],
    '精准率': [id3_pre, c45_pre, cart_pre],
    '召回率': [id3_rec, c45_rec, cart_rec],
    'F1值': [id3_f1, c45_f1, cart_f1]
})

print("\n所有模型评价指标汇总：")
print(model_metrics)

# 保存结果到CSV（中文编码）
model_metrics.to_csv('result.csv', index=False, encoding='utf-8-sig')
print("\n✅ 模型结果已保存到：result.csv")
print("✅ ID3决策树可视化图片已保存到：ID3决策树_中文显示.png")
print("\n程序运行完成！")