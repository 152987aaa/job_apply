# ==============================================
# CART决策树 - 巴黎住房分类（独立完整版）
# ==============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 1. 全局设置（解决中文显示） ----------------------
mpl.rcParams['font.sans-serif'] = ['SimHei']  # Windows黑体；Mac替换为['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

# ---------------------- 2. 自定义函数：英文数字转数值（无额外依赖） ----------------------
def convert_english_to_number(word):
    """适配数据集的英文数字转阿拉伯数字，覆盖常见场景"""
    num_map = {
        'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
        'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13,
        'fourteen':14, 'fifteen':15, 'twenty':20, 'thirty':30, 'forty':40, 'fifty':50,
        'thirty-nine':39, 'forty-five':45, 'fifty-two':52, 'sixty':60
    }
    word = word.lower().replace('-', ' ')  # 处理连字符（如thirty-nine）
    return num_map.get(word, 0)  # 未知值兜底返回0

# ---------------------- 3. 加载并预处理数据集（适配你的CSV） ----------------------
# 加载数据（替换为你的CSV文件路径）
df = pd.read_csv('ParisHousing.csv')
print(f"✅ 数据集加载完成，形状：{df.shape}")

# 步骤1：处理numberOfRooms列（英文数字→数值）
df['numberOfRooms'] = df['numberOfRooms'].apply(convert_english_to_number)
print("✅ 房间数列：英文数字已转换为阿拉伯数字")

# 步骤2：布尔列转0/1（True→1，False→0）
bool_cols = [col for col in ['isNewBuilt', 'hasStormProtector', 'hasStorageRoom', 'hasGuestRoom'] if col in df.columns]
for col in bool_cols:
    df[col] = df[col].astype(int)
print(f"✅ 布尔列 {bool_cols}：已转换为0/1数值型")

# 步骤3：字符串分类列编码（如PoolAndYard）
str_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'category']
for col in str_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print(f"✅ 字符串列 {str_cols}：已完成数值编码")

# ---------------------- 4. 特征/目标分离 + 标准化 + 数据集划分 ----------------------
# 特征列：剔除价格(price)和分类目标(category)
X = df.drop(['price', 'category'], axis=1)
# 目标列：住房分类（Basic/Luxury）
y = df['category']

# 标准化连续特征（CART对量纲不敏感，但标准化不影响结果，保持和前序模型一致）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分层划分训练集(70%)/测试集(30%)，保证类别分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"✅ 数据集划分完成：训练集{X_train.shape} | 测试集{X_test.shape}")

# ---------------------- 5. 构建CART决策树（核心代码） ----------------------
# CART分类核心参数：criterion='gini'（基尼指数），二叉树结构
cart_model = DecisionTreeClassifier(
    criterion='gini',               # 基尼指数（CART分类核心）
    max_depth=8,                    # 限制树深度，防止过拟合（CART剪枝思想）
    min_samples_split=20,           # 节点分裂最小样本数（少于20不分裂）
    min_samples_leaf=5,             # 叶节点最小样本数（CART后剪枝基础）
    random_state=42,                # 固定随机种子，结果可复现
    splitter='best'                 # 选择最优二分划分点（CART默认）
)

# 训练模型
cart_model.fit(X_train, y_train)
# 测试集预测
y_pred_cart = cart_model.predict(X_test)

# ---------------------- 6. CART模型详细评估 ----------------------
print("\n" + "="*65)
print("CART决策树 - 巴黎住房分类模型评估结果")
print("="*65)

# 核心分类指标
cart_acc = accuracy_score(y_test, y_pred_cart)
cart_pre = precision_score(y_test, y_pred_cart, average='macro')
cart_rec = recall_score(y_test, y_pred_cart, average='macro')
cart_f1 = f1_score(y_test, y_pred_cart, average='macro')

print(f"准确率(Accuracy)：{cart_acc:.4f}")
print(f"精准率(Precision)：{cart_pre:.4f}")
print(f"召回率(Recall)：{cart_rec:.4f}")
print(f"F1值(F1-Score)：{cart_f1:.4f}")

# 混淆矩阵（直观展示分类对错）
cm = confusion_matrix(y_test, y_pred_cart)
cm_df = pd.DataFrame(
    cm, index=['真实_基础型', '真实_豪华型'], columns=['预测_基础型', '预测_豪华型']
)
print("\n混淆矩阵（分类对错明细）：")
print(cm_df)

# 分类报告（按类别细化指标）
print("\n分类报告（按住房类型）：")
print(classification_report(y_test, y_pred_cart, target_names=['基础型住房', '豪华型住房']))

# CART特征重要性（基尼指数下降幅度衡量）
feature_importance = pd.DataFrame({
    '特征名称': X.columns,
    '基尼重要性': cart_model.feature_importances_  # 重要性=特征划分带来的基尼指数下降总和
}).sort_values('基尼重要性', ascending=False)
print("\nCART核心特征重要性（前6）：")
print(feature_importance.head(6))

# ---------------------- 7. CART决策树可视化（中文+二叉树结构） ----------------------
plt.figure(figsize=(20, 12))
plot_tree(
    cart_model,
    max_depth=3,                    # 展示前3层（二叉树结构更清晰）
    feature_names=X.columns,        # 特征列名
    class_names=['基础型住房', '豪华型住房'],  # 中文类别名
    filled=True,                    # 按基尼值填充颜色（颜色越深纯度越高）
    fontsize=10,                    # 字体大小
    impurity=True,                  # 显示基尼指数（CART核心参考）
    rounded=True,                   # 圆角节点
    proportion=True                 # 显示样本占比
)
plt.title('CART决策树（巴黎住房分类）- 二叉树结构（前3层）', fontsize=20, pad=25)
plt.xlabel('特征二分划分（基尼指数最优）', fontsize=14)
plt.ylabel('决策分支', fontsize=14)
# 保存高清可视化图
plt.savefig('CART决策树_巴黎住房分类.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 8. 结果保存（便于后续分析） ----------------------
# 保存核心指标
metrics_df = pd.DataFrame({
    '评估指标': ['准确率', '精准率', '召回率', 'F1值'],
    '数值': [cart_acc, cart_pre, cart_rec, cart_f1]
})
metrics_df.to_csv('CART模型评估指标.csv', index=False, encoding='utf-8-sig')

# 保存特征重要性
feature_importance.to_csv('CART特征重要性.csv', index=False, encoding='utf-8-sig')

print("\n✅ CART模型结果已保存：")
print("   - CART决策树_巴黎住房分类.png（可视化图）")
print("   - CART模型评估指标.csv（核心分类指标）")
print("   - CART特征重要性.csv（特征贡献度）")
print("\n💡 CART模型运行完成！")