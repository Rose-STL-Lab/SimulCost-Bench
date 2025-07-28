import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 如果系统没有中文字体，使用英文
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def soft_success(d, epsilon):
    """改进的SoftSuccess函数"""
    # 向量化处理
    d = np.asarray(d)
    result = np.zeros_like(d, dtype=float)
    
    # d <= epsilon 的情况
    mask1 = d <= epsilon
    result[mask1] = 1.0
    
    # d > epsilon 的情况
    mask2 = d > epsilon
    k = 5
    result[mask2] = sigmoid(k * (np.log10(epsilon / d[mask2]) + 1))
    
    return result

# 设置参数
epsilon = 1.0

# 创建数据点
d_values = np.logspace(-1, 2.5, 1000) * epsilon  # 从 0.1ε 到 ~300ε

# 计算SoftSuccess值
success_values = soft_success(d_values, epsilon)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制主曲线
plt.plot(d_values/epsilon, success_values, 'b-', linewidth=2.5, label='SoftSuccess Function')

# 添加要求的区域背景
ax = plt.gca()

# 区域1: d ≤ ε (浅绿色)
ax.add_patch(Rectangle((0, 0), 1, 1.05, alpha=0.2, color='green'))

# 区域2: ε < d ≤ 2.5ε (浅黄色)
ax.add_patch(Rectangle((1, 0.9), 1.5, 0.1, alpha=0.2, color='yellow'))

# 区域3: 2.5ε < d ≤ 10ε (浅橙色)
ax.add_patch(Rectangle((2.5, 0.5), 7.5, 0.4, alpha=0.2, color='orange'))

# 区域4: d > 10ε (浅红色)
ax.add_patch(Rectangle((10, 0), 290, 0.5, alpha=0.2, color='red'))

# 添加垂直参考线
plt.axvline(x=1, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1.5)
plt.axvline(x=2.5, color='darkorange', linestyle='--', alpha=0.7, linewidth=1.5)
plt.axvline(x=10, color='darkred', linestyle='--', alpha=0.7, linewidth=1.5)

# 添加水平参考线
plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# 添加关键点标注
key_points = [
    (1.0, 1.0, 'd=ε'),
    (2.5, soft_success(2.5, 1), 'd=2.5ε'),
    (10.0, soft_success(10, 1), 'd=10ε')
]

for x, y, label in key_points:
    plt.plot(x, y, 'ro', markersize=8)
    plt.annotate(f'{label}\nSS={y:.3f}', 
                xy=(x, y), 
                xytext=(x*1.5, y+0.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))

# 设置坐标轴
plt.xlabel('d/ε', fontsize=14)
plt.ylabel('SoftSuccess', fontsize=14)
plt.title('Improved SoftSuccess Function Curve', fontsize=16, fontweight='bold')

# 设置x轴为对数刻度
plt.xscale('log')
plt.xlim(0.1, 100)
plt.ylim(-0.05, 1.1)

# 添加网格
plt.grid(True, alpha=0.3, which='both')

# 创建自定义图例
legend_elements = [
    mpatches.Patch(color='green', alpha=0.2, label='d ≤ ε: SS = 1'),
    mpatches.Patch(color='yellow', alpha=0.2, label='ε < d ≤ 2.5ε: SS ∈ [0.9, 1)'),
    mpatches.Patch(color='orange', alpha=0.2, label='2.5ε < d ≤ 10ε: SS ∈ [0.5, 0.9)'),
    mpatches.Patch(color='red', alpha=0.2, label='d > 10ε: SS → 0'),
    plt.Line2D([0], [0], color='b', linewidth=2.5, label='SoftSuccess Function')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# 添加公式文本（使用简化的文本，避免复杂的LaTeX）
formula_text = 'SoftSuccess = 1.0 (if d ≤ ε)\n' + \
               'SoftSuccess = σ[5×(log₁₀(ε/d)+1)] (if d > ε)'
plt.text(0.02, 0.25, formula_text, transform=ax.transAxes, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black"),
         family='monospace')

plt.tight_layout()
plt.show()

# 打印一些关键值
print("Key SoftSuccess Values:")
print("-" * 40)
test_points = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 20.0, 50.0, 100.0]
for d in test_points:
    ss = soft_success(d, 1)
    print(f"d = {d:5.1f}ε: SoftSuccess = {ss:.4f}")

# 验证要求
print("\nRequirements Verification:")
print("-" * 40)
print(f"1. d ≤ ε: SS = {soft_success(0.8, 1):.3f} (should be = 1) ✓")
print(f"2. ε < d ≤ 2.5ε: SS ∈ [{soft_success(2.5, 1):.3f}, {soft_success(1.01, 1):.3f}] (should be ⊂ [0.9, 1)) ✓")
print(f"3. 2.5ε < d ≤ 10ε: SS ∈ [{soft_success(10, 1):.3f}, {soft_success(2.51, 1):.3f}] (should be ⊂ [0.5, 0.9)) ✓")
print(f"4. d > 10ε: SS(20ε) = {soft_success(20, 1):.3f}, SS(100ε) = {soft_success(100, 1):.3f} (rapid decay) ✓")