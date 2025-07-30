# 评估指标转换指导：从 Hard Success 到 Soft Success

## 任务说明

请将当前的 **Hard Success** 评估指标转换为 **Soft Success** 评估指标。Hard Success 是一个二元指标（0或1），而 Soft Success 是一个连续指标（0到1之间），能够更细致地反映模型解决方案的质量。

## Hard Success 定义（原始指标）

```
Hard Success = 1 if d ≤ ε else 0
```
其中：
- `d` 是模型解决方案与参考解之间的距离
- `ε` 是容差阈值

## Soft Success 定义（新指标）

请使用以下公式计算 Soft Success：

```
SoftSuccess = {
    1.0                                                          if d ≤ ε
    α·exp(-β·(r-1)^γ) + (1-α)·(1/(1 + ω·(r-1)^δ))              if d > ε
}
```

其中：
- `r = d/ε` （归一化距离）
- 参数值：
  - `α = 0.7`
  - `β = 0.05`
  - `γ = 1.5`
  - `ω = 0.3`
  - `δ = 2.2`

## 实现要求

### 1. 单个 (d, ε) 对的计算

```python
def soft_success(d, epsilon):
    """计算单个 (d, epsilon) 对的 Soft Success 值"""
    r = d / epsilon
    
    if r <= 1:
        return 1.0
    
    # 参数
    alpha = 0.7
    beta = 0.05
    gamma = 1.5
    omega = 0.3
    delta = 2.2
    
    # 双组分衰减函数
    exp_component = np.exp(-beta * (r - 1)**gamma)
    logistic_component = 1 / (1 + omega * (r - 1)**delta)
    
    return alpha * exp_component + (1 - alpha) * logistic_component
```

### 2. 多个 (d, ε) 对的处理

当存在多个 (d, ε) 对时：
1. 对每个 (d_i, ε_i) 独立计算 SS_i
2. 取所有 SS_i 的**算术平均值**作为最终结果

```python
def soft_success_multi(d_list, epsilon_list):
    """计算多个 (d, epsilon) 对的平均 Soft Success 值"""
    ss_values = []
    for d, eps in zip(d_list, epsilon_list):
        ss = soft_success(d, eps)
        ss_values.append(ss)
    
    return np.mean(ss_values)  # 算术平均
```

## 转换步骤

1. **识别 Hard Success 计算位置**：找到代码中在计算 Efficiency 时使用 Hard Success 的地方
2. **替换为 Soft Success 函数**：将二元判断替换为上述连续函数
3. **处理批量计算**：如果有多个 tolerance，确保正确处理多个 (d, ε) 对
4. **更新相关文档和注释**：说明指标从二元变为连续值 [0, 1]

## 示例转换

### 转换前（Hard Success）
```python
# 原始代码
success = 1 if distance <= tolerance else 0
```

### 转换后（Soft Success）
```python
# 新代码
success = soft_success(distance, tolerance)
```

## 注意事项

1. **数值稳定性**：确保 epsilon 不为 0，避免除零错误
2. **向量化优化**：对于大批量计算，考虑使用 NumPy 向量化操作
3. **结果解释**：
   - SS = 1.0：完美成功（d ≤ ε）
   - SS ∈ [0.9, 1.0)：接近成功
   - SS ∈ [0.1, 0.6]：部分成功
   - SS < 0.1：基本失败

## 验证指标行为

转换后，请验证 Soft Success 在关键点的表现：
- 当 d = ε 时：SS = 1.0
- 当 d = 2.5ε 时：SS ≈ 0.8
- 当 d = 10ε 时：SS ≈ 0.18
- 当 d = 20ε 时：SS ≈ 0.01

这样的设计使得指标能够：
- 在模型"接近成功"时给予较高分数（鼓励接近正确的尝试）
- 在模型"完全失败"时快速衰减到 0
- 提供更细粒度的模型性能评估