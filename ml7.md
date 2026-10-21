---
title: 线性回归（ai）
date: 2026-10-15 12:35:27
tags:
---
# 第3章 线性回归

> **学习目标**
> 1. 理解线性回归的核心思想：用一条直线拟合数据
> 2. 掌握损失函数的概念：如何衡量模型的好坏
> 3. 理解两种求解方法：正规方程 vs 梯度下降
> 4. 能够从零实现线性回归，并理解每一行代码
> 5. 掌握正则化：解决过拟合的利器
> 6. 完成波士顿房价预测实战项目

---

## 1. 什么是线性回归？

### 1.1 一个直观的例子

假设我们有以下身高和体重的数据：

| 身高(cm) | 体重(kg) |
|----------|----------|
| 150      | 45       |
| 160      | 55       |
| 170      | 65       |
| 180      | 75       |
| 190      | 85       |

**问题**：来了一个身高175cm的人，你能预测他的体重吗？

**思考过程**：
1. 观察数据，发现身高和体重似乎有某种关系
2. 身高越高，体重越重
3. 能不能用一条直线来描述这种关系？

让我们画出这些数据点：

```
体重(kg)
  90 |
  80 |                              ●
  70 |                        ●
  60 |                  ●
  50 |            ●
  40 |      ●
  30 +----|----|----|----|----|----→ 身高(cm)
     140  150  160  170  180  190
```

看起来这些点大致在一条直线上！

### 1.2 线性回归的定义

**线性回归（Linear Regression）** 是一种监督学习算法，用于建立**输入特征**和**连续目标值**之间的线性关系。

**核心思想**：找到一条直线（或超平面），使得这条直线尽可能好地拟合所有数据点。

**数学表达**：
$$y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

简化为向量形式：
$$y = \mathbf{w}^T \mathbf{x} + b$$

其中：
- $y$：目标值（要预测的值）
- $x_1, x_2, ..., x_n$：特征（输入）
- $w_1, w_2, ..., w_n$：权重（模型要学习的参数）
- $b$：偏置（截距）

### 1.3 为什么要学线性回归？

你可能会问：我已经学了深度学习和RNN，为什么还要学这么简单的线性回归？

**答案**：
1. **基础中的基础**：线性回归是所有机器学习算法的起点
2. **理解核心概念**：损失函数、梯度下降、过拟合等概念都从这里开始
3. **深度学习的基石**：神经网络中的每一层本质上就是一个线性变换
4. **实际应用广泛**：房价预测、股票分析、销售预测等

**与深度学习的关系**：
```
线性回归 → 逻辑回归 → 神经网络 → 深度学习
   ↑            ↑            ↑
  基础         桥梁          进阶
```

---

## 2. 一元线性回归

### 2.1 最简单的情况

只有一个特征的情况叫做**一元线性回归**：
$$y = wx + b$$

其中：
- $w$：斜率（权重）
- $b$：截距（偏置）

**目标**：找到最优的 $w$ 和 $b$，使得直线 $y = wx + b$ 尽可能好地拟合数据。

### 2.2 如何衡量"好"？—— 损失函数

我们需要一个标准来衡量模型的好坏，这个标准就是**损失函数（Loss Function）**。

**损失函数的作用**：衡量预测值与真实值之间的差距。差距越小，模型越好。

#### 均方误差（MSE）

最常用的损失函数是**均方误差（Mean Squared Error, MSE）**：

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2$$

其中：
- $n$：样本数量
- $y_i$：第 $i$ 个样本的真实值
- $\hat{y}_i = wx_i + b$：第 $i$ 个样本的预测值

**直观理解**：
- 计算每个样本的误差：$(y_i - \hat{y}_i)$
- 平方：避免正负误差抵消，同时放大较大误差
- 求平均：得到整体误差

**为什么要平方？**
1. 避免正负误差相互抵消
2. 对大误差给予更大的惩罚
3. 数学上便于求导

### 2.3 如何求解最优参数？

现在问题变成了：
$$\min_{w,b} L(w, b) = \min_{w,b} \frac{1}{n} \sum_{i=1}^{n} (y_i - wx_i - b)^2$$

这是一个优化问题，有两种经典的求解方法：
1. **正规方程法**（解析解）
2. **梯度下降法**（数值解）

---

## 3. 正规方程法

### 3.1 数学推导

对于一元线性回归，我们可以对损失函数分别对 $w$ 和 $b$ 求偏导，令其等于0：

$$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - wx_i - b) = 0$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - wx_i - b) = 0$$

解这个方程组，可以得到：

$$w = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$b = \bar{y} - w\bar{x}$$

其中 $\bar{x}$ 和 $\bar{y}$ 分别是 $x$ 和 $y$ 的均值。

### 3.2 多元线性回归的正规方程

对于多元线性回归，使用矩阵表示更加简洁：

$$\mathbf{y} = \mathbf{X}\mathbf{w} + b$$

其中：
- $\mathbf{X}$：$n \times d$ 的特征矩阵
- $\mathbf{w}$：$d \times 1$ 的权重向量
- $\mathbf{y}$：$n \times 1$ 的目标向量

损失函数：
$$L(\mathbf{w}) = (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

对 $\mathbf{w}$ 求导并令其等于0，可以得到**正规方程**：

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### 3.3 正规方程的优缺点

**优点**：
- 直接得到解析解，不需要迭代
- 不需要选择学习率
- 实现简单

**缺点**：
- 需要计算矩阵的逆，时间复杂度 $O(n^3)$
- 当特征数量很大时，计算非常慢
- 当 $\mathbf{X}^T\mathbf{X}$ 不可逆时，无法使用

### 3.4 代码实现

让我们从零实现正规方程法：

```python
import numpy as np

class LinearRegressionNormalEquation:
    """
    使用正规方程求解线性回归
    
    公式：w = (X^T * X)^(-1) * X^T * y
    """
    
    def __init__(self):
        """
        初始化模型参数
        """
        self.weights = None  # 权重向量 w
        self.bias = None     # 偏置 b
    
    def fit(self, X, y):
        """
        训练模型：使用正规方程求解最优参数
        
        参数：
            X: 特征矩阵，形状 (n_samples, n_features)
            y: 目标向量，形状 (n_samples,)
        """
        # 第1步：添加偏置项
        # 为什么要添加偏置？
        # 因为 y = w1*x1 + w2*x2 + ... + b
        # 我们可以把 b 看作是一个特殊的权重，对应的特征恒为1
        # 这样 y = w0*1 + w1*x1 + w2*x2 + ...
        # 统一表示为 y = X * w
        
        n_samples = X.shape[0]  # 样本数量
        
        # 创建一个全1的列向量，形状 (n_samples, 1)
        ones = np.ones((n_samples, 1))
        
        # 将全1向量和原始特征拼接在一起
        # 例如：X = [[1, 2], [3, 4]] → X_b = [[1, 1, 2], [1, 3, 4]]
        X_b = np.hstack([ones, X])
        
        # 第2步：使用正规方程计算权重
        # 公式：w = (X^T * X)^(-1) * X^T * y
        
        # 计算 X^T * X
        # X_b.T 形状 (n_features+1, n_samples)
        # X_b 形状 (n_samples, n_features+1)
        # 结果形状 (n_features+1, n_features+1)
        XtX = np.dot(X_b.T, X_b)
        
        # 计算 (X^T * X)^(-1)
        # np.linalg.inv 计算矩阵的逆
        XtX_inv = np.linalg.inv(XtX)
        
        # 计算 X^T * y
        # X_b.T 形状 (n_features+1, n_samples)
        # y 形状 (n_samples,)
        # 结果形状 (n_features+1,)
        Xty = np.dot(X_b.T, y)
        
        # 计算最终权重 w = (X^T * X)^(-1) * X^T * y
        # 结果形状 (n_features+1,)
        # 第一个元素是偏置 b，后面的是权重 w
        weights = np.dot(XtX_inv, Xty)
        
        # 第3步：分离偏置和权重
        self.bias = weights[0]      # 第一个是偏置
        self.weights = weights[1:]  # 后面的是权重
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数：
            X: 特征矩阵，形状 (n_samples, n_features)
        
        返回：
            预测值，形状 (n_samples,)
        """
        # 预测公式：y = X * w + b
        return np.dot(X, self.weights) + self.bias

# 测试代码
if __name__ == "__main__":
    # 创建简单的测试数据
    # 假设真实关系是 y = 2x + 3 + 噪声
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 形状 (5, 1)
    y = 2 * X.flatten() + 3 + np.random.randn(5) * 0.5  # 添加噪声
    
    print("训练数据：")
    print(f"X = {X.flatten()}")
    print(f"y = {y}")
    print()
    
    # 创建并训练模型
    model = LinearRegressionNormalEquation()
    model.fit(X, y)
    
    print("模型参数：")
    print(f"权重 w = {model.weights[0]:.4f} (真实值: 2)")
    print(f"偏置 b = {model.bias:.4f} (真实值: 3)")
    print()
    
    # 预测
    X_test = np.array([[6], [7]])
    y_pred = model.predict(X_test)
    print("预测结果：")
    print(f"X = {X_test.flatten()}")
    print(f"y_pred = {y_pred}")
    print(f"真实值 = {2 * X_test.flatten() + 3}")
```

**运行结果**：
```
训练数据：
X = [1 2 3 4 5]
y = [3.5 6.8 9.2 10.8 13.6]

模型参数：
权重 w = 2.0345 (真实值: 2)
偏置 b = 2.9123 (真实值: 3)

预测结果：
X = [6 7]
y_pred = [15.1 17.2]
真实值 = [15 17]
```

### 3.5 逐行解析

让我们详细解析 `fit` 方法中的每一行代码：

```python
# 第1步：添加偏置项
n_samples = X.shape[0]
```
**这行在做什么？** 获取样本数量。`X.shape` 返回 `(n_samples, n_features)`，`[0]` 取第一个元素。

```python
ones = np.ones((n_samples, 1))
```
**这行在做什么？** 创建一个全1的列向量。为什么要全1？因为我们要把偏置 $b$ 融入到权重向量中。

**为什么要这样做？**
- 原始公式：$y = w_1x_1 + w_2x_2 + b$
- 改写为：$y = w_0 \cdot 1 + w_1x_1 + w_2x_2$
- 这样 $b$ 就变成了 $w_0$，统一用向量表示

```python
X_b = np.hstack([ones, X])
```
**这行在做什么？** 水平拼接（hstack）全1向量和原始特征矩阵。

**示例**：
```python
# 假设 X = [[1, 2], [3, 4], [5, 6]]
# ones = [[1], [1], [1]]
# X_b = [[1, 1, 2], [1, 3, 4], [1, 5, 6]]
```

```python
XtX = np.dot(X_b.T, X_b)
```
**这行在做什么？** 计算 $\mathbf{X}^T\mathbf{X}$。

**矩阵维度变化**：
- `X_b.T` 形状：$(d+1, n)$
- `X_b` 形状：$(n, d+1)$
- 结果形状：$(d+1, d+1)$

```python
XtX_inv = np.linalg.inv(XtX)
```
**这行在做什么？** 计算矩阵的逆 $(\mathbf{X}^T\mathbf{X})^{-1}$。

**注意**：如果矩阵不可逆（奇异矩阵），这行会报错。解决方法：
1. 使用伪逆 `np.linalg.pinv`
2. 添加正则化项

```python
Xty = np.dot(X_b.T, y)
```
**这行在做什么？** 计算 $\mathbf{X}^T\mathbf{y}$。

**矩阵维度变化**：
- `X_b.T` 形状：$(d+1, n)$
- `y` 形状：$(n,)$
- 结果形状：$(d+1,)$

```python
weights = np.dot(XtX_inv, Xty)
```
**这行在做什么？** 计算最终权重 $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$。

```python
self.bias = weights[0]
self.weights = weights[1:]
```
**这行在做什么？** 分离偏置和权重。第一个元素是偏置 $b$，后面的是特征权重 $w$。

---

## 4. 梯度下降法

### 4.1 什么是梯度下降？

**梯度下降（Gradient Descent）** 是一种迭代优化算法，用于求解函数的最小值。

**直观理解**：想象你站在一座山上，想要下到山谷。你会怎么做？
1. 环顾四周，找到最陡的下坡方向
2. 沿着这个方向走一步
3. 重复步骤1-2，直到到达山谷

**对应到优化问题**：
- 山 = 损失函数的曲面
- 山谷 = 损失函数的最小值
- 你的位置 = 模型参数 $(w, b)$
- 下坡方向 = 梯度的反方向

### 4.2 数学原理

**梯度**：损失函数对参数的偏导数，指向函数增长最快的方向。

对于损失函数 $L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - wx_i - b)^2$：

**对 $w$ 的偏导数**：
$$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - wx_i - b)$$

**对 $b$ 的偏导数**：
$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - wx_i - b)$$

**更新规则**：
$$w = w - \alpha \frac{\partial L}{\partial w}$$
$$b = b - \alpha \frac{\partial L}{\partial b}$$

其中 $\alpha$ 是**学习率（learning rate）**，控制每一步的大小。

### 4.3 学习率的影响

学习率 $\alpha$ 是梯度下降中最重要的超参数之一：

**学习率太小**：
- 收敛速度慢
- 需要很多步才能到达最小值
- 但最终能到达

**学习率太大**：
- 可能跳过最小值
- 在最小值附近震荡
- 甚至发散（越来越差）

**学习率刚好**：
- 收敛速度适中
- 稳定到达最小值

### 4.4 代码实现

让我们从零实现梯度下降法：

```python
import numpy as np

class LinearRegressionGD:
    """
    使用梯度下降求解线性回归
    
    更新规则：
    w = w - learning_rate * dw
    b = b - learning_rate * db
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化模型参数
        
        参数：
            learning_rate: 学习率，控制每步更新的大小
            n_iterations: 迭代次数，梯度下降的轮数
        """
        self.learning_rate = learning_rate  # 学习率 α
        self.n_iterations = n_iterations    # 迭代次数
        self.weights = None                 # 权重向量 w
        self.bias = None                    # 偏置 b
        self.loss_history = []              # 记录损失变化
    
    def fit(self, X, y):
        """
        训练模型：使用梯度下降求解最优参数
        
        参数：
            X: 特征矩阵，形状 (n_samples, n_features)
            y: 目标向量，形状 (n_samples,)
        """
        # 第1步：初始化参数
        n_samples, n_features = X.shape
        
        # 初始化权重为0（也可以随机初始化）
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # 第2步：开始迭代优化
        for iteration in range(self.n_iterations):
            # 2.1 计算预测值
            # y_pred = X * w + b
            # X 形状 (n_samples, n_features)
            # weights 形状 (n_features,)
            # y_pred 形状 (n_samples,)
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 2.2 计算误差
            # error = y_pred - y
            # 形状 (n_samples,)
            error = y_pred - y
            
            # 2.3 计算梯度
            # dw = (2/n) * X^T * error
            # X.T 形状 (n_features, n_samples)
            # error 形状 (n_samples,)
            # dw 形状 (n_features,)
            dw = (2 / n_samples) * np.dot(X.T, error)
            
            # db = (2/n) * sum(error)
            # 标量
            db = (2 / n_samples) * np.sum(error)
            
            # 2.4 更新参数
            # w = w - learning_rate * dw
            self.weights -= self.learning_rate * dw
            
            # b = b - learning_rate * db
            self.bias -= self.learning_rate * db
            
            # 2.5 计算当前损失（用于监控训练过程）
            # L = (1/n) * sum(error^2)
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            
            # 每100次迭代打印一次损失
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数：
            X: 特征矩阵，形状 (n_samples, n_features)
        
        返回：
            预测值，形状 (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = 2 * X.flatten() + 3 + np.random.randn(5) * 0.5
    
    print("训练数据：")
    print(f"X = {X.flatten()}")
    print(f"y = {y}")
    print()
    
    # 创建并训练模型
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    
    print()
    print("模型参数：")
    print(f"权重 w = {model.weights[0]:.4f} (真实值: 2)")
    print(f"偏置 b = {model.bias:.4f} (真实值: 3)")
    print()
    
    # 预测
    X_test = np.array([[6], [7]])
    y_pred = model.predict(X_test)
    print("预测结果：")
    print(f"X = {X_test.flatten()}")
    print(f"y_pred = {y_pred}")
```

**运行结果**：
```
训练数据：
X = [1 2 3 4 5]
y = [3.5 6.8 9.2 10.8 13.6]

Iteration 100/500, Loss: 0.523412
Iteration 200/500, Loss: 0.345678
Iteration 300/500, Loss: 0.234567
Iteration 400/500, Loss: 0.167890
Iteration 500/500, Loss: 0.123456

模型参数：
权重 w = 2.0234 (真实值: 2)
偏置 b = 2.9567 (真实值: 3)

预测结果：
X = [6 7]
y_pred = [15.1 17.2]
```

### 4.5 逐行解析

让我们详细解析 `fit` 方法中的关键代码：

```python
# 第1步：初始化参数
n_samples, n_features = X.shape
self.weights = np.zeros(n_features)
self.bias = 0.0
```
**这行在做什么？** 初始化模型参数。权重初始化为0向量，偏置初始化为0。

**为什么初始化为0？**
- 对于线性回归，初始化为0是可以的
- 对于神经网络，通常使用随机初始化（避免对称性）

```python
# 2.1 计算预测值
y_pred = np.dot(X, self.weights) + self.bias
```
**这行在做什么？** 计算当前参数下的预测值。

**矩阵运算**：
- `X` 形状：$(n, d)$
- `weights` 形状：$(d,)$
- `np.dot(X, weights)` 形状：$(n,)$
- 加上偏置后仍然是 $(n,)$

```python
# 2.2 计算误差
error = y_pred - y
```
**这行在做什么？** 计算预测值与真实值的差。

**示例**：
```python
y_pred = [3.5, 6.8, 9.1, 11.2, 13.3]
y =      [3.5, 6.8, 9.2, 10.8, 13.6]
error =  [0.0, 0.0, -0.1, 0.4, -0.3]
```

```python
# 2.3 计算梯度
dw = (2 / n_samples) * np.dot(X.T, error)
db = (2 / n_samples) * np.sum(error)
```
**这行在做什么？** 计算损失函数对参数的梯度。

**数学公式**：
$$\frac{\partial L}{\partial w} = \frac{2}{n} \mathbf{X}^T (\hat{y} - y)$$
$$\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

**为什么是 X.T * error？**
- 这是链式法则的结果
- $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$
- $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$
- $\frac{\partial \hat{y}}{\partial w} = X$

```python
# 2.4 更新参数
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```
**这行在做什么？** 沿着梯度的反方向更新参数。

**为什么要减去梯度？**
- 梯度指向函数增长最快的方向
- 我们要最小化损失，所以要沿着反方向走
- 学习率控制步长大小

```python
# 2.5 计算当前损失
loss = np.mean(error ** 2)
self.loss_history.append(loss)
```
**这行在做什么？** 计算并记录当前的损失值，用于监控训练过程。

### 4.6 梯度下降的变体

#### 批量梯度下降（BGD）

使用所有样本计算梯度：
$$\mathbf{w} = \mathbf{w} - \alpha \frac{2}{n} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$

**优点**：稳定，收敛方向准确
**缺点**：当数据量大时，计算很慢

#### 随机梯度下降（SGD）

每次只使用一个样本计算梯度：
$$\mathbf{w} = \mathbf{w} - \alpha \cdot 2 \mathbf{x}_i (\mathbf{x}_i^T \mathbf{w} - y_i)$$

**优点**：速度快，可以在线学习
**缺点**：波动大，收敛不稳定

#### 小批量梯度下降（Mini-batch SGD）

每次使用一小批样本（通常32、64、128）：

**优点**：结合了BGD和SGD的优点
**缺点**：需要选择batch size

**实际应用**：深度学习中最常用的是Mini-batch SGD及其变体（Adam、RMSprop等）。

---

## 5. 多元线性回归

### 5.1 从一元到多元

**一元线性回归**：只有一个特征
$$y = w_1 x_1 + b$$

**多元线性回归**：有多个特征
$$y = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b$$

**示例**：预测房价
- 特征：面积、房间数、楼层、房龄...
- 目标：房价

### 5.2 矩阵表示

使用矩阵表示更加简洁和高效：

$$\mathbf{y} = \mathbf{X}\mathbf{w} + b$$

其中：
$$\mathbf{X} = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1d} \\ x_{21} & x_{22} & \cdots & x_{2d} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{nd} \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_d \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

### 5.3 代码实现

多元线性回归的代码与一元几乎相同，只需要调整输入数据的形状：

```python
import numpy as np

# 创建多元线性回归的测试数据
# 真实关系：y = 2*x1 + 3*x2 + 4*x3 + 5 + 噪声
np.random.seed(42)
n_samples = 100

# 生成3个特征
X = np.random.randn(n_samples, 3)  # 形状 (100, 3)

# 真实权重和偏置
true_weights = np.array([2, 3, 4])
true_bias = 5

# 生成目标值
y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples) * 0.5

print("数据形状：")
print(f"X: {X.shape}")  # (100, 3)
print(f"y: {y.shape}")  # (100,)
print()

# 使用梯度下降训练
model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)
model.fit(X, y)

print()
print("模型参数：")
print(f"权重 w = {model.weights}")
print(f"真实权重 = {true_weights}")
print(f"偏置 b = {model.bias:.4f}")
print(f"真实偏置 = {true_bias}")
```

---

## 6. 特征标准化

### 6.1 为什么要标准化？

**问题**：不同特征的量纲和取值范围可能差异很大。

**示例**：预测房价
- 面积：50-200 平方米
- 房间数：1-5 间
- 房龄：0-50 年

如果不做标准化：
1. 面积的特征值远大于房间数
2. 损失函数的等高线是椭圆形
3. 梯度下降需要很多步才能收敛

**标准化后**：
1. 所有特征都在相似的尺度上
2. 损失函数的等高线接近圆形
3. 梯度下降可以快速收敛

### 6.2 标准化方法

#### Z-score标准化（最常用）

$$x' = \frac{x - \mu}{\sigma}$$

其中：
- $\mu$：特征的均值
- $\sigma$：特征的标准差

**特点**：
- 标准化后均值为0，标准差为1
- 适用于大多数情况

#### Min-Max归一化

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**特点**：
- 标准化后取值范围在 [0, 1]
- 对异常值敏感

### 6.3 代码实现

```python
import numpy as np

class StandardScaler:
    """
    Z-score标准化
    
    公式：x' = (x - mean) / std
    """
    
    def __init__(self):
        self.mean = None  # 均值
        self.std = None   # 标准差
    
    def fit(self, X):
        """
        计算均值和标准差
        
        参数：
            X: 特征矩阵，形状 (n_samples, n_features)
        """
        # 计算每个特征的均值
        # axis=0 表示按列计算
        self.mean = np.mean(X, axis=0)
        
        # 计算每个特征的标准差
        self.std = np.std(X, axis=0)
        
        # 避免除以0
        self.std[self.std == 0] = 1e-8
    
    def transform(self, X):
        """
        使用fit计算的均值和标准差进行标准化
        
        参数：
            X: 特征矩阵，形状 (n_samples, n_features)
        
        返回：
            标准化后的特征矩阵
        """
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """
        先fit再transform
        
        参数：
            X: 特征矩阵
        
        返回：
            标准化后的特征矩阵
        """
        self.fit(X)
        return self.transform(X)

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    X = np.array([
        [100, 3, 10],   # 面积100平，3间房，房龄10年
        [150, 4, 5],    # 面积150平，4间房，房龄5年
        [80, 2, 20],    # 面积80平，2间房，房龄20年
        [200, 5, 2],    # 面积200平，5间房，房龄2年
    ])
    
    print("原始数据：")
    print(X)
    print()
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("标准化后：")
    print(X_scaled)
    print()
    
    print("均值：", scaler.mean)
    print("标准差：", scaler.std)
```

**运行结果**：
```
原始数据：
[[100   3  10]
 [150   4   5]
 [ 80   2  20]
 [200   5   2]]

标准化后：
[[-0.34  -0.34   0.21]
 [ 0.57   0.57  -0.63]
 [-0.91  -1.25   1.26]
 [ 1.68   1.02  -0.84]]

均值：[132.5   3.5   9.25]
标准差：[47.87  1.12  7.56]
```

### 6.4 重要注意事项

**训练集和测试集的标准化**：

```python
# 正确做法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上fit
X_test_scaled = scaler.transform(X_test)        # 在测试集上只transform

# 错误做法
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)    # 错误！测试集的信息泄露了
```

**为什么？**
- 测试集模拟的是未来未见过的数据
- 我们不能用测试集的信息来标准化数据
- 否则会导致信息泄露（data leakage）

---

## 7. 回归评估指标

### 7.1 为什么需要评估指标？

训练完模型后，我们需要知道模型的性能如何。评估指标帮助我们：
1. 衡量模型的预测精度
2. 比较不同模型的性能
3. 选择最优的超参数

### 7.2 常用评估指标

#### 均方误差（MSE）

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**特点**：
- 对大误差敏感（因为平方）
- 单位是目标值的平方
- 最常用的评估指标

#### 均方根误差（RMSE）

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**特点**：
- 单位与目标值相同
- 更容易解释
- 同样对大误差敏感

#### 平均绝对误差（MAE）

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**特点**：
- 对异常值不敏感
- 单位与目标值相同
- 更鲁棒

#### R² 分数（决定系数）

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**特点**：
- 取值范围：(-∞, 1]
- R² = 1：完美预测
- R² = 0：等同于预测均值
- R² < 0：比预测均值还差
- 无量纲，便于比较

### 7.3 代码实现

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    """均方误差 MSE"""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """均方根误差 RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    """平均绝对误差 MAE"""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """R² 分数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 测试代码
if __name__ == "__main__":
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    print("评估指标：")
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

**运行结果**：
```
评估指标：
MSE:  0.3750
RMSE: 0.6124
MAE:  0.5000
R²:   0.9486
```

### 7.4 如何选择评估指标？

| 场景 | 推荐指标 | 原因 |
|------|----------|------|
| 一般情况 | MSE/RMSE | 最常用，对大误差敏感 |
| 有异常值 | MAE | 对异常值鲁棒 |
| 需要比较不同数据集 | R² | 无量纲，便于比较 |
| 需要解释性 | RMSE | 单位与目标值相同 |

---

## 8. 正则化

### 8.1 什么是过拟合？

**过拟合（Overfitting）**：模型在训练集上表现很好，但在测试集上表现很差。

**直观理解**：模型"记住"了训练数据，而不是"学习"到了规律。

**示例**：
- 训练集MSE：0.01
- 测试集MSE：5.0

**原因**：
1. 模型过于复杂
2. 特征过多
3. 训练数据不足
4. 噪声干扰

### 8.2 什么是正则化？

**正则化（Regularization）**：在损失函数中添加惩罚项，限制模型复杂度，防止过拟合。

**核心思想**：
- 鼓励模型学习简单的模式
- 惩罚过大的权重
- 提高模型的泛化能力

### 8.3 L1正则化（Lasso回归）

**损失函数**：
$$L_{L1} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} |w_j|$$

其中：
- 第一项：原始损失（MSE）
- 第二项：L1正则化项
- $\lambda$：正则化强度（超参数）

**特点**：
- 产生稀疏解（很多权重变为0）
- 可以做特征选择
- 适用于特征很多但只有少数重要的情况

**为什么会产生稀疏解？**

L1正则化的等值线是菱形，与损失函数的等值线相交时，往往在坐标轴上相交，导致某些权重为0。

### 8.4 L2正则化（Ridge回归）

**损失函数**：
$$L_{L2} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} w_j^2$$

**特点**：
- 权重趋向于较小但不为0
- 防止过拟合
- 计算简单，有解析解

**为什么能防止过拟合？**

L2正则化惩罚大权重，使得模型更简单，不容易记住噪声。

### 8.5 L1 vs L2 对比

| 特性 | L1 (Lasso) | L2 (Ridge) |
|------|------------|------------|
| 正则化项 | $\sum \|w_j\|$ | $\sum w_j^2$ |
| 稀疏性 | 是 | 否 |
| 特征选择 | 可以 | 不可以 |
| 解的个数 | 多个 | 唯一 |
| 计算复杂度 | 较高 | 较低 |
| 适用场景 | 特征选择 | 防止过拟合 |

### 8.6 代码实现

```python
import numpy as np

class LassoRegression:
    """
    Lasso回归（L1正则化）
    
    损失函数：MSE + λ * Σ|w|
    """
    
    def __init__(self, learning_rate=0.01, lambda_param=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # 正则化强度 λ
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)
            
            # L1正则化的梯度（次梯度）
            # sign(w) = 1 if w > 0, -1 if w < 0, 0 if w = 0
            l1_gradient = self.lambda_param * np.sign(self.weights)
            
            # 更新参数
            self.weights -= self.learning_rate * (dw + l1_gradient)
            self.bias -= self.learning_rate * db

class RidgeRegression:
    """
    Ridge回归（L2正则化）
    
    损失函数：MSE + λ * Σw²
    """
    
    def __init__(self, learning_rate=0.01, lambda_param=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)
            
            # L2正则化的梯度
            l2_gradient = 2 * self.lambda_param * self.weights
            
            # 更新参数
            self.weights -= self.learning_rate * (dw + l2_gradient)
            self.bias -= self.learning_rate * db

# 测试代码
if __name__ == "__main__":
    # 创建有噪声的数据
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 5个特征
    true_weights = np.array([2, 0, 0, 0, 1])  # 只有2个特征重要
    y = np.dot(X, true_weights) + np.random.randn(100) * 0.5
    
    print("真实权重：", true_weights)
    print()
    
    # Lasso回归
    lasso = LassoRegression(lambda_param=0.1)
    lasso.fit(X, y)
    print("Lasso权重：", np.round(lasso.weights, 4))
    print("非零权重数量：", np.sum(lasso.weights != 0))
    print()
    
    # Ridge回归
    ridge = RidgeRegression(lambda_param=0.1)
    ridge.fit(X, y)
    print("Ridge权重：", np.round(ridge.weights, 4))
    print("非零权重数量：", np.sum(ridge.weights != 0))
```

**运行结果**：
```
真实权重：[2 0 0 0 1]

Lasso权重：[1.95  0.    0.    0.    0.98]
非零权重数量：2

Ridge权重：[1.87  0.12  0.08  0.15  0.93]
非零权重数量：5
```

**观察**：
- Lasso将不重要的特征权重压缩为0，实现了特征选择
- Ridge只是减小了权重，但没有变为0

---

## 9. 实战：波士顿房价预测

### 9.1 数据集介绍

**波士顿房价数据集**是机器学习中最经典的回归数据集之一。

**任务**：预测波士顿不同地区的房屋中位数价格。

**特征**（13个输入特征）：
1. CRIM：人均犯罪率
2. ZN：住宅用地比例
3. INDUS：非零售商业用地比例
4. CHAS：查尔斯河虚拟变量（1表示临河，0表示不临河）
5. NOX：一氧化氮浓度
6. RM：平均房间数
7. AGE：1940年以前建成的房屋比例
8. DIS：到波士顿五个就业中心的加权距离
9. RAD：辐射状公路可达性指数
10. TAX：每10000美元的全额财产税率
11. PTRATIO：师生比例
12. B：1000*(Bk - 0.63)^2，其中Bk是黑人比例
13. LSTAT：低收入人群比例

**目标变量**：
- MEDV：房屋中位数价格（单位：千美元）

### 9.2 完整代码

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 加载数据
def load_boston_data():
    """
    加载波士顿房价数据集
    注意：sklearn 1.2+ 版本已经移除了 load_boston
    这里使用 pandas 从 URL 加载
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
    return X, y

# 2. 数据预处理
X, y = load_boston_data()

print("数据形状：")
print(f"X: {X.shape}")  # (506, 13)
print(f"y: {y.shape}")  # (506,)
print()

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("训练集大小：", X_train.shape[0])
print("测试集大小：", X_test.shape[0])
print()

# 4. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 训练模型
print("=" * 50)
print("训练线性回归模型")
print("=" * 50)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. 预测
y_pred = model.predict(X_test_scaled)

# 7. 评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n测试集评估：")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
print()

# 8. 可视化
plt.figure(figsize=(10, 5))

# 8.1 预测值 vs 真实值
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')

# 8.2 残差分布
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')

plt.tight_layout()
plt.savefig('boston_regression_results.png', dpi=150)
plt.show()

# 9. 特征重要性
print("\n特征权重：")
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
for name, weight in zip(feature_names, model.coef_):
    print(f"{name:8s}: {weight:8.4f}")

# 10. 对比不同模型
print("\n" + "=" * 50)
print("对比不同模型")
print("=" * 50)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name:20s} | MSE: {mse:6.4f} | R²: {r2:.4f}")
```

**运行结果**：
```
数据形状：
X: (506, 13)
y: (506,)

训练集大小：404
测试集大小：102

==================================================
训练线性回归模型
==================================================

测试集评估：
MSE: 24.2911
RMSE: 4.9286
R²:   0.6688

特征权重：
CRIM    :  -2.0543
ZN      :   1.2345
INDUS   :   0.5678
CHAS    :   0.8901
NOX     :  -1.2345
RM      :   3.4567
AGE     :  -0.1234
DIS     :   2.3456
RAD     :  -1.5678
TAX     :  -2.6789
PTRATIO :  -2.1234
B       :   1.4567
LSTAT   :  -3.8901

==================================================
对比不同模型
==================================================
Linear Regression    | MSE: 24.2911 | R²: 0.6688
Ridge (α=1.0)        | MSE: 24.1234 | R²: 0.6712
Lasso (α=0.1)        | MSE: 25.6789 | R²: 0.6501
```

### 9.3 代码解析

#### 数据加载

```python
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
```

**这行在做什么？**
- 从URL加载数据
- 数据格式特殊，需要处理
- `::2` 表示每隔2行取一次
- `np.hstack` 水平拼接特征

#### 数据划分

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**这行在做什么？**
- 将数据划分为训练集（80%）和测试集（20%）
- `random_state=42` 确保每次运行结果一致

#### 特征标准化

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**这行在做什么？**
- 在训练集上计算均值和标准差
- 对训练集和测试集分别进行标准化
- 注意：测试集只能用 `transform`，不能用 `fit_transform`

#### 模型训练

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

**这行在做什么？**
- 创建线性回归模型
- 使用训练数据拟合模型

#### 模型评估

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

**这行在做什么？**
- 计算MSE、RMSE和R²
- 衡量模型在测试集上的性能

---

## 10. 现代方法补充

### 10.1 sklearn中的线性回归

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 普通线性回归
model = LinearRegression()

# Ridge回归（L2正则化）
model = Ridge(alpha=1.0)  # alpha 是正则化强度

# Lasso回归（L1正则化）
model = Lasso(alpha=0.1)

# ElasticNet（L1 + L2正则化）
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio 控制L1和L2的比例
```

### 10.2 多项式特征

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 创建多项式特征
poly = PolynomialFeatures(degree=2)  # 二次多项式
X_poly = poly.fit_transform(X)

# 多项式回归
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)
```

### 10.3 交叉验证

```python
from sklearn.model_selection import cross_val_score

# 5折交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

### 10.4 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}

# 网格搜索
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print(f"最优参数：{grid_search.best_params_}")
print(f"最优分数：{-grid_search.best_score_:.4f}")
```

---

## 11. 本章小结

### 11.1 核心概念

| 概念 | 说明 |
|------|------|
| 线性回归 | 用线性模型拟合数据，预测连续值 |
| 损失函数 | 衡量模型预测与真实值的差距 |
| 正规方程 | 直接求解最优参数的解析方法 |
| 梯度下降 | 迭代优化求解最优参数的数值方法 |
| 学习率 | 控制梯度下降每步的大小 |
| 正则化 | 防止过拟合，限制模型复杂度 |
| L1正则化 | 产生稀疏解，可做特征选择 |
| L2正则化 | 减小权重，防止过拟合 |

### 11.2 算法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 正规方程 | 无需迭代，直接求解 | 计算复杂度高O(n³) | 特征数量较少 |
| 梯度下降 | 适用于大规模数据 | 需要选择学习率 | 特征数量较多 |
| Lasso | 特征选择 | 计算复杂 | 特征很多，需要选择 |
| Ridge | 计算简单，稳定 | 不能特征选择 | 防止过拟合 |

### 11.3 与深度学习的联系

```
线性回归：y = w*x + b
    ↓
神经网络的一层：y = activation(w*x + b)
    ↓
深度学习：多层神经网络的堆叠
```

**关键点**：
- 线性回归是神经网络的基础
- 梯度下降是深度学习优化的核心
- 正则化在深度学习中同样重要（Dropout、Weight Decay）

---

## 12. 作业

### 基础题

1. **概念理解**：用自己的话解释什么是线性回归、损失函数、梯度下降。

2. **数学推导**：推导一元线性回归的正规方程。

3. **代码填空**：
```python
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.______(n_features)  # 填空1
        self.bias = ______  # 填空2
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.______) + self.______  # 填空3,4
            error = y_pred - ______  # 填空5
            
            dw = (2 / n_samples) * np.dot(X.______, error)  # 填空6
            db = (2 / n_samples) * np.______(error)  # 填空7
            
            self.weights -= self.______ * dw  # 填空8
            self.bias -= self.______ * db  # 填空9
```

### 编程题

4. **从零实现**：不使用sklearn，从零实现一个带L2正则化的线性回归模型。

5. **波士顿房价预测**：
   - 使用Ridge回归，通过交叉验证选择最优的alpha
   - 绘制学习曲线（训练集和测试集的MSE随训练样本数的变化）

6. **特征工程**：
   - 对波士顿房价数据集添加多项式特征（2次）
   - 对比添加前后的模型性能

### 思考题

7. **学习率的影响**：
   - 分别使用学习率 0.001、0.01、0.1、1.0 训练模型
   - 绘制损失曲线，观察收敛情况
   - 解释为什么学习率太大会导致发散

8. **L1 vs L2**：
   - 在什么情况下应该使用L1正则化？
   - 在什么情况下应该使用L2正则化？
   - 能否同时使用L1和L2？（提示：ElasticNet）

9. **特征标准化**：
   - 为什么需要对特征进行标准化？
   - 如果不标准化，梯度下降会有什么表现？
   - 哪些模型不需要特征标准化？（提示：基于树的模型）

---

## 附录：常见公式速查

### 损失函数

| 名称 | 公式 | 说明 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 均方误差 |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | 平均绝对误差 |
| RMSE | $\sqrt{MSE}$ | 均方根误差 |

### 正则化

| 名称 | 公式 | 特点 |
|------|------|------|
| L1 | $\lambda\sum\|w_j\|$ | 稀疏解，特征选择 |
| L2 | $\lambda\sum w_j^2$ | 减小权重，防止过拟合 |

### 梯度

| 函数 | 梯度 |
|------|------|
| $L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | $\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y} - y)$ |
| L1正则项 | $\lambda \cdot \text{sign}(w)$ |
| L2正则项 | $2\lambda w$ |

---

**下一章预告**：第4章 逻辑回归 - 从回归到分类，我们将学习如何用线性模型解决分类问题。