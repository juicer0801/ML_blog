---
title: 线性回归
date: 2026-10-15 12:35:23
tags:
---

# 线性回归(LinearRegressor)

{% asset_img image-20230912090316271.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-17_23-13-29.png "Hexo 博客封面示例" %}

## 线性回归介绍

**学习目标：**

1.理解线性回归是什么？

2.知道一元线性回归和多元线性回归的区别

3.知道线性回归的应用场景

### 【理解】举个栗子

假若有了身高和体重数据，来了播仔的身高，你能预测播仔体重吗?

{% asset_img image-20230901101426794.png "Hexo 博客封面示例" %}

 这是一个回归问题，该如何求解呢?

**思路**:先从已知身高X和体重Y中找规律，再预测

{% asset_img image-20230901101621761.png "Hexo 博客封面示例" %}

•数学问题：用一条线来拟合身高和体重之间的关系，再对新数据进行预测

{% asset_img image-20230901101918502.png "Hexo 博客封面示例" %}

{% asset_img image-20230901101931661.png "Hexo 博客封面示例" %}

方程 Y = kX + b

k160 + b = 56.3  -- (1)

k166 + b = 60.6  –- (2)

。。。。

k: 斜率  b:截距

若：y = 0.9 x + (-93)

​    0.9*176 +（-93）= ? 

### 【理解】线性回归

线性回归(Linear regression)是利用 **回归方程(函数)** 对 **一个或多个自变量(特征值)和因变量(目标值)之间** 关系进行建模的一种分析方式。

{% asset_img image-20230901102250602.png "Hexo 博客封面示例" %}

{% asset_img image-20230901102402944.png "Hexo 博客封面示例" %}



注意事项：

1 为什么叫线性模型？因为求解的w，都是w的零次幂（常数项）所以叫成线性模型

2 在线性回归中，从数据中获取的规律其实就是学习权重系数w

3 某一个权重值w越大，说明这个权重的数据对房子价格影响越大 



### 【知道】线性回归分类

- 一元线性回归

y = kx +b 

目标值只与一个因变量有关系

{% asset_img image-20230901102857178.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-17_23-28-48.png "Hexo 博客封面示例" %}

- 多元线性回归

{% asset_img image-20230901102940614.png "Hexo 博客封面示例" %}

目标值只与多个因变量有关系

{% asset_img image-20230901103000204.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-17_23-26-34.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-17_23-27-41.png "Hexo 博客封面示例" %}


### 【知道】应用场景

{% asset_img image-20230901103123601.png "Hexo 博客封面示例" %}



## 线性回归问题的求解

**学习目标：**

1.知道线性回归API的使用

2.知道损失函数是什么

3.复习导数和矩阵的相关内容

4.理解正规方程法

5.掌握梯度下降算法的内容



### 【实操】线性回归API的应用

预测播仔身高

已知数据:

{% asset_img image-20230901104147248.png "Hexo 博客封面示例" %}

 需求:播仔身高是176，请预测体重?



{% asset_img image-20230901104237860.png "Hexo 博客封面示例" %}



通过线性回归API可快速的找到一条红色直线，是怎么求解的呢？

{% asset_img image-20230901104626001.png "Hexo 博客封面示例" %}

课堂代码：
```python
"""
线性回归介绍(Linear Regressor):
    概述/目的:
        用线性公式 来描述 多个自变量(特征)  和 1个因变量(标签)之间 关系的, 对其关系进行建模, 基于特征 预测 标签.
        线性回归属于: 有监督学习, 即: 有特征, 有标签, 且标签是连续的.
    分类:
        一元线性回归:  1个特征列 + 1个标签列
        多元线性回归:  多个特征列 + 1个标签列
    公式:
        一元线性回归:
            y = kx + b => wx + b
                k: 数学中叫斜率, 在机器学习中Weight(权重), 简称: w
                b: 数学中叫截距, 在机器学习中Bias(偏置), 简称: b
        多元线性回归:
            y = w1x1 + w2x2 + w3x3 + ... + wnxn + b
              = w的转置 * x + b

    误差 = 预测值 - 真实值
    损失函数(Loss Function, 也叫成本函数, 代价函数, 目标函数, Cost Function):
        用于描述 每个样本点 和 其预测值之间关系的, 让损失函数最小, 就是让 误差和小, 线性回归效率, 评估就越高.
    问题: 如何让损失函数最小?
    答案:
        思路1: 正规方程法.
        思路2: 梯度下降法.

    损失函数分类:
        最小二乘: 每个样本点误差的平方和
        MSE(Mean Square Error, 均方误差): 每个样本点误差的平方和 / 样本个数
        RMSE(Root Mean Square Error, 均方根误差): 均方误差 开平方根
        MAE(Mean Absolute Error, 均绝对误差): 每个样本点误差的绝对值和 / 样本个数

矩阵相关:
    1范数 = 向量中各元素 绝对值 之和.
    2范数 = 向量的模长, 即: 各个元素平方总和, 开平方根
"""
# 导包
from sklearn.linear_model import LinearRegression

# 案例: 演示线性回归API入门.

# 1. 准备数据.
x_train = [[160], [166], [172], [174], [180]]       # 训练集的特征
y_train = [56.3, 60.6, 65.1, 68.5, 75]              # 训练集的标签
x_test = [[176]]                                    # 测试集的特征

# 2. 数据的预处理, 这里不需要.
# 3. 特征工程(特征提取, 特征预处理), 这里不需要.

# 4. 模型训练
# 4.1 创建模型对象.
estimator = LinearRegression()
# 4.2 具体的训练动作.
estimator.fit(x_train, y_train)
# 4.3 因为是线性回归模型, 我们可以查看下: 斜率(w, 权重), 截距(b, 偏置)
print(f'权重: {estimator.coef_}')         # 0.92942177
print(f'偏置: {estimator.intercept_}')    # -93.27346938775514

# 5. 模型预测.
y_pre = estimator.predict(x_test)
print(f'预测值为: {y_pre}')               # 70.3047619

# 6. 模型评估.

```

### 【掌握】损失函数

需要设置一个评判标准

{% asset_img image-20230901110253313.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_09-58-16.png "Hexo 博客封面示例" %}

损失函数最小：

{% asset_img Snipaste_2026-10-18_10-01-12.png "Hexo 博客封面示例" %}

 **误差概念**：用预测值y – 真实值y就是误差

**损失函数**：衡量每个样本预测值与真实值效果的函数

“红色直线能更好的拟合所有点”也就是误差最小，误差和最小

损失函数数学如何表达呢？又如何求损失函数的最小值呢？

{% asset_img image-20230901110606206.png "Hexo 博客封面示例" %}

{% asset_img image-20230901110842936.png "Hexo 博客封面示例" %}

{% asset_img image-20230901110853094.png "Hexo 博客封面示例" %}

当损失函数取最小值时，得到k就是最优解

{% asset_img image-20230901113810862.png "Hexo 博客封面示例" %}

{% asset_img image-20230901113822728.png "Hexo 博客封面示例" %}

{% asset_img image-20230901113834036.png "Hexo 博客封面示例" %}

想求一条直线更好的拟合所有点 y = kx + b 

- ​     引入损失函数(衡量预测值和真实值效果) Loss(k, b) 

- ​     通过一个优化方法，求损失函数最小值，得到K最优解

{% asset_img image-20230912103729099.png "Hexo 博客封面示例" %}

回归的损失函数：

- *均方误差* *(*Mean-Square Error, MSE*)*

{% asset_img image-20230901114527268.png "Hexo 博客封面示例" %}

- *平均绝对误差* *(**Mean Absolute Error* *,* *MAE**)*

  {% asset_img image-20230901114539398.png "Hexo 博客封面示例" %}



{% asset_img image-20230901114816872.png "Hexo 博客封面示例" %}





### 【复习】导数和矩阵

#### 【知道】常见的数据表述

- 为什么要学习标量、向量、矩阵、张量?
  - 因机器学习、深度学习中经常用，不要因是数学就害怕
  - 宗旨:用到就学什么，不要盲目的展开、大篇幅学数学
- 标量scalar :一个独立存在的数，只有大小没有方向
- 向量vector :向量指一列顺序排列的元素。默认是列向量

{% asset_img image-20230901115201758.png "Hexo 博客封面示例" %}

{% asset_img image-20230901115210623.png "Hexo 博客封面示例" %}

- 矩阵matrix :二维数组

  {% asset_img image-20230901115223778.png "Hexo 博客封面示例" %}

  {% asset_img image-20230901115232323.png "Hexo 博客封面示例" %}

- 张量Tensor :多维数组，张量是基于向量和矩阵的推广

  {% asset_img image-20230901115246275.png "Hexo 博客封面示例" %}

#### 【知道】导数

当函数y=f（x）的自变量x在一点$x_0$上产生一个增量Δx时，函数输出值的增量Δy与自变量增量Δx的比值在Δx趋于0时的极限a如果存在，a即为在$x_0$处的导数，记作$f^\prime(x_0)$或df($x_0$)/dx。

{% asset_img 导数 "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_11-14-22.png "Hexo 博客封面示例" %}

导数公式：

{% asset_img Snipaste_2026-10-18_11-15-45.png"Hexo 博客封面示例" %}

导数是函数的局部性质。一个函数在某一点的导数描述了这个函数在这一点附近的变化率。

函数在某一点的导数就是该函数所代表的曲线在这一点上的切线斜率

常见函数的导数：


|                        公式                        |                             例子                             |
| :------------------------------------------------: | :----------------------------------------------------------: |
|                   $(C)^\prime=0$                   | $\left(5\right)^\prime=0$         $\left(10\right)^\prime=0$ |
| $\left(x^\alpha\right)^\prime=\alpha x^{\alpha-1}$ | $\left(x^3\right)^\prime=3 x^{2}$      $\left(x^5\right)^\prime=5 x^{4}$ |
|       $\left(a^x\right)^\prime=a^{x}\ln{a}$        | $\left(2^x\right)^\prime=2^x\ln{2}$      $\left(7^x\right)^\prime=7^x\ln{7}$ |
|          $\left(e^x\right)^\prime=e^{x}$           |               $\left(e^x\right)^\prime=e^{x}$                |
| $\left(\log{_a}x\right)^\prime=\frac{1}{x\ln{a}}$  | $\left(\log{_{10}}x\right)^\prime=\frac{1}{x\ln{10}}$      $\left(\log{_{6}}x\right)^\prime=\frac{1}{x\ln{6}}$ |
|      $\left(\ln{x}\right)^\prime=\frac{1}{x}$      |           $\left(\ln{x}\right)^\prime=\frac{1}{x}$           |
|       $\left(\sin{x}\right)^\prime=\cos{x}$        |            $\left(\sin{x}\right)^\prime=\cos{x}$             |
|       $\left(\cos{x}\right)^\prime=-\sin{x}$       |            $\left(\cos{x}\right)^\prime=-\sin{x}$            |

导数的四则运算：

|                             公式                             |                             例子                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| $\left[u(x)\pm v(x)\right]^\prime=u^\prime(x) \pm v^\prime(x)$ | $(e^x+4\ln{x})^\prime=(e^x)^\prime+(4\ln{x})^\prime=e^x+\frac{4}{x}$ |
| $\left[u(x)\cdot v(x)\right]^\prime=u^\prime(x) \cdot v(x) + u(x) \cdot v^\prime(x)$ | $(\sin{x}\cdot\ln{x})^\prime=\cos{x}\cdot\ln{x}+\sin{x}\cdot\frac{1}{x}$ |
| $\left[\frac{u(x)}{v(x)}\right]^\prime=\frac{u^\prime(x) \cdot v(x) - u(x) \cdot v^\prime(x)}{v^2(x)}$ | $\left(\frac{e^x}{\cos{x}}\right)^\prime=\frac{e^x\cdot\cos{x}-e^x\cdot(-\sin{x})}{cos^2(x)}$ |
|         $\{g[h(x)]\}^\prime=g^\prime(h)*h^\prime(x)$         |    $(\sin{2x})^\prime=\cos{2x}\cdot(2x)^\prime=2\cos(2x)$    |

复合函数求导：g(h)是外函数 h(x)是内函数。先对外函数求导，再对内函数求导

{% asset_img image-20230901143204713.png "Hexo 博客封面示例" %}



导数求极值：

导数为0的位置是函数的极值点 

{% asset_img image-20230901143529603.png "Hexo 博客封面示例" %}

#### 【知道】偏导

{% asset_img image-20230901144053678.png "Hexo 博客封面示例" %}

partial z / partial x

{% asset_img image-20230901144102962.png "Hexo 博客封面示例" %}

练习：

{% asset_img Snipaste_2026-10-18_11-29-43.png "Hexo 博客封面示例" %}

#### 【知道】向量

向量运算：
可以把向量看成seris对象

{% asset_img image-20230901144711751.png "Hexo 博客封面示例" %}

{% asset_img image-20230901145009232.png "Hexo 博客封面示例" %}

#### 【知道】矩阵

x就是特征列矩阵

DataFrame也是一种张量

{% asset_img image-20230901145231388.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_11-12-46.png "Hexo 博客封面示例" %}

{% asset_img image-20230901145322438.png "Hexo 博客封面示例" %}

{% asset_img image-20230901145840373.png "Hexo 博客封面示例" %}

{% asset_img image-20230901150113602.png "Hexo 博客封面示例" %}

方阵，对称方阵：
{% asset_img Snipaste_2026-10-18_12-05-55.png "Hexo 博客封面示例" %}

单位矩阵：

{% asset_img Snipaste_2026-10-18_12-08-08.png "Hexo 博客封面示例" %}

矩阵转置：

{% asset_img Snipaste_2026-10-18_12-09-15.png "Hexo 博客封面示例" %}

#### 范数
{% asset_img Snipaste_2026-10-18_11-41-51.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_11-43-33.png "Hexo 博客封面示例" %}

#### 总结
{% asset_img Snipaste_2026-10-18_12-11-04.png "Hexo 博客封面示例" %}

#### 练习
{% asset_img Snipaste_2026-10-18_12-18-38.png "Hexo 博客封面示例" %}

### 【了解】一元线性回归的解析解

{% asset_img image-20230901150406897 "Hexo 博客封面示例" %}

矩阵与单位矩阵相乘


{% asset_img image-20230901150708528.png "Hexo 博客封面示例" %}

{% asset_img image-20230912145402703.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_13-56-56.png "Hexo 博客封面示例" %}

{% asset_imgSnipaste_2026-10-18_14-03-54.png "Hexo 博客封面示例" %}


### 【了解】多元线性回归的解析解-正规方程法

{% asset_img image-20230901152528805.png "Hexo 博客封面示例" %}

{% asset_img image-20230911234116911.png "Hexo 博客封面示例" %}


每个特征都有自己的权重
{% asset_img Snipaste_2026-10-18_15-06-16.png "Hexo 博客封面示例" %}

j当前样本的列，i当前样本的行，d是总列数（特征个数），m（n）是总行数（样本数）

每个样本都在执行相同的操作，故可以累加：
{% asset_img Snipaste_2026-10-18_15-09-14.png "Hexo 博客封面示例" %}


x的转置 * x = l2范数的平方
{% asset_img Snipaste_2026-10-18_14-32-06.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_14-58-10.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_15-02-10.png "Hexo 博客封面示例" %}

{% asset_img image-20230901152745308.png "Hexo 博客封面示例" %}

{% asset_img image-20230901152758396.png "Hexo 博客封面示例" %}

正规方程问题：
1. 矩阵要有逆，否则无解
2. 运力量过大，会造成内存溢出

**难点**：转置的原因

{% asset_img Snipaste_2026-10-18_15-47-13.png "Hexo 博客封面示例" %}

但是这时又有一个问题，这么说的话，不应该转置w吗？
处理一个样本当然没问题，可是我们有很多个样本，

一个样本：

{% asset_img Snipaste_2026-10-18_15-51-55.png "Hexo 博客封面示例" %}

多个样本：pandas就不会将这个特征矩阵X给变成**列展示数据**的样子，保留原样，此时不用对w进行转置即可做矩阵乘法

{% asset_img Snipaste_2026-10-18_15-58-35.png "Hexo 博客封面示例" %}

本质上来说特征是一个一维张量，长度为n，也可以说特征是长度为3的张量

### 梯度下降算法

梯度就是一个矢量

#### 【掌握】梯度下降算法思想

当前点 - 学习率 * 损失函数

什么是梯度下降法

• 求解函数极值还有更通用的方法就是梯度下降法。顾名思义:沿着梯度下降的方向求解极小值 • 举个例子:坡度最陡下山法

{% asset_img image-20230901183007785.png "Hexo 博客封面示例" %}

- 输入:初始化位置S;每步距离为a 。输出:从位置S到达山底
- 步骤1:令初始化位置为山的任意位置S
- 步骤2:在当前位置环顾四周，如果四周都比S高返回S;否则执行步骤3
- 步骤3: 在当前位置环顾四周，寻找坡度最陡的方向，令其为x方向
- 步骤4:沿着x方向往下走，长度为a，到达新的位置S‘
- 步骤5:在S‘位置环顾四周，如果四周都比S‘高，则返回S‘。否则转到步骤3

小结:通过循环迭代的方法不断更新位置S (相当于不断更新权重参数w)

{% asset_img image-20230901183020726.png "Hexo 博客封面示例" %}

 最终找到最优解 这个方法可用来求损失函数最优解， 比正规方程更通用

```
梯度下降过程就和下山场景类似
可微分的损失函数，代表着一座山
寻找的函数的最小值，也就是山底
```

{% asset_img image-20230901183113930.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_16-21-49.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183125661.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_16-28-11.png "Hexo 博客封面示例" %}

{% asset_img image-20230912163031832.png "Hexo 博客封面示例" %}

多变量梯度下降：

θ1与θ2不一样，注意求偏导

{% asset_img image-20230901183139728.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183139728.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183139728.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183152966.png "Hexo 博客封面示例" %}







#### 【理解】银行信贷案例

有无关信息，进行特征映射，名字全部映射为1

{% asset_img image-20230901183222050.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183240178.png "Hexo 博客封面示例" %}

均方误差公式前面多除一个2是为了平衡导数，方便计算

{% asset_img Snipaste_2026-10-18_19-46-36.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_19-49-27.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_20-07-08.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183301618.png "Hexo 博客封面示例" %}



{% asset_img image-20230901183315009.png "Hexo 博客封面示例" %}





#### 【了解】梯度下降算法分类
GD（Gradient Descent）

{% asset_img image-20230901183333299.png "Hexo 博客封面示例" %}

{% asset_img image-20230901183346878.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_21-07-34.png "Hexo 博客封面示例" %}





#### 【理解】正规方程和梯度下降算法的对比

{% asset_img image-20230901162716376.png "Hexo 博客封面示例" %}

## 回归评估方法

**学习目标：**

1.掌握常用的回归评估方法

2.了解不同评估方法的特点



**为什么要进行线性回归模型的评估**

我们希望衡量预测值和真实值之间的差距，

会用到MAE、MSE、RMSE多种测评函数进行评价

### 【知道】平均绝对误差

**Mean Absolute Error (MAE)**

{% asset_img mae "Hexo 博客封面示例" %}

- 上面的公式中：n 为样本数量, y 为实际值, $\hat{y}$ 为预测值

- MAE 越小模型预测约准确

Sklearn 中MAE的API

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_predict)
```

### 【知道】均方误差

   **Mean Squared Error (MSE)**

{% asset_img mse "Hexo 博客封面示例" %}

- 上面的公式中：n 为样本数量, y 为实际值, $\hat{y}$ 为预测值
- MSE 越小模型预测约准确

Sklearn 中MSE的API

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_predict)
```

### 【知道】 均方根误差

**Root Mean Squared Error (RMSE)**

{% asset_img rmse "Hexo 博客封面示例" %}

- 上面的公式中：n 为样本数量, y 为实际值, $\hat{y}$ 为预测值
- RMSE 越小模型预测约准确

### 【了解】 三种指标的比较

我们绘制了一条直线 **y = 2x +5** 用来拟合 **y = 2x + 5 + e.** 这些数据点，其中e为噪声

{% asset_img rmse2 "Hexo 博客封面示例" %}

从上图中我们发现 MAE 和 RMSE 非常接近，都表明模型的误差很低（MAE 或 RMSE 越小，误差越小！）。 但是MAE 和 RMSE 有什么区别？为什么MAE较低？

- 对比MAE 和 RMSE的公式，RMSE的计算公式中有一个平方项，因此：大的误差将被平方，因此会增加 RMSE 的值

- 可以得出结论，RMSE 会放大预测误差较大的样本对结果的影响，而 MAE 只是给出了平均误差

- 由于 RMSE 对误差的 **平方和求平均** 再开根号，大多数情况下RMSE>MAE

  举例 (1+3)/2 = 2   $\sqrt{(1^2+3^2)/2 }= \sqrt{10/2} = \sqrt{5} = 2.236$

我们再看下一个例子

{% asset_img rmse3 "Hexo 博客封面示例" %}

橙色线与第一张图中的直线一样：**y = 2x +5** 

蓝色的点为： **y = y + sin(x)\*exp(x/20) + e**  其中 exp() 表示指数函数

我们看到对比第一张图，所有的指标都变大了，RMSE 几乎是 MAE 值的两倍，因为它对预测误差较大的点比较敏感

我们是否可以得出结论： RMSE是更好的指标？ 某些情况下MAE更有优势，例如：

- 假设数据中有少数异常点偏差很大，如果此时根据 RMSE 选择线性回归模型，可能会选出过拟合的模型来
- 在这种情况下，由于数据中的异常点极少，选择具有最低 MAE 的回归模型可能更合适
- 除此之外，当两个模型计算RMSE时数据量不一致，也不适合在一起比较 

**我们尚不知道w（θ）到底是多少，因此对他们进行初始化，全部设置为1**
而且上面计算出每个样本的梯度，再算平均梯度

{% asset_img Snipaste_2026-10-18_21-17-36.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_21-20-43.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-18_21-22-05.png "Hexo 博客封面示例" %}


总结：
{% asset_img Snipaste_2026-10-18_21-22-42.png "Hexo 博客封面示例" %}

## 【实操】波士顿房价预测案例

### 【知道】线性回归API

sklearn.linear_model.LinearRegression(fit_intercept=True)

- 通过正规方程优化
- 参数：fit_intercept，是否计算偏置
- 属性：LinearRegression.coef_ （回归系数） LinearRegression.intercept_（偏置）

sklearn.linear_model.SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate ='constant', eta0=0.01)

- 参数：loss（损失函数类型），fit_intercept（是否计算偏置）learning_rate （学习率）
- 属性：SGDRegressor.coef_ （回归系数）SGDRegressor.intercept_ （偏置）

{% asset_img Snipaste_2026-10-18_22-22-44.png "Hexo 博客封面示例" %}

深度学习中学习率是可以动态变换的，详见blog

### 【实操】波士顿房价预测

{% asset_img image-20230913092037241 "Hexo 博客封面示例" %}

#### 案例背景介绍

数据介绍

{% asset_img 006tNbRwly1ga8u37zooxj317g0tc7dk.png "Hexo 博客封面示例" %}

{% asset_img 006tNbRwly1ga8u39xrmlj30xo0ryk16.png "Hexo 博客封面示例" %}

> 给定的这些特征，是专家们得出的影响房价的结果属性。我们此阶段不需要自己去探究特征是否有用，只需要使用这些特征。到后面量化很多特征需要我们自己去寻找



#### 案例分析

回归当中的数据大小不一致，是否会导致结果影响较大。所以需要做标准化处理。

- 数据分割与标准化处理
- 回归预测
- 线性回归的算法效果评估



####  回归性能评估

均方误差(Mean Squared Error, MSE)评价机制：

$\Large MSE = \frac{1}{m}\sum_{i=1}^{m}(y^i-\hat{y})^2$

sklearn中的API：sklearn.metrics.mean_squared_error(y_true, y_pred)

- 均方误差回归损失
- y_true:真实值
- y_pred:预测值
- return:浮点数结果



#### 代码实现

```python
# 0.导包
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

# 1.加载数据
boston = load_boston()
# print(boston)

# 2.数据集划分
x_train,x_test,y_train,y_test =train_test_split(boston.data,boston.target,test_size=0.2,random_state=22)

# 3.标准化
process=StandardScaler()
x_train=process.fit_transform(x_train)
x_test=process.transform(x_test)

# 4.模型训练
# 4.1 实例化(正规方程)
# model =LinearRegression(fit_intercept=True)
model = SGDRegressor(learning_rate='constant',eta0=0.01)
# 4.2 fit
model.fit(x_train,y_train)

# print(model.coef_)
# print(model.intercept_)
# 5.预测
y_predict=model.predict(x_test)

print(y_predict)

# 6.模型评估

print(mean_squared_error(y_test,y_predict))


```



1.2.0 以上版本实现

```python
# 0.导包
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

# 1.加载数据
# boston = load_boston()
# print(boston)
import pandas as pd
import numpy as np


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 2.数据集划分
# x_train,x_test,y_train,y_test =train_test_split(boston.data,boston.target,test_size=0.2,random_state=22)
x_train,x_test,y_train,y_test =train_test_split(data,target,test_size=0.2,random_state=22)

# 3.标准化
process=StandardScaler()
x_train=process.fit_transform(x_train)
x_test=process.transform(x_test)

# 4.模型训练
# 4.1 实例化(正规方程)
# model =LinearRegression(fit_intercept=True)
model = SGDRegressor(learning_rate='constant',eta0=0.01)
# 4.2 fit
model.fit(x_train,y_train)

# print(model.coef_)
# print(model.intercept_)
# 5.预测
y_predict=model.predict(x_test)

print(y_predict)

# 6.模型评估

print(mean_squared_error(y_test,y_predict))


```

#### 课堂代码
```python
"""
案例:
    演示 正规方程法 线性回归对象 完成 波士顿房价预测案例.

回顾:
    线性回归算法 属于 有监督学习之  有特征, 有标签, 且标签是连续的.
    线性回归分类:
        一元线性回归: 1个特征列, 1个标签列.
        多元线性回归: 多个特征列, 1个标签列.
    线性回归大白话解释:
       它是用线性公式来描述 特征 和 标签之间关系的, 方便做预测, 公式如下:
       一元线性回归: y = w * x + b
       多元线性回归: y = w1 * x1 + w2 * x2 + w3 * x3 + ... + wn * xn + b = w的转置 * x + b
    如何衡量线性回归模型的好坏?
        思路:
            预测值和真实值之间的误差, 误差越小, 模型越好 => 损失函数
        具体的方案:
            1. 最小二乘.        每个(样本)误差平方和
            2. 均方误差(MSE)    每个(样本)误差平方和 / 样本总数
            3. 均方根误差(RMSE)  每个(样本)误差平方和 / 样本总数 的 平方根
            4. 平均绝对误差(MAE) 每个(样本)误差绝对值和 / 样本总数
    如何让损失函数最小?
        思路1: 梯度下降法.     =>   全梯度下降(Full Gradient Descent, FGD), 随机梯度下降(SGD), 小批量梯度下降(推荐, Min-Batch), 随机平均梯度下降(SAG)
        思路2: 正规方程法.

    机器学习开发流程:
        1. 加载数据.
        2. 数据的预处理.
        3. 特征工程(特征提取, 特征预处理...)
        4. 模型训练.
        5. 模型预测.
        6. 模型评估
"""

# 导包
# from sklearn.datasets import load_boston                # 数据
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估, RMSE, MAE
from sklearn.linear_model import Ridge, RidgeCV

import pandas as pd
import numpy as np

# 1. 加载 波士顿房价数据.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None) # /转义，解决报错


data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # hstack()函数作用: 水平拼接数组

target = raw_df.values[1::2, 2]
# print(f'特征: {data.shape}')      # (506, 13)
# print(f'标签: {target.shape}')    # (506,)
#
# print(f'特征数据: {data[:5]}')
# print(f'标签数据: {target[:5]}')

# 2. 数据的预处理.  切分训练集 和 测试集.
# 参1: 特征数据. 参2: 标签数据. 参3: 测试集占训练集的比例. 参4: 随机种子.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=23)

# 3. 特征工程(特征提取, 特征预处理...)
# 3.1 创建标准化对象.
transfer = StandardScaler()

# 3.2 对训练集进行标准化.
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建 线性回归 正规方程 模型对象.
estimator = LinearRegression(fit_intercept=True)    # fit_intercept: 是否需要截距(Bias, 偏置), 默认是True


# 4.2 模型训练.
estimator.fit(x_train, y_train)

# 4.3 打印模型计算出的 w(权重, weight) 和 b(偏置, bias).
print(f'权重: {estimator.coef_}')
print(f'偏置: {estimator.intercept_}')

# 5. 模型预测.
y_pre = estimator.predict(x_test)
print(f'预测结果为: {y_pre}')

# 6. 模型评估
# 参1: 测试集的标签数据. 参2: 预测结果.
print(f'均方误差为: {mean_squared_error(y_test, y_pre)}')            # MSE: 均方误差, 公式: 每个样本的误差平方和 / 样本总数

print(f'均方根误差为: {root_mean_squared_error(y_test, y_pre)}')     # RMSE: 均方根误差, 公式: 每个样本的误差平方和 / 样本总数, 开平方根

print(f'平均绝对误差为: {mean_absolute_error(y_test, y_pre)}')        # MAE: 平均绝对误差, 公式: 每个样本的误差 绝对值和 / 样本总数
```
输出示例：

{% asset_img Snipaste_2026-10-18_22-48-38.png "Hexo 博客封面示例" %}

#### 课堂代码；随机梯度下降
```python
"""
案例:
    演示 随机梯度下降法 线性回归对象 完成 波士顿房价预测案例.

回顾:
    线性回归算法 属于 有监督学习之  有特征, 有标签, 且标签是连续的.
    线性回归分类:
        一元线性回归: 1个特征列, 1个标签列.
        多元线性回归: 多个特征列, 1个标签列.
    线性回归大白话解释:
       它是用线性公式来描述 特征 和 标签之间关系的, 方便做预测, 公式如下:
       一元线性回归: y = w * x + b
       多元线性回归: y = w1 * x1 + w2 * x2 + w3 * x3 + ... + wn * xn + b = w的转置 * x + b
    如何衡量线性回归模型的好坏?
        思路:
            预测值和真实值之间的误差, 误差越小, 模型越好 => 损失函数
        具体的方案:
            1. 最小二乘.        每个(样本)误差平方和
            2. 均方误差(MSE)    每个(样本)误差平方和 / 样本总数
            3. 均方根误差(RMSE)  每个(样本)误差平方和 / 样本总数 的 平方根
            4. 平均绝对误差(MAE) 每个(样本)误差绝对值和 / 样本总数
    如何让损失函数最小?
        思路1: 梯度下降法.     =>   全梯度下降(Full Gradient Descent, FGD), 随机梯度下降(SGD), 小批量梯度下降(推荐, Min-Batch), 随机平均梯度下降(SAG)
        思路2: 正规方程法.

    机器学习开发流程:
        1. 加载数据.
        2. 数据的预处理.
        3. 特征工程(特征提取, 特征预处理...)
        4. 模型训练.
        5. 模型预测.
        6. 模型评估
"""

# 导包
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估, RMSE, MAE
from sklearn.linear_model import Ridge, RidgeCV

import pandas as pd
import numpy as np

# 1. 加载 波士顿房价数据.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # hstack()函数作用: 水平拼接数组
target = raw_df.values[1::2, 2]
# print(f'特征: {data.shape}')      # (506, 13)
# print(f'标签: {target.shape}')    # (506,)
#
# print(f'特征数据: {data[:5]}')
# print(f'标签数据: {target[:5]}')

# 2. 数据的预处理.  按照8:2 切分 训练集和测试集.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=23)

# 3. 特征工程(特征提取, 特征预处理...)
# 3.1 创建 标准化对象.
transfer = StandardScaler()
# 3.2 对训练集和测试集进行标准化处理.
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
#fit_intercept=True；是否计算截距
# estimator = LinearRegression(fit_intercept=True)    # 正规方程法 线性回归对象.

# 4.1 创建 梯度下降 线性回归 模型对象.
# 参1: fit_intercept: 是否计算截距.
# 参2: learning_rate: 学习率模式 -> 常量, 即: 不会发生改变.
# 参3: eta0: 学习率.
estimator = SGDRegressor(fit_intercept=True, learning_rate='constant', eta0=0.01)

# 4.2 模型训练.
estimator.fit(x_train, y_train)
# 4.3 打印 权重 和 偏置.
print(f'权重: {estimator.coef_}')
print(f'偏置: {estimator.intercept_}')

# 5. 模型预测.
y_pre = estimator.predict(x_test)

# 6. 模型评估
# MSE: 均方误差, 每个误差的平方和 / 样本总数
print(f'均方误差: {mean_squared_error(y_test, y_pre)}')          # 参1: 测试集的真实标签, 参2: 测试集的预测标签

# RMSE: 均方根误差, 均方误差的平方根
print(f'均方根误差: {root_mean_squared_error(y_test, y_pre)}')   # 参1: 测试集的真实标签, 参2: 测试集的预测标签

# MAE: 平均绝对误差, 每个误差绝对值和 / 样本总数
print(f'平均绝对误差: {mean_absolute_error(y_test, y_pre)}')     # 参1: 测试集的真实标签, 参2: 测试集的预测标签
```


## 正则化

**学习目标：**

1.掌握过拟合、欠拟合的概念

2.掌握过拟合、欠拟合产生的原因

3.知道什么是正则化，以及正则化的方法

### 【理解】 欠拟合与过拟合

过拟合：一个假设 **在训练数据上能够获得比其他假设更好的拟合， 但是在测试数据集上却不能很好地拟合数据** (体现在准确率下降)，此时认为这个假设出现了过拟合的现象。(模型过于复杂)

欠拟合：一个假设 **在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据** ，此时认为这个假设出现了欠拟合的现象。(模型过于简单)

过拟合和欠拟合的区别：

{% asset_img 006tNbRwly1ga8u2rlw69j315m0oc40y.png "Hexo 博客封面示例" %}

欠拟合在训练集和测试集上的误差都较大

过拟合在训练集上误差较小，而测试集上误差较大

{% asset_img image-20230913101352444.png "Hexo 博客封面示例" %}

### 【实践】通过代码认识过拟合和欠拟合

绘制数据

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # 计算均方误差
from sklearn.model_selection import train_test_split


def dm01_欠拟合():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)

    # loc: 均值, scale: 标准差, normal: 正态分布.加噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    
    # 3. 训练模型
    X = x.reshape(-1, 1)
    estimator.fit(X, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)           # 散点图
    plt.plot(x, y_predict, color='r')   # 折线图(预测值, 拟合回归线)
    plt.show()                  # 具体的绘图



if __name__ == '__main__':
    dm01_欠拟合()

```

{% asset_img 1 "Hexo 博客封面示例" %}

```python
#计算均方误差
from sklearn.metrics import mean_squared_error
mean_squared_error(y,y_predict)

#3.0750025765636577
```

添加二次项，绘制图像

```python
def dm02_模型ok():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    X2 = np.hstack([X, X ** 2])
    estimator.fit(X2, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X2)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    # plt.plot(x, y_predict)
    plt.show()  # 具体的绘图
```

{% asset_img 2.png "Hexo 博客封面示例" %}

```python
#计算均方误差和准确率

from sklearn.metrics import mean_squared_error
mean_squared_error(y,y_predict2)

#1.0987392142417858
```

再次加入高次项，绘制图像，观察均方误差结果

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # 计算均方误差
from sklearn.model_selection import train_test_split


def dm01_欠拟合():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    estimator.fit(X, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)           # 散点图
    plt.plot(x, y_predict, color='r')   # 折线图(预测值, 拟合回归线)
    plt.show()                  # 具体的绘图

def dm02_模型ok():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    X2 = np.hstack([X, X ** 2])
    estimator.fit(X2, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X2)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    # plt.plot(x, y_predict)
    plt.show()  # 具体的绘图


def dm03_过拟合():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    # hstack() 函数用于将多个数组在行上堆叠起来, 即: 数据增加高次项.
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    plt.show()  # 具体的绘图

if __name__ == '__main__':
    # dm01_欠拟合()
    # dm02_模型ok()
    dm03_过拟合()

```

#### 课堂代码
```python
"""
案例:
    演示 欠拟合, 正好拟合, 过拟合, L1正则化, L2正则化的 效果图.

回顾:
    欠拟合:    模型在训练集 和 测试集表现效果都不好.
    正好拟合:   模型在训练集 和 测试集表现效果都好.
    过拟合:    模型在训练集表现好, 测试集表现不好.

过拟合, 欠拟合解释:
    产生原因:
        欠拟合: 模型简单.
        过拟合: 模型复杂.
    解决方案:
        欠拟合: 增加特征, 从而增加 模型的复杂度.
        过拟合: 减少模型复杂度, 手动筛选(减少)特征, L1和L2正则化.

L1和L2正则化介绍:
    目的/思路:
        都是基于 惩罚系数 来修改(特征列的)权重的, 惩罚系数越大, 则修改力度就越大, 对应的权重就越小.
    区别:
        L1正则化, 可以实现让权重变为0, 从而达到 特征选择的目的.
        L2正则化, 只能让权重无限趋近于0, 但是不能为0.
    大白话:
        我要去爬山, 带了个小包, 装了: 登山杖, 水, 面包, 衣服, 雨伞, 鞋子...  发现包装不下了.
        L1正则化: 可以实现去掉一些不是必选的, 例如: 当天去, 当前回, 且天气晴朗 -> 不带雨伞, 鞋子, 即: 权重为0
        L2正则化: 换一个非常非常大的包, 还是那些物品, 但是空间占用(权重)就变小了...
"""

# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估, RMSE, MAE
from sklearn.linear_model import Lasso, Ridge           # L1正则化, L2正则化


# 1. 定义函数, 模拟: 欠拟合.
def dm01_under_fitting():
    # 1. 准备数据.
    # 1.1 指定随机种子, 则每次生成(噪声)的数据都是固定的.
    np.random.seed(23)

    # 1.2 随机生成x轴 100个数据, 模拟: 特征.
    x = np.random.uniform(-3, 3, 100)   # 参1: 最小值, 参2: 最大值, 参3: 生成个数

    # 1.3 基于x轴值, 通过线性公式, 生成y轴 100个数据, 模拟: 标签.
    # 线性公式: y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参1: 平均值, 参2: 标准差, 参3: 生成个数
    
    # 1.4 查看生成的 x轴(特征) 和 y轴(标签)的数据.
    print(f'特征(x): {x[:5]}')        # [1, 2, 3, 4, 5]
    print(f'标签(y): {y[:5]}')

    # 2. 数据预处理, 把x轴(特征)转成 多行1列的形式.
    X = x.reshape(-1, 1)
    print(f'处理后的特征: {X[:5]}')    # [[1], [2], [3], [4], [5]]

    # 3. 特征工程, 这里不做了, 直接用100条数据, 先训练, 后预测.

    # 4. 模型训练.
    # 4.1 创建模型对象.
    estimator = LinearRegression()      # 正规方程 线性回归模型.
    # 4.2 模型训练.
    estimator.fit(X, y)                 # 参1: 处理后的特征数据. 参2: 标签数据.

    # 5. 模型预测.
    y_predict = estimator.predict(X)    # 处理后的特征数据

    # 6. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')      # 参1: 真实值, 参2: 预测值.

    # 7. 绘图.
    plt.scatter(x, y)                           # 以散点图的形式绘制 真实值.
    plt.plot(x, y_predict, color='red')   # 以线图的形式绘制 预测值.
    plt.show()

# 2. 定义函数, 模拟: 正好拟合.
def dm02_just_fitting():
    # 1. 准备数据.
    # 1.1 指定随机种子, 则每次生成(噪声)的数据都是固定的.
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据, 模拟: 特征.
    x = np.random.uniform(-3, 3, 100)  # 参1: 最小值, 参2: 最大值, 参3: 生成个数
    # 1.3 基于x轴值, 通过线性公式, 生成y轴 100个数据, 模拟: 标签.
    # 线性公式: y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参1: 平均值, 参2: 标准差, 参3: 生成个数
    # 1.4 查看生成的 x轴(特征) 和 y轴(标签)的数据.
    print(f'特征(x): {x[:5]}')  # [1, 2, 3, 4, 5]
    print(f'标签(y): {y[:5]}')

    # 2. 数据预处理, 把x轴(特征)转成 多行1列的形式.
    # 2.1 把上述的x轴(特征)转成 多行1列的形式, 即: [[1], [2], [3], [4], [5]]
    X = x.reshape(-1, 1)
    print(f'处理后的特征: {X[:5]}')  # [[1], [2], [3], [4], [5]]

    # 2.2 因为目前特征列只有1列, 模型过于简单, 会出现欠拟合的问题, 我们增加1列 特征列, 从而增加模型的复杂度.
    # 即: 把数据从  [[1], [2], [3], [4], [5]] =>  [[1, 1], [2, 4], [3, 9], [4, 16], [5, 25]]
    X2 = np.hstack([X, X ** 2])  # 该函数作用: 横向拼接, 即: 拼接2个数组, 拼接后数组的行数不变, 拼接后数组的列数等于拼接前数组的列数之和.
    print(f'处理后的特征: {X2[:5]}')

    # 3. 特征工程, 这里不做了, 直接用100条数据, 先训练, 后预测.

    # 4. 模型训练.
    # 4.1 创建模型对象.
    estimator = LinearRegression()  # 正规方程 线性回归模型.
    # 4.2 模型训练.
    estimator.fit(X2, y)  # 参1: 处理后的特征数据. 参2: 标签数据.

    # 5. 模型预测.
    y_predict = estimator.predict(X2)  # 处理后的特征数据

    # 6. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')  # 参1: 真实值, 参2: 预测值.

    # 7. 绘图.
    plt.scatter(x, y)  # 以散点图的形式绘制 真实值.
    # np.sort(x): 对x轴(特征)排序, 默认是: 升序.
    # np.argsort(x): 对x轴(特征)排序, 返回排序后的索引.
    # 例如:   排序前x轴是 [11, 33, 22] -> 对应索引: [0, 1, 2]
    # 排序后: x轴是      [11, 22, 33] -> 对应索引: [0, 2, 1]
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以线图的形式绘制 预测值.
    plt.show()


# 3. 定义函数, 模拟: 过拟合.
def dm03_over_fitting():
    # 1. 准备数据.
    # 1.1 指定随机种子, 则每次生成(噪声)的数据都是固定的.
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据, 模拟: 特征.
    x = np.random.uniform(-3, 3, 100)  # 参1: 最小值, 参2: 最大值, 参3: 生成个数
    # 1.3 基于x轴值, 通过线性公式, 生成y轴 100个数据, 模拟: 标签.
    # 线性公式: y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参1: 平均值, 参2: 标准差, 参3: 生成个数
    # 1.4 查看生成的 x轴(特征) 和 y轴(标签)的数据.
    print(f'特征(x): {x[:5]}')  # [1, 2, 3, 4, 5]
    print(f'标签(y): {y[:5]}')

    # 2. 数据预处理, 把x轴(特征)转成 多行1列的形式.
    # 2.1 把上述的x轴(特征)转成 多行1列的形式, 即: [[1], [2], [3], [4], [5]]
    X = x.reshape(-1, 1)
    print(f'处理后的特征: {X[:5]}')  # [[1], [2], [3], [4], [5]]

    # 2.2 因为目前特征列只有1列, 模型过于简单, 为了模拟过拟合, 我们增加9列 特征列, 从而增加模型的复杂度.
    # 即: 把数据从  [[1], [2], [3], [4], [5]] =>  [[1, 1**2, 1**3, 1**4, 1**5...], [2, 2**2, 2**3, 2**4, 2**5], [3...]...]
    # 该函数作用: 横向拼接, 即: 拼接多个数组, 拼接后数组的行数不变, 拼接后数组的列数等于拼接前数组的列数之和.
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'处理后的特征: {X3[:5]}')

    # 3. 特征工程, 这里不做了, 直接用100条数据, 先训练, 后预测.

    # 4. 模型训练.
    # 4.1 创建模型对象.
    estimator = LinearRegression()  # 正规方程 线性回归模型.
    # 4.2 模型训练.
    estimator.fit(X3, y)  # 参1: 处理后的特征数据. 参2: 标签数据.

    # 5. 模型预测.
    y_predict = estimator.predict(X3)  # 处理后的特征数据

    # 6. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')  # 参1: 真实值, 参2: 预测值.

    # 7. 绘图.
    plt.scatter(x, y)  # 以散点图的形式绘制 真实值.
    # np.sort(x): 对x轴(特征)排序, 默认是: 升序.
    # np.argsort(x): 对x轴(特征)排序, 返回排序后的索引.
    # 例如:   排序前x轴是 [11, 33, 22] -> 对应索引: [0, 1, 2]
    # 排序后: x轴是      [11, 22, 33] -> 对应索引: [0, 2, 1]
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以线图的形式绘制 预测值.
    plt.show()

# 4. 定义函数, 模拟: L1正则化.
def dm04_l1_regularization():
    # 1. 准备数据.
    # 1.1 指定随机种子, 则每次生成(噪声)的数据都是固定的.
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据, 模拟: 特征.
    x = np.random.uniform(-3, 3, 100)  # 参1: 最小值, 参2: 最大值, 参3: 生成个数
    # 1.3 基于x轴值, 通过线性公式, 生成y轴 100个数据, 模拟: 标签.
    # 线性公式: y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参1: 平均值, 参2: 标准差, 参3: 生成个数
    # 1.4 查看生成的 x轴(特征) 和 y轴(标签)的数据.
    print(f'特征(x): {x[:5]}')  # [1, 2, 3, 4, 5]
    print(f'标签(y): {y[:5]}')

    # 2. 数据预处理, 把x轴(特征)转成 多行1列的形式.
    # 2.1 把上述的x轴(特征)转成 多行1列的形式, 即: [[1], [2], [3], [4], [5]]
    X = x.reshape(-1, 1)
    print(f'处理后的特征: {X[:5]}')  # [[1], [2], [3], [4], [5]]

    # 2.2 因为目前特征列只有1列, 模型过于简单, 为了模拟过拟合, 我们增加9列 特征列, 从而增加模型的复杂度.
    # 即: 把数据从  [[1], [2], [3], [4], [5]] =>  [[1, 1**2, 1**3, 1**4, 1**5...], [2, 2**2, 2**3, 2**4, 2**5], [3...]...]
    # 该函数作用: 横向拼接, 即: 拼接多个数组, 拼接后数组的行数不变, 拼接后数组的列数等于拼接前数组的列数之和.
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'处理后的特征: {X3[:5]}')

    # 3. 特征工程, 这里不做了, 直接用100条数据, 先训练, 后预测.

    # 4. 模型训练.
    # 4.1 创建模型对象.
    # estimator = LinearRegression()  # 正规方程 线性回归模型.
    # 改为创建 L1正则化对象.
    estimator = Lasso(alpha=0.1)    # alpha: 正则化系数(惩罚系数), 默认是: 1.

    # 4.2 模型训练.
    estimator.fit(X3, y)  # 参1: 处理后的特征数据. 参2: 标签数据.

    # 5. 模型预测.
    y_predict = estimator.predict(X3)  # 处理后的特征数据

    # 6. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')  # 参1: 真实值, 参2: 预测值.

    # 7. 绘图.
    plt.scatter(x, y)  # 以散点图的形式绘制 真实值.
    # np.sort(x): 对x轴(特征)排序, 默认是: 升序.
    # np.argsort(x): 对x轴(特征)排序, 返回排序后的索引.
    # 例如:   排序前x轴是 [11, 33, 22] -> 对应索引: [0, 1, 2]
    # 排序后: x轴是      [11, 22, 33] -> 对应索引: [0, 2, 1]
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以线图的形式绘制 预测值.
    plt.show()

# 5. 定义函数, 模拟: L2正则化.
def dm05_l2_regularization():
    # 1. 准备数据.
    # 1.1 指定随机种子, 则每次生成(噪声)的数据都是固定的.
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据, 模拟: 特征.
    x = np.random.uniform(-3, 3, 100)  # 参1: 最小值, 参2: 最大值, 参3: 生成个数
    # 1.3 基于x轴值, 通过线性公式, 生成y轴 100个数据, 模拟: 标签.
    # 线性公式: y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)  # 参1: 平均值, 参2: 标准差, 参3: 生成个数
    # 1.4 查看生成的 x轴(特征) 和 y轴(标签)的数据.
    print(f'特征(x): {x[:5]}')  # [1, 2, 3, 4, 5]
    print(f'标签(y): {y[:5]}')

    # 2. 数据预处理, 把x轴(特征)转成 多行1列的形式.
    # 2.1 把上述的x轴(特征)转成 多行1列的形式, 即: [[1], [2], [3], [4], [5]]
    X = x.reshape(-1, 1)
    print(f'处理后的特征: {X[:5]}')  # [[1], [2], [3], [4], [5]]

    # 2.2 因为目前特征列只有1列, 模型过于简单, 为了模拟过拟合, 我们增加9列 特征列, 从而增加模型的复杂度.
    # 即: 把数据从  [[1], [2], [3], [4], [5]] =>  [[1, 1**2, 1**3, 1**4, 1**5...], [2, 2**2, 2**3, 2**4, 2**5], [3...]...]
    # 该函数作用: 横向拼接, 即: 拼接多个数组, 拼接后数组的行数不变, 拼接后数组的列数等于拼接前数组的列数之和.
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    print(f'处理后的特征: {X3[:5]}')

    # 3. 特征工程, 这里不做了, 直接用100条数据, 先训练, 后预测.

    # 4. 模型训练.
    # 4.1 创建模型对象.
    # estimator = LinearRegression()  # 正规方程 线性回归模型.

    # 改为创建 L1正则化对象.
    # estimator = Lasso(alpha=0.1)    # alpha: 正则化系数(惩罚系数), 默认是: 1.

    # 改为创建 L2正则化对象.
    estimator = Ridge(alpha=10)

    # 4.2 模型训练.
    estimator.fit(X3, y)  # 参1: 处理后的特征数据. 参2: 标签数据.

    # 5. 模型预测.
    y_predict = estimator.predict(X3)  # 处理后的特征数据

    # 6. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')  # 参1: 真实值, 参2: 预测值.

    # 7. 绘图.
    plt.scatter(x, y)  # 以散点图的形式绘制 真实值.
    # np.sort(x): 对x轴(特征)排序, 默认是: 升序.
    # np.argsort(x): 对x轴(特征)排序, 返回排序后的索引.
    # 例如:   排序前x轴是 [11, 33, 22] -> 对应索引: [0, 1, 2]
    # 排序后: x轴是      [11, 22, 33] -> 对应索引: [0, 2, 1]
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')  # 以线图的形式绘制 预测值.
    plt.show()


# 6. 测试
if __name__ == '__main__':
    # dm01_under_fitting()
    # dm02_just_fitting()
    # dm03_over_fitting()
    # dm04_l1_regularization()
    dm05_l2_regularization()
```

{% asset_img 3.png "Hexo 博客封面示例" %}

通过上述观察发现，随着加入的高次项越来越多，拟合程度越来越高，均方误差也随着加入越来越小。说明已经不再欠拟合了。

问题：如何判断出现过拟合呢？

将数据集进行划分：对比X、X2、X5的测试集的均方误差

X的测试集均方误差

```python
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 5)
estimator = LinearRegression()
estimator.fit(X_train,y_train)
y_predict = estimator.predict(X_test)

mean_squared_error(y_test,y_predict)
#3.153139806483088
```

X2的测试集均方误差

```python
X_train,X_test,y_train,y_test = train_test_split(X2,y,random_state = 5)
estimator = LinearRegression()
estimator.fit(X_train,y_train)
y_predict = estimator.predict(X_test)
mean_squared_error(y_test,y_predict)
#1.111873885731967
```

X5的测试集的均方误差

```python
X_train,X_test,y_train,y_test = train_test_split(X5,y,random_state = 5)
estimator = LinearRegression()
estimator.fit(X_train,y_train)
y_predict = estimator.predict(X_test)
mean_squared_error(y_test,y_predict)
#1.4145580542309835
```

### 【理解】 原因以及解决办法

**欠拟合产生原因：** 学习到数据的特征过少

解决办法：

**1）添加其他特征项**，有时出现欠拟合是因为特征项不够导致的，可以添加其他特征项来解决

**2）添加多项式特征**，模型过于简单时的常用套路，例如将线性模型通过添加二次项或三次项使模型泛化能力更强

**过拟合产生原因：** 原始特征过多，存在一些嘈杂特征， 模型过于复杂是因为模型尝试去兼顾所有测试样本

解决办法：

1）重新清洗数据，导致过拟合的一个原因有可能是数据不纯，如果出现了过拟合就需要重新清洗数据。

2）增大数据的训练量，还有一个原因就是我们用于训练的数据量太小导致的，训练数据占总数据的比例过小。

**3）正则化**

4）减少特征维度

### 【理解】正则化

在解决回归过拟合中，我们选择正则化。但是对于其他机器学习算法如分类算法来说也会出现这样的问题，除了一些算法本身作用之外（决策树、神经网络），我们更多的也是去自己做特征选择，包括之前说的删除、合并一些特征

{% asset_img 006tNbRwly1ga8u2sjcw9j314o0g8wkd.png "Hexo 博客封面示例" %}

**如何解决？**

{% asset_img 006tNbRwly1ga8u2tduvuj30zs0kctav.png "Hexo 博客封面示例" %}

**在学习的时候，数据提供的特征有些影响模型复杂度或者这个特征的数据点异常较多，所以算法在学习的时候尽量减少这个特征的影响（甚至删除某个特征的影响），这就是正则化**

注：调整时候，算法并不知道某个特征影响，而是去调整参数得出优化的结

#### **L1正则化**

- 假设𝐿(𝑊)是未加正则项的损失，𝜆是一个超参，控制正则化项的大小。
- 则最终的损失函数：$𝐿=𝐿(𝑊)+ \lambda*\sum_{i=1}^{n}\lvert w_i\rvert$ 

作用：用来进行特征选择，主要原因在于L1正则化会使得较多的参数为0，从而产生稀疏解,可以将0对应的特征遗弃，进而用来选择特征。一定程度上L1正则也可以防止模型过拟合。

{% asset_img image-20230913145042318 "Hexo 博客封面示例" %}

**L1正则为什么可以产生稀疏解（可以特征选择）**

稀疏性：向量中很多维度值为0

- 对其中的一个参数 $ w_i $ 计算梯度，其他参数同理，α是学习率，sign(wi)是符号函数。

{% asset_img l2_4 "Hexo 博客封面示例" %}

L1的梯度：

$𝐿=𝐿(𝑊)+ \lambda*\sum_{i=1}^{n}\lvert w_i\rvert$ 

$\frac{\partial L}{\partial w_{i}} = \frac{\partial L(W)}{\partial w_{i}}+\lambda sign(w_{i})$

LASSO回归: from sklearn.linear_model import Lasso



#### **L2正则化**

- 假设𝐿(𝑊)是未加正则项的损失，𝜆是一个超参，控制正则化项的大小。
- 则最终的损失函数：$𝐿=𝐿(𝑊)+ \lambda*\sum_{i=1}^{n}w_{i}^{2}$ 

作用：主要用来防止模型过拟合，可以减小特征的权重

优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象

Ridge回归: from sklearn.linear_model import Ridge





#### **正则化案例**

```python
X10 = np.hstack([X2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10]) 
estimator3 = LinearRegression() 
estimator3.fit(X10,y) 
y_predict3 = estimator3.predict(X10) 

plt.scatter(x,y) 
plt.plot(np.sort(x),y_predict3[np.argsort(x)],color = 'r') 
plt.show()

estimator3.coef_

array([ 1.32292089e+00,  2.03952017e+00, -2.88731664e-01, -1.24760429e+00,
        8.06147066e-02,  3.72878513e-01, -7.75395040e-03, -4.64121137e-02,
        1.84873446e-04,  2.03845917e-03])
```

{% asset_img l2_5 "Hexo 博客封面示例" %}

```python
from sklearn.linear_model import Lasso  # L1正则
from sklearn.linear_model import Ridge  # 岭回归 L2正则

def dm04_模型过拟合_L1正则化():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化L1正则化模型, 做实验: alpha惩罚力度越来越大, k值越来越小.
    estimator = Lasso(alpha=0.005)
    # 3. 训练模型
    X = x.reshape(-1, 1)
    # hstack() 函数用于将多个数组在行上堆叠起来, 即: 数据增加高次项.
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)
    print(f'权重: {estimator.coef_}')

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    plt.show()  # 具体的绘图
```

{% asset_img l2_6 "Hexo 博客封面示例" %}

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error  # 计算均方误差
from sklearn.model_selection import train_test_split


def dm01_欠拟合():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    estimator.fit(X, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)           # 散点图
    plt.plot(x, y_predict, color='r')   # 折线图(预测值, 拟合回归线)
    plt.show()                  # 具体的绘图

def dm02_模型ok():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    X2 = np.hstack([X, X ** 2])
    estimator.fit(X2, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X2)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    # plt.plot(x, y_predict)
    plt.show()  # 具体的绘图

def dm03_过拟合():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化 线性回归模型.
    estimator = LinearRegression()
    # 3. 训练模型
    X = x.reshape(-1, 1)
    # hstack() 函数用于将多个数组在行上堆叠起来, 即: 数据增加高次项.
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    plt.show()  # 具体的绘图


def dm04_模型过拟合_L1正则化():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化L1正则化模型, 做实验: alpha惩罚力度越来越大, k值越来越小.
    estimator = Lasso(alpha=0.005)
    # 3. 训练模型
    X = x.reshape(-1, 1)
    # hstack() 函数用于将多个数组在行上堆叠起来, 即: 数据增加高次项.
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)
    print(f'权重: {estimator.coef_}')

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    plt.show()  # 具体的绘图

def dm05_模型过拟合_L2正则化():
    # 1. 准备x, y数据, 增加上噪声.
    # 用于设置随机数生成器的种子（seed）, 种子一样, 每次生成相同序列.
    np.random.seed(666)
    # x: 随机数, 范围为 (-3, 3), 100个.
    x = np.random.uniform(-3, 3, size=100)
    # loc: 均值, scale: 标准差, normal: 正态分布.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 实例化L2正则化模型, 做实验: alpha惩罚力度越来越大, k值越来越小.
    estimator = Ridge(alpha=0.005)
    # 3. 训练模型
    X = x.reshape(-1, 1)
    # hstack() 函数用于将多个数组在行上堆叠起来, 即: 数据增加高次项.
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)
    print(f'权重: {estimator.coef_}')

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print("预测值:", y_predict)

    # 5. 计算均方误差 => 模型评估
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    # 6. 画图
    plt.scatter(x, y)  # 散点图
    # sort()  该函数直接返回一个排序后的新数组。
    # numpy.argsort()   该函数返回的是数组值从小到大排序时对应的索引值
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')  # 折线图(预测值, 拟合回归线)
    plt.show()  # 具体的绘图

if __name__ == '__main__':
    # dm01_欠拟合()
    # dm02_模型ok()
    dm03_过拟合()
    # dm04_模型过拟合_L1正则化()
    # dm05_模型过拟合_L2正则化()

```

{% asset_img l2_7.png "Hexo 博客封面示例" %}









## 作业

1.完成线性回归部分的思维导图



2.描述梯度下降算法思想，自己推导银行信贷的案例





3.说明欠拟合和过拟合的相关内容



3.使用L1和L2正则化方法实现波士顿房价预测