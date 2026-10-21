---
title: 逻辑回归
date: 2026-10-15 12:35:29
tags:
---
# 逻辑回归

{% asset_img Snipaste_2026-10-19_10-45-14.png "Hexo 博客封面示例" %}


## 逻辑回归简介

**学习目标：**

1.知道逻辑回归的应用场景

2.复习逻辑回归应用到的数学知识

### 【了解】应用场景

{% asset_img image-20230904115156371.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-19_10-50-41.png "Hexo 博客封面示例" %}

逻辑回归是解决二分类问题的利器

### 【熟悉】数学知识

#### 【知道】sigmoid函数

{% asset_img image-20230913152535607.png "Hexo 博客封面示例" %}

{% asset_img image-20230904144434180.png "Hexo 博客封面示例" %}

{% asset_img image-20230904144454316.png "Hexo 博客封面示例" %}



#### 【理解】概率

{% asset_img image-20230904144658969.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-19_10-57-10.png "Hexo 博客封面示例" %}


#### 【理解】极大似然估计

核心思想：根据观测到的结果来估计模型算法中的未知参数

设模型中含有待估参数w，可以取很多值。已经知道了样本观测值，从w的一切可能值中（选出一个使该观察值出现的概率为最大的值，作为w参数的估计值，这就是极大似然估计。（顾名思义：就是看上去那个是最大可能的意思）

举个例子：
联合概率

假设有一枚不均匀的硬币，出现正面的概率和反面的概率是不同的。假定出现正面的概率为𝜃， 抛了6次得到如下现象 D = {正面，反面，反面，正面，正面，正面}。每次投掷事件都是相互独立的。 则根据产生的现象D，来估计参数𝜃是多少?

```
P(D|𝜃) = P {正面，反面，反面，正面，正面，正面}
 = P(正面|𝜃) P(正面|𝜃) P(正面|𝜃) P(正面|𝜃) P(正面|𝜃) P(正面|𝜃)
```

$= \theta * (1-\theta) * (1-\theta) \theta * \theta * \theta = \theta^4(1 - \theta)^2$


问题转化为:求此函数的极大值时，估计𝜃为多少

{% asset_img image-20230904145453737.png "Hexo 博客封面示例" %}



#### 【知道】对数函数



{% asset_img image-20230904145530932.png "Hexo 博客封面示例" %}


#### 练习


{% asset_img Snipaste_2026-10-19_14-32-49.png "Hexo 博客封面示例" %}



## 逻辑回归原理

**学习目标**
1.理解逻辑回归算法的原理

2.知道逻辑回归的损失函数

### 【理解】原理

逻辑回归概念 Logistic Regression

• 一种分类模型，把线性回归的输出，作为逻辑回归的输入。 

• 输出是(0, 1)之间的值

{% asset_img Snipaste_2026-10-19_14-35-03.png "Hexo 博客封面示例" %}


举个例子：
{% asset_img Snipaste_2026-10-19_14-38-15.png "Hexo 博客封面示例" %}
很明显逻辑回归的评估方式并不合理
他只记录准确率，而非具体的人的true 或 false



• 基本思想

1. 利用线性模型 f(x) = wx + b 根据特征的重要性计算出一个值
2. 再使用 sigmoid 函数将 f(x) 的输出值映射为概率值
   1. 设置阈值(eg:0.5)，输出概率值大于 0.5，则将未知样本输出为 1 类
   2. 否则输出为 0 类

3.逻辑回归的假设函数
 h(w) = sigmoid(wx + b )

线性回归的输出，作为逻辑回归的输入

      

**在逻辑回归中，当预测结果不对的时候，我们该怎么衡量其损失呢？**

我们来看下图(下图中，设置阈值为0.6)，

{% asset_img image-20220121161828121.png "Hexo 博客封面示例" %} 

那么如何去衡量逻辑回归的预测结果与真实结果的差异？



### 【知道】损失函数

Loss函数里，必有一项式子为0，因为y必须是0和1其中一个，而另一个式子中， 若是真实类别，概率P越大越好

{% asset_img image-20230904151300483.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-19_14-43-03.png "Hexo 博客封面示例" %}

{% asset_img image-20230904151334262.png "Hexo 博客封面示例" %}


逻辑回归 损失函数 = - 极大似然估计函数
{% asset_img image-20230904151343779.png "Hexo 博客封面示例" %}


### 总结

{% asset_img Snipaste_2026-10-19_14-55-59.png "Hexo 博客封面示例" %}




## 逻辑回归API

**学习目标：**

1.知道逻辑回归的API

2.动手实现癌症分类案例

### 【知道】API介绍

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C = 1.0)
```

**solver** **损失函数优化方法**:

训练速度:liblinear 对小数据集场景训练速度更快，sag 和 saga 对大数据集更快一些。 2 正则化:

1. newton-cg、lbfgs、sag、saga 支持 L2 正则化或者没有正则化
2. 2liblinear 和 saga 支持 L1 正则化

**penalty**:正则化的种类，l1 或者 l2

**C**:正则化力度

默认将类别数量**少**的当做正例



### 【实践】癌症分类案例

- 数据介绍

（1）699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤相关的医学特征，

     最后一列表示肿瘤类型的数值。

（2）包含16个缺失值，用"?"标出。

（3）2表示良性，4表示恶性

> 数据描述
>
> （1）699条样本，共11列数据，第一列用语检索的id，后9列分别是与肿瘤
>
> 相关的医学特征，最后一列表示肿瘤类型的数值。
>
> （2）包含16个缺失值，用"?"标出。

	

- 案例分析

```
1.获取数据
2.基本数据处理
2.1 缺失值处理
2.2 确定特征值,目标值
2.3 分割数据
3.特征工程(标准化)
4.机器学习(逻辑回归)
5.模型评估
```

- 代码实现

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # 逻辑回归模型
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dm01_LogisticRegression():
    # 1. 获取数据.
    data = pd.read_csv('data/breast-cancer-wisconsin.csv')
    data.info()

    # 2. 数据预处理.
    # data = data.replace(to_replace='?', value=np.NAN)
    data = data.replace('?', np.NaN)
    data = data.dropna()
    data.info()

    # 3. 确定特征值和目标值.
    x = data.iloc[:, 1:-1]
    y = data.Class
    print(f'x.head(): {x.head()}')
    print(f'y.head(): {y.head()}')

    # 3. 分割数据.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
    # 4. 特征处理.
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 5. 模型训练.
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 6. 模型预测
    y_predict = estimator.predict(x_test)
    print(f'预测值: {y_predict}')

    # 7. 模型评估
    print(f'准确率: {estimator.score(x_test, y_test)}')
    print(f'准确率: {accuracy_score(y_test, y_predict)}')

if __name__ == '__main__':
    dm01_LogisticRegression()

```

课堂代码：
```python
"""
案例:
    演示逻辑回归模型实现 癌症预测.

逻辑回归模型介绍:
    概述:
        属于有监督学习, 即: 有特征, 有标签, 且表示是离散的.
        主要适用于: 二分类.
    原理:
        把线性回归处理后的预测值 -> 通过 Sigmoid激活函数, 映射到[0, 1] 概率 -> 基于自定义的阈值, 结合概率来 分类.
    损失函数:
        极大似然估计函数的 负数形式.

回顾: 机器学习项目流程
    1. 加载数据.
    2. 数据预处理.
    3. 特征工程(提取, 预处理...)
    4. 模型训练.
    5. 模型预测.
    6. 模型评估.
"""

# 导包
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression     # 逻辑回归模型
from sklearn.preprocessing import StandardScaler        # 标准化
from sklearn.model_selection import train_test_split    # 训练集和测试集分割
from sklearn.metrics import accuracy_score              # 模型评估

# 1. 加载数据.
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info()     # 查看数据信息

# 2. 数据预处理.
# 2.1 把 ? 替换成 np.nan,  参1: 要被替换的值,  参2: 用来替换的值,  参3: 是否替换源数据, 默认为False
data.replace('?', np.nan, inplace=True)
# 2.2 缺失值处理 -> 删除.
data.dropna(axis=0, inplace=True)   # axis=0, 表示行, 删除包含缺失值的行.
# 2.3 打印处理后的信息.
# data.info()

# 3. 特征工程(提取, 预处理...)
# 3.1 特征提取之 提取特征和标签.
x = data.iloc[:, 1:-1]  # 按照行号, 列索引获取数据,  :表示所有行   1:-1 表示从第1列到最后1列, 包左不包右
# y = data.iloc[:, -1]    # 获取最后一列
# y = data['Class']       # 获取最后一列, 效果同上
y = data.Class            # 获取最后一列, 效果同上
# 3.2 查看下特征 和 标签.
print(x[:5])
print(y[:5])
print(x.shape, y.shape)
# 3.3 切割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
# 3.4 特征工程: 标准化
# 3.4.1 创建标准化对象.
transfer = StandardScaler()
# 3.4.2 对训练集进行标准化.  训练 + 标准化
x_train = transfer.fit_transform(x_train)
# 3.4.3 对测试集进行标准化.  标准化
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建模型对象 -> 逻辑回归模型
estimator = LogisticRegression()
# 4.2 模型训练.
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_pre = estimator.predict(x_test)
print(f'预测值为: {y_pre}')

# 6. 模型评估.
# 正确率(准确率), 公式为: 预测对的 / 样本总数
print(f'预测前评估, 正确率: {estimator.score(x_test, y_test)}')     # 测试集的特征, 标签.
print(f'预测后评估, 正确率: {accuracy_score(y_test, y_pre)}')       # 测试集的标签, 预测值.

# 思考: 逻辑回归模型能用 准确率来评测吗?
# 答案: 可以, 但是结果不精准, 因为逻辑回归模型主要用于 二分类, 即: A类还是B类, 不能说 97%的A类, 3%的B类.
# 所以要通过 混淆矩阵来评测, 即: 精确率, 召回率, F1值(F1-Score), ROC曲线, AUC值.
```

## 分类评估方法

**学习目标：**

1.理解混淆矩阵的构建方法

2.掌握精确率，召回率和F1score的计算方法

3.知道ROC曲线和AUC指标

### 【理解】混淆矩阵

{% asset_img image-20230904162952115.png "Hexo 博客封面示例" %}

混淆矩阵作用在测试集样本集中：

1. 真实值是 **正例** 的样本中，被分类为 **正例** 的样本数量有多少，这部分样本叫做真正例（TP，True Positive）
2. 真实值是 **正例** 的样本中，被分类为 **假例** 的样本数量有多少，这部分样本叫做伪反例（FN，False Negative）
3. 真实值是 **假例** 的样本中，被分类为 **正例** 的样本数量有多少，这部分样本叫做伪正例（FP，False Positive）
4. 真实值是 **假例** 的样本中，被分类为 **假例** 的样本数量有多少，这部分样本叫做真反例（TN，True Negative）

{% asset_img Snipaste_2026-10-19_16-16-31.png "Hexo 博客封面示例" %}

> True Positive ：表示样本真实的类别
> Positive ：表示样本被预测为的类别


**例子：**

样本集中有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例，则：

**模型 A：** 预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本

1. 真正例 TP 为：3 
2. 伪反例 FN 为：3
3. 伪正例 FP 为：0
4. 真反例 TN：4

**模型 B：** 预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本

1. 真正例 TP 为：6
2. 伪反例 FN 为：0
3. 伪正例 FP 为：3
4. 真反例 TN：1

{% asset_img Snipaste_2026-10-19_16-19-55.png "Hexo 博客封面示例" %}

练习：

{% asset_img Snipaste_2026-10-19_16-22-41.png "Hexo 博客封面示例" %}

我们会发现：TP+FN+FP+TN = 总样本数量

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd


def dm01_混淆矩阵四个指标():
    # 1. 定义样本集, 6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

    # 2. 定义标签
    labels = ['恶性', '良性']

    # 3. 模型A => 预测对了 3个恶性, 4个良性.
    y_pred_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
    result = confusion_matrix(y_true, y_pred_A)
    print(pd.DataFrame(result, index=labels, columns=labels))

    # 4. 模型B => 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本.
    y_pred_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性']
    result = confusion_matrix(y_true, y_pred_B)
    print(pd.DataFrame(result, index=labels, columns=labels))


    # 5. 计算精确率 (precision)
    print(f'模型A精确率: {precision_score(y_true, y_pred_A, pos_label="恶性")}')
    print(f'模型B精确率: {precision_score(y_true, y_pred_B, pos_label="恶性")}')

    # 6. 计算召回率 (recall)
    print(f'模型A召回率: {recall_score(y_true, y_pred_A, pos_label="恶性")}')
    print(f'模型B召回率: {recall_score(y_true, y_pred_B, pos_label="恶性")}')

    # 7. 计算F1值
    print(f'模型A F1值: {f1_score(y_true, y_pred_A, pos_label="恶性")}')
    print(f'模型B F1值: {f1_score(y_true, y_pred_B, pos_label="恶性")}')


if __name__ == '__main__':
    dm01_混淆矩阵四个指标()
```

课堂代码：
```python
"""
案例: 演示混淆矩阵.

混淆矩阵:
    概述:
        用于展示 真实值 和 预测值之间, 正例, 反例的情况.
    默认:
        会用 分类少的 样本当做 正例.
    混淆矩阵名词解释:
                    预测值(正例)     预测值(反例)             # True(真), False(伪),   Positive(正例),  Negative(反例)
        真实值(正例)   真正例(TP)     伪反例(FN)
        真实值(反例)   伪正例(FP)     真反例(TN)
"""

# 导包
import pandas as pd
from sklearn.metrics import confusion_matrix

# 1. 定义数据集, 表示: 真实样本(共计10个, 6个恶性, 4个良性) => 设置: 恶性(正例), 良性(反例)
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',   '良性', '良性', '良性', '良性']

# 2. 定义标签名.
label = ['恶性', '良性']        # 正样本(正例), 负样本(反例)
df_label = ['恶性(正例)', '良性(反例)']

# 3. 定义 预测结果A, 预测对了 -> 3个恶性肿瘤, 4个良性肿瘤.
y_pre_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性',   '良性', '良性', '良性', '良性']
# 4. 把上述的 预测结果A 转换成 混淆矩阵.
# 参1: 真实样本, 参2: 预测样本, 参3: 样本标签(正例, 反例)
cm_A = confusion_matrix(y_train, y_pre_A, labels=label)
print(f'混淆矩阵A: \n {cm_A}')
# 5. 把混淆矩阵 转换成 DataFrame.
df_A = pd.DataFrame(cm_A, index=df_label, columns=df_label)
print(f'预测结果A对应的DataFrame对象: \n {df_A}')
print('-' * 22)

# 6. 定义 预测结果B, 预测对了 -> 6个恶性肿瘤, 1个良性肿瘤.
y_pre_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',   '恶性', '恶性', '恶性', '良性']
# 7. 把上述的 预测结果B 转换成 混淆矩阵.
cm_B = confusion_matrix(y_train, y_pre_B, labels=label)
print(f'混淆矩阵B: \n {cm_B}')
# 8. 把混淆矩阵 转换成 DataFrame.
df_B = pd.DataFrame(cm_B, index=df_label, columns=df_label)
print(f'预测结果B对应的DataFrame对象: \n {df_B}')
```





### 【掌握】Precision（精确率）


精确率也叫做查准率，指的是对正例样本的预测准确率。比如：我们把恶性肿瘤当做正例样本，则我们就需要知道模型对恶性肿瘤的预测准确率。


{% asset_img 05.png "Hexo 博客封面示例" %}



**例子：**

样本集中有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例，则：

**模型 A：** 预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本

1. 真正例 TP 为：3 
2. 伪反例 FN 为：3
3. 假正例 FP 为：0
4. 真反例 TN：4
5. **精准率：3/(3+0) = 100%**



**模型 B：** 预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本

1. 真正例 TP 为：6
2. 伪反例 FN 为：0
3. 假正例 FP 为：3
4. 真反例 TN：1
5. **精准率：6/(6+3) = 67%**







### 【掌握】Recall（召回率）


召回率也叫做查全率，指的是预测为真正例样本占所有真实正例样本的比重。例如：我们把恶性肿瘤当做正例样本，则我们想知道模型是否能把所有的恶性肿瘤患者都预测出来。


{% asset_img 06.png "Hexo 博客封面示例" %}


**例子：**

样本集中有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例，则：

**模型 A：** 预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本

1. 真正例 TP 为：3 
2. 伪反例 FN 为：3
3. 假正例 FP 为：0
4. 真反例 TN：4
5. **精准率：3/(3+0) = 100%**
6. **召回率：3/(3+3)=50%**

**模型 B：** 预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本

1. 真正例 TP 为：6
2. 伪反例 FN 为：0
3. 假正例 FP 为：3
4. 真反例 TN：1
5. **精准率：6/(6+3) = 67%**
6. **召回率：6/(6+0)= 100%** 




###  【掌握】F1-score

如果我们对模型的精度、召回率都有要求，希望知道模型在这两个评估方向的综合预测能力如何？则可以使用 F1-score 指标。

{% asset_img 07.png "Hexo 博客封面示例" %}

样本集中有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例，则：

**模型 A：** 预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本



1. 真正例 TP 为：3 
2. 伪反例 FN 为：3
3. 假正例 FP 为：0
4. 真反例 TN：4
5. **精准率：3/(3+0) = 100%**
6. **召回率：3/(3+3)=50%**
7. **F1-score：(2\*3)/(2\*3+3+0)=67%**

**模型 B：** 预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本

1. 真正例 TP 为：6
2. 伪反例 FN 为：0
3. 假正例 FP 为：3
4. 真反例 TN：1
5. **精准率：6/(6+3) = 67%**
6. **召回率：6/(6+0)= 100%** 
7. **F1-score：(2\*6)/(2\*6+0+3)=80%**


综合代码：
```python
"""
案例:
    演示混淆矩阵 和 精确率, 召回率, F1值.

回顾: 逻辑回归
    概述:
        属于有监督学习, 即: 有特征, 有标签, 且标签是离散的.
        适用于 二分类.
    评估:
        精确率, 召回率, F1值

混淆矩阵:
    概述:
        用来描述 真实值  和 预测值之间关系的.
    图解:
                        预测标签(正例)        预测标签(反例)
        真实标签(正例)      真正例(TP)           伪反例(FN)
        真实标签(反例)      伪正例(FP)           真反例(TN)
    单词:
        True: 真,  False: 假(伪)
        Positive: 正例
        Negative: 反例

    结论:
        1. 模拟使用 分类少的 充当 正例.
        2. 精确率 = 真正例 在 预测正例中的占比, 即: tp / (tp + fp)
        3. 召回率 = 真正例 在 真正例中的占比, 即: tp / (tp + fn)
        4. F1值 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
"""

# 导包
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score    # 混淆矩阵, 精确率, 召回率, F1值


# 需求: 已知有10个样本, 6个恶性肿瘤(正例), 4个良性肿瘤(反例).
# 模型A预测结果为: 预测对了3个恶性肿瘤, 预测对了4个良性肿瘤
# 模型B预测结果为: 预测对了6个恶性肿瘤, 预测对了1个良性肿瘤
# 请针对于上述的数据集, 搭建 混淆矩阵, 并分别计算模型A, 模型B的 精确率, 召回率, F1值.

# 1. 定义变量, 记录: 样本数据
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',     '良性', '良性', '良性', '良性']

# 2. 定义变量, 记录: 模型A的预测结果
y_pred_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性',    '良性', '良性', '良性', '良性']

# 3. 定义变量, 记录: 模型B的预测结果
y_pred_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',     '良性', '恶性', '恶性', '恶性']

# 4. 用标签标记 正例, 反例.
label = ['恶性', '良性'] # [正例，反例]
df_label = ['恶性(正例)', '良性(反例)']

# 5. 针对于 真实值(y_train) 和 模型A的预测结果(y_pred_A), 搭建 混淆矩阵.
cm_A = confusion_matrix(y_train, y_pred_A, labels=label)
print(f'混淆矩阵A: \n {cm_A}')

# 6. 为了测试结果更好看, 把上述的 混淆矩阵 转换成 DataFrame.
#index（行上打标签）  columns(列上打标签)
df_A = pd.DataFrame(cm_A, index=df_label, columns=df_label)
print(f'混淆矩阵A的 DataFrame对象形式: \n {df_A}')

# 7. 针对于 真实值(y_train) 和 模型B的预测结果(y_pred_B), 搭建 混淆矩阵.
cm_B = confusion_matrix(y_train, y_pred_B, labels=label)
print(f'混淆矩阵B: \n {cm_B}')

# 8. 为了测试结果更好看, 把上述的 混淆矩阵 转换成 DataFrame.
df_B = pd.DataFrame(cm_B, index=df_label, columns=df_label)
print(f'混淆矩阵B的 DataFrame对象形式: \n {df_B}')

# 9. 计算A模型的 精确率, 召回率, F1值.
print(f'模型A 精确率: {precision_score(y_train, y_pred_A, pos_label='恶性')}')   # 参1: 真实值, 参2: 预测值, 参3: 正例的标签
print(f'模型A 召回率: {recall_score(y_train, y_pred_A, pos_label="恶性")}')      # 参1: 真实值, 参2: 预测值, 参3: 正例的标签
print(f'模型A F1值: {f1_score(y_train, y_pred_A, pos_label="恶性")}')           # 参1: 真实值, 参2: 预测值, 参3: 正例的标签

# 10. 计算B模型的 精确率, 召回率, F1值.
print(f'模型B 精确率: {precision_score(y_train, y_pred_B, pos_label='恶性')}')   # 参1: 真实值, 参2: 预测值, 参3: 正例的标签
print(f'模型B 召回率: {recall_score(y_train, y_pred_B, pos_label="恶性")}')      # 参1: 真实值, 参2: 预测值, 参3: 正例的标签
print(f'模型B F1值: {f1_score(y_train, y_pred_B, pos_label="恶性")}')           # 参1: 真实值, 参2: 预测值, 参3: 正例的标签
```


###  【知道】ROC曲线和AUC指标

####  ROC 曲线

ROC 曲线：我们分别考虑正负样本的情况：

1. 正样本中被预测为正样本的概率，即：TPR （True Positive Rate）
2. 负样本中被预测为正样本的概率，即：FPR （False Positive Rate）

{% asset_img image-20230904182146483.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-19_17-06-28.png "Hexo 博客封面示例" %}

ROC 曲线图像中，4 个特殊点的含义：

{% asset_img Snipaste_2026-10-19_17-08-16.png "Hexo 博客封面示例" %}

1. (0, 0) 表示所有的正样本都预测为错误，所有的负样本都预测正确
2. (1, 0) 表示所有的正样本都预测错误，所有的负样本都预测错误
3. (1, 1) 表示所有的正样本都预测正确，所有的负样本都预测错误
4. (0, 1) 表示所有的正样本都预测正确，所有的负样本都预测正确



#### 绘制 ROC 曲线

假设：在网页某个位置有一个广告图片或者文字，该广告共被展示了 6 次，有 2 次被浏览者点击了。每次点击的概率如下：



| 样本 | 是否被点击 | 预测点击概率 |
| :--: | :--------: | :----------: |
|  1   |     1      |     0.9      |
|  2   |     0      |     0.7      |
|  3   |     1      |     0.8      |
|  4   |     0      |     0.6      |
|  5   |     0      |     0.5      |
|  6   |     0      |     0.4      |

根据预测点击概率排序之后：

| 样本 | 是否被点击 | 预测点击概率 |
| :--: | :--------: | :----------: |
|  1   |     1      |     0.9      |
|  3   |     1      |     0.8      |
|  2   |     0      |     0.7      |
|  4   |     0      |     0.6      |
|  5   |     0      |     0.5      |
|  6   |     0      |     0.4      |


{% asset_img Snipaste_2026-10-19_17-31-21.png "Hexo 博客封面示例" %}


{% asset_img Snipaste_2026-10-19_17-29-05.png "Hexo 博客封面示例" %}


{% asset_img Snipaste_2026-10-19_17-30-03.png "Hexo 博客封面示例" %}


{% asset_img Snipaste_2026-10-19_17-30-29.png "Hexo 博客封面示例" %}



绘制 ROC 曲线：

**阈值：0.9**

1. 原本为正例的 1、3 号的样本中 3 号样本被分类错误，则 TPR = 1/2 = 0.5
2. 原本为负例的 2、4、5、6 号样本没有一个被分为正例，则 FPR = 0

**阈值：0.8**

1. 原本为正例的 1、3 号样本被分类正确，则 TPR = 2/2 = 1
2. 原本为负例的 2、4、5、6 号样本没有一个被分为正例，则 FPR = 0

**阈值：0.7**

1. 原本为正例的 1、3 号样本被分类正确，则 TPR = 2/2 = 1
2. 原本为负类的 2、4、5、6 号样本中 2 号样本被分类错误，则 FPR = 1/4 = 0.25

**阈值：0.6**

1. 原本为正例的 1、3 号样本被分类正确，则 TPR = 2/2 = 1
2. 原本为负类的 2、4、5、6 号样本中 2、4 号样本被分类错误，则 FPR = 2/4 = 0.5

**阈值：0.5**

1. 原本为正例的 1、3 号样本被分类正确，则 TPR = 2/2 = 1
2. 原本为负类的 2、4、5、6 号样本中 2、4、5 号样本被分类错误，则 FPR = 3/4 = 0.75

**阈值 0.4**

1. 原本为正例的 1、3 号样本被分类正确，则 TPR = 2/2 = 1
2. 原本为负类的 2、4、5、6 号样本全部被分类错误，则 FPR = 4/4 = 1



(0, 0.5)、(0, 1)、(0.25, 1)、(0.5, 1)、(0.75, 1)、(1, 1)



由 TPR 和 FPR 构成的 ROC 图像为：



{% asset_img image-20230914105309023.png "Hexo 博客封面示例" %}



#### AUC 值



1. 我们发现：图像越靠近 (0,1) 点则模型对正负样本的辨别能力就越强
2. 我们发现：图像越靠近 (0, 1) 点则 ROC 曲线下面的面积就会越大
3. AUC 是 ROC 曲线下面的面积，该值越大，则模型的辨别能力就越强
4. AUC  范围在 [0, 1] 之间
5. 当 AUC= 1 时，该模型被认为是完美的分类器，但是几乎不存在完美分类器

> AUC 值主要评估模型对正例样本、负例样本的辨别能力.

### 总结
 
{% asset_img Snipaste_2026-10-19_17-33-01.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-19_17-36-46.png "Hexo 博客封面示例" %}

####  **分类评估报告api**

```python
sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None )
  '''
  y_true：真实目标值
  y_pred：估计器预测目标值
  labels:指定类别对应的数字
  target_names：目标类别名称
  return：每个类别精确率与召回率
  '''
```



####  **AUC计算API**

```
from sklearn.metrics import roc_auc_score
  sklearn.metrics.roc_auc_score(y_true, y_score)
    计算ROC曲线面积，即AUC值
    y_true：每个样本的真实类别，必须为0(反例),1(正例)标记
    y_score：预测得分，可以是正例的估计概率、置信值或者分类器方法的返回值
```







## 【实践】电信客户流失预测

**学习目标：**

1.了解案例的背景信息

2.知道案例的处理流程

3.动手实现电信客户流失案例的代码





### 数据集介绍

- 流失用户指的使用过产品因为某些原因不再使用该产品。随着产品的更新迭代，都会存在一定的流失情况，这时正常现象。流失用户的比例和变化趋势能够反映该产品当前是否存在问题以及未来的发展趋势。
- 当用户群体庞大时，有限的人力和精力不能为每个用户都投入大量的时间。如果公司可以预测哪些用户可能提前流失，这样就能将主要精力聚焦于这些用户，实施有效的用户挽留策略，提高用户粘性。
- 本项目旨在通过分析特征属性确定用户流失的原因，以及哪些因素可能导致用户流失。建立预测模型来判断用户是否流失，并提出用户流失预警策略。
- 具体数据说明如下：数据集中总计7043条数据，21个特征字段，最终分类特征为Churn：用户是否流失，具体内容如下所示：
  customerID：用户ID
  gender：性别
  SeniorCitizen：是否是老人
  Partner：是否有伴侣
  Dependents：是否有需要抚养的孩子
  tenure：任职
  PhoneService：是否办理电话服务
  MultipleLines：是否开通了多条线路
  InternetService：是否开通网络服务和开通的服务类型（光纤、电话拨号）
  TechSupport：是否办理技术支持服务
  OnlineBackup：是否办理在线备份服务
  OnlineSecurity：是否办理在线安全服务
  DeviceProtection：是否办理设备保护服务
  StreamingTV：是否办理电视服务
  StreamingMovies：是否办理电影服务
  Contract：签订合约的时长
  PaperlessBilling：是否申请了无纸化账单
  PaymentMethod：付款方式（电子支票、邮寄支票、银行自动转账、信用卡自动转账）
  MonthlyCharges：月消费
  TotalCharges：总消费
  Churn：用户是否流失

### 处理流程

1、数据基本处理

 	查看数据的基本信息

	对类别数据数据进行one-hot处理

	查看标签分布情况


{% asset_img Snipaste_2026-10-19_17-52-07.png "Hexo 博客封面示例" %}

{% asset_img Snipaste_2026-10-19_18-01-42.png "Hexo 博客封面示例" %}

2、特征筛选

	分析哪些特征对标签值影响大

	初步筛选出对标签影响比较大的特征，形成x、y

3、模型训练

	模型训练

	交叉验证网格搜索等

4、模型评估

	精确率

	Roc_AUC指标计算


### 课堂代码
```python
"""
案例:
    电信客户流失分析.

目的:
    1. 演示逻辑回归的相关操作, 主要是: 二分法(流失, 不流失)
    2. 演示逻辑回归的评估操作, 主要是: 混淆矩阵, 准确率, 召回率, F1值, ROC曲线, AUC值, 分类评估报告(了解)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
#                             准确率           精确率            召回率        F1值        roc曲线          分类评估报告
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# 1. 定义函数, 用于实现: 数据预处理.
def dm01_数据预处理():
    # 1. 读取数据.
    data = pd.read_csv('./data/churn.csv')
    data.info()

    # 2. 因为上述的Churn, gender是字符串类型, 我们对其做独热编码(one-hot)处理.
    data = pd.get_dummies(data)
    data.info()
    print(data.head(10))

    # 3. 删除列, 因为热编码之后, 会多出一个列, 我们删除掉.
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    print(data.head(10))

    # 4. 修改列名.
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    print(data.head(10))

    # 5. 我们查看下数据集中, 标签 是否是 均衡的.
    print(data.flag.value_counts()) # False -> 不流失, True -> 流失

# 2. 定义函数, 用于显示: 月度会员的流失情况.
def dm02_会员流失可视化情况():
    # 1. 读取数据.
    data = pd.read_csv('./data/churn.csv')
    # 2. 对上述的数据做 热编码处理.
    data = pd.get_dummies(data)
    # 3. 删除列, 因为热编码之后, 会多出一个列, 我们删除掉.
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    # 4. 修改列名.
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    # 5. 查看数据集的分布情况.
    print(data.flag.value_counts())
    print(data.columns) # 查看所有列名.

    # 6. 通过计数柱状图, 绘制(月度)会员的流失情况.
    # 参数x意思是: x轴的列名(是否是月度会员, 0 -> 不是会员, 1 -> 是会员)
    # 参数hue意思是: 根据hue的值, 将数据进行分类(False -> 不流失, True -> 流失)
    sns.countplot(data, x='Contract_Month', hue='flag')
    plt.show()


# 3. 定义函数, 用于实现: 逻辑回归模型的训练和评估.
def dm03_逻辑回归模型训练评估():
    # 1. 读取数据.
    data = pd.read_csv('./data/churn.csv')
    # 2. 对上述的数据做 热编码处理.
    data = pd.get_dummies(data)
    # 3. 删除列, 因为热编码之后, 会多出一个列, 我们删除掉.
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    # 4. 修改列名.
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)
    # 5. 查看数据集, 从中筛除: 特征列 和 标签列.
    # print(data.head(10))    # 特征列: Contract_Month, PaymentElectronic, internet_other
    # print(data.columns)     # 标签列: flag

    # 6. 拆分训练集和测试集.
    x = data[['Contract_Month', 'PaymentElectronic', 'internet_other']]
    y = data['flag']
    # print(len(x), len(y))
    # print(x.head(10))
    # print(y.head(10))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

    # 7. 创建逻辑回归模型, 并训练.
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 8. 模型预测.
    y_predict = estimator.predict(x_test)
    print(f'预测值为: {y_predict}')

    # 9. 模型评估.
    # 9.1 准确率.
    print(f'准确率: {estimator.score(x_test, y_test)}')
    print(f'准确率: {accuracy_score(y_test, y_predict)}')  # 真实值, 预测值.
    print('-' * 22)
    # 9.2 精确率.
    print(f'精确率: {precision_score(y_test, y_predict)}')
    print('-' * 22)
    # 9.3 召回率.
    print(f'召回率: {recall_score(y_test, y_predict)}')
    print('-' * 22)
    # 9.4 F1值
    print(f'F1值: {f1_score(y_test, y_predict)}')
    print('-' * 22)
    # 9.5 roc曲线
    print(f'roc曲线: {roc_auc_score(y_test, y_predict)}')
    print('-' * 22)
    # 9.6 分类评估报告
    # 参数macro avg意思是: 宏平均, 是指: 所有的分类器, 都按照 macro 的方式, 计算平均值/
    # 不考虑样本的权重, 直接平均, 跟样本的数量, 权重无关, 所有特征权重都一样, 适合于 数据集比较平衡的情况.

    # 参数weighted avg意思是: 权重平均, 是指: 所有的分类器, 都按照 weighted 的方式, 计算平均值/
    # 考虑样本的权重, 根据样本的权重, 计算平均值, 适合于 数据集比较不平衡的情况.
    print(f'分类评估报告: {classification_report(y_test, y_predict)}')

# 4. 在main函数中测试.
if __name__ == '__main__':
    # dm01_数据预处理()
    # dm02_会员流失可视化情况()
    dm03_逻辑回归模型训练评估()
```

### 课堂代码2
```python
"""
案例:
    通过逻辑回归算法, 针对于电信用户数据建模, 进行流失预测分析.
"""

# 导包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  # 准确率,精确率,召回率,F1值, 分类评估报告

# 1. 定义函数, 演示: 数据的预处理.
def dm01_data_preprocess():
    # 1. 读取csv文件, 获取到df对象.
    churn_df = pd.read_csv('./data/churn.csv')
    
    # 2. 查看(处理前)的数据集.
    # churn_df.info()
    # print(churn_df.head(5))
   
    # 3. 因为Churn 和 gender列是字符串, 所以需要进行one-hot编码(热编码处理).
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])

    # 4. 查看(处理后)的数据集
    # churn_df.info()
    # print(churn_df.head(5))
   
    # 5. 删除one-hot处理后, 冗余的列.
    # 参1: 要删除的列. 参2: 1表示删除列   参3: inplace=True表示直接修改原数据.
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)

    # churn_df.info()
    # print(churn_df.head(5))
    
    # 6. 修改列名, 将Churn_Yes -> flag, 充当标签列.
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    churn_df.info()
    print(churn_df.head(5))     # False -> 不流失, True -> 流失

    # 7. 查看数据值的分布.
    print(churn_df.flag.value_counts())     # False: 5174, True: 1869

# 2. 定义函数, 演示: 数据的可视化.
def dm02_data_visualization():
    # 1. 读取csv文件, 获取到df对象.
    churn_df = pd.read_csv('./data/churn.csv')

    # 2. 对object类型的列(数据)做 one-hot编码处理.
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])

    # 3. 删除one-hot处理后, 冗余的列.
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)

    # 4. 修改列名, 将Churn_Yes -> flag, 充当标签列.
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    # 5. 查看数据值的分布.
    print(churn_df.flag.value_counts())     # False: 5174, True: 1869

    # 6. 查看列名, 方便我们一会儿抽取 特征.   
    # 列名为: ['Partner_att', 'Dependents_att', 'landline', 'internet_att', 'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
    # Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic', 'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Female']
    print(churn_df.columns)

    # 7. 数据的可视化, 绘制 计数柱状图.
    # 参1: 数据集, 参2: x轴的列名(月度会员), 参3: hue表示分组, 根据分组进行绘制, 这里是: 是否流失(False->不流失, True->流失).
    sns.countplot(data=churn_df, x='Contract_Month', hue='flag')
    plt.show()


# 3. 定义函数, 演示: 逻辑回归算法的模型训练,预测,评估.
def dm03_logistic_regression():
    # 1. 加载数据集.
    churn_df = pd.read_csv('./data/churn.csv')

    # 2.数据的预处理.
    # 2.1 对object类型的列(数据)做 one-hot编码处理.
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])

    # 2.2 删除one-hot处理后, 冗余的列.
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True) # 按列删除（axis=1）

    # 2.3 修改列名, 将Churn_Yes -> flag, 充当标签列.
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    # 2.4 提取特征列 和 标签列
    # x的特征列: 月度会员, 是否有互联网服务, 是否是电子支付
    x = churn_df[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_df['flag']        # False -> 不流失, True -> 流失

    # 2.5 划分训练集和测试集.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    # 3.特征工程(例如: 特征提取, 特征预处理->归一化, 标准化 ...), 暂不处理

    # 4.模型训练.
    # 4.1 创建(逻辑回归)模型对象.
    estimator = LogisticRegression()
    # 4.2 模型训练.
    estimator.fit(x_train, y_train)

    # 5.模型预测.
    y_pred = estimator.predict(x_test)
    print(f'预测值为: {y_pred}')

    # 6.模型评估.
    print(f'准确率: {estimator.score(x_test, y_test)}')    # 预测前, 0.7679205110007097
    print(f'准确率: {accuracy_score(y_test, y_pred)}')     # 预测后, 0.7679205110007097

    print(f'精确率: {precision_score(y_test, y_pred)}')    # 0.5807692307692308
    print(f'召回率: {recall_score(y_test, y_pred)}')       # 0.4092140921409214
    print(f'F1值: {f1_score(y_test, y_pred)}')            # 0.48012718600953896

    # macro avg: 宏平均, 即: 不考虑样本权重, 直接求平均. 适用于: 数据均衡的情况.
    # weighted avg: 样本权重平均, 即: 考虑样本权重, 求平均. 适用于: 数据不均衡的情况.
    print(f'分类评估报告: \n{classification_report(y_test, y_pred)}')

# 4. 测试
if __name__ == '__main__':
    # dm01_data_preprocess()
    # dm02_data_visualization()
    dm03_logistic_regression()
```

###  案例实现

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# 1. 定义函数, 表示: 数据基本处理
def dm01_数据基本处理():
    # 1. 读取数据, 查看数据的基本信息.
    churn_pd = pd.read_csv('data/churn.csv')
    # churn_pd.info()
    # print(f'churn_pd.describe(): {churn_pd.describe()}')
    # print(f'churn_pd: {churn_pd}')

    # 2. 处理类别型的数据, 类别型数据做 one-hot编码(热编码).
    churn_pd = pd.get_dummies(churn_pd)
    churn_pd.info()
    # print(f'churn_pd: {churn_pd}')

    # 3. 去除列 Churn_No, gender_Male
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)  # 按列删除
    print(f'churn_pd: {churn_pd}')

    # 4. 列标签重命名, 打印列名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    print(f'列名: {churn_pd.columns}')

    # 5. 查看标签的分布情况 0.26用户流失
    value_counts = churn_pd.flag.value_counts()
    print(value_counts)


# 2. 定义函数, 表示: 特征筛选
def dm02_特征筛选():
    # 1. 读取数据
    churn_pd = pd.read_csv('data/churn.csv')
    # 2. 处理类别型的数据, 类别型数据做 one-hot编码(热编码).
    churn_pd = pd.get_dummies(churn_pd)
    # 3. 去除列 Churn_No, gender_Male
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 4. 列标签重命名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 5. 查看标签的分布情况
    value_counts = churn_pd.flag.value_counts()
    print(value_counts)
    # 6. 查看Contract_Month 是否预签约流失情况
    sns.countplot(data=churn_pd, x='Contract_Month', hue='flag')
    plt.show()


# 3. 定义函数, 表示: 模型训练 和 评测
def dm03_模型训练和评测():
    # 1. 读取数据
    churn_pd = pd.read_csv('data/churn.csv')

    # 2. 数据预处理
    # 2.1 处理类别型的数据, 类别型数据做 one-hot编码(热编码).
    churn_pd = pd.get_dummies(churn_pd)
    # 2.2 去除列 Churn_No, gender_Male
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 2.3 列标签重命名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)

    # 3. 特征处理.
    # 3.1 提取特征和标签
    x = churn_pd[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_pd['flag']
    # 3.2 训练集和测试集的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)

    # 4. 模型训练.
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 5. 模型预测
    y_predict = estimator.predict(x_test)
    print(f'预测结果: {y_predict}')

    # 6. 模型评估
    print(f'准确率: {accuracy_score(y_test, y_predict)}')
    print(f'准确率: {estimator.score(x_test, y_test)}')

    # 计算AUC值.
    print(f'AUC值: {roc_auc_score(y_test, y_predict)}')



if __name__ == '__main__':
    # dm01_数据基本处理()
    # dm02_特征筛选()
    dm03_模型训练和评测()

```


分类评估报告：
参数macro avg意思是: 宏平均, 是指: 所有的分类器, 都按照 macro 的方式, 计算平均值/不考虑样本的权重, **直接平均**, 跟样本的数量, 权重无关, 所有特征权重都一样, 适合于 数据集比较平衡的情况.

参数weighted avg意思是: 权重平均, 是指: 所有的分类器, 都按照 weighted 的方式, 计算平均值/考虑样本的权重, 根据样本的权重, 计算平均值, 适合于 数据集比较不平衡的情况




## 作业

1. 使用思维导图总结逻辑回归的内容



2. 理解分类评估方法并进行详细的描述



3.动手实现癌症分类和电信用户流失案例(数据处理，特征工程，CV。。。。)
