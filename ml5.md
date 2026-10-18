---
title: 首次感到差异
date: 2026-10-15 12:35:21
tags:
---
# 为什么 sklearn 一行 `fit` 就能训练，而 PyTorch 要自己搭图和写循环？

## 引子

如果你是从深度学习入门、最近才开始接触 scikit-learn，大概率会被同一个反差冲击到。

在 PyTorch 里，训练一个模型意味着：定义一个继承 `nn.Module` 的类、在 `__init__` 里声明各层、实现 `forward`、再手写训练循环（loss、optimizer、backward、zero_grad）。

而在 sklearn 里：

```python
estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train, y_train)
```

两行结束。

这个差异常被解读为"两个领域的风格不同"，但它背后其实是一个明确的设计问题：**训练流程的控制权归谁**。本文从这个问题出发，梳理两者的差异，并说明为什么近年深度学习又开始重新长出 `fit` 风格的 API。

---

## 一、sklearn 的 `fit` 封装了什么

先澄清一个常见误解：`KNeighborsClassifier(n_neighbors=3)` 这一行并不是"训练"，它只是构建了一个**配置对象**——把超参数存进 `self`，此时模型中不存在任何学到的参数。

真正的工作在 `fit` 里。而 `fit` 能用一个方法概括一切，前提是**算法流程是固定的**。以 KNN 为例，它是惰性学习，`fit` 内部几乎只是把训练数据存起来：

```python
def fit(self, X, y):
    self._fit_X = X
    self._y = y
    return self
```

即便换成有参数的模型（`LogisticRegression`、`SVC`），其优化算法（L-BFGS、SMO、坐标下降）同样是**写死在框架源码里**的，用户能改的只有超参数（`C`、`kernel`、`max_iter`）。

这就是 sklearn 统一 API（`fit / predict / transform / score`）成立的前提：**模型结构是常量，变化的只有超参数**。正因如此，"训练"才可以被完整地封装成一个黑盒方法。

---

## 二、深度学习为什么必须自己搭图

神经网络不符合上述前提。它不是一个"算法 + 若干超参数"的组合，而是一张**由用户自由组合的计算图**。CNN、RNN、Transformer、残差连接、自定义 attention——不存在统一的"结构模板"。

因此框架只能提供**积木**（`nn.Linear`、`nn.Conv2d`、`nn.Module`），由你拼出前向传播。继承 `nn.Module` 的目的不是"训练模型"，而是**描述计算图**。

这里有一个值得注意的对称性：PyTorch 的 `forward()` 也是**你写、框架调用**的。所以两个世界都存在"框架调用用户代码"这一层。区别在于用户交出去的东西不同：

- **sklearn**：你交出去的是**超参数**，算法逻辑在框架里；
- **PyTorch**：你交出去的是**计算逻辑本身**，框架只负责张量运算与自动求导。

另一处细节差异在实例化时刻：`nn.Linear(10, 3)` 实例化时权重就已随机初始化，而 sklearn 估计器在实例化时不持有任何参数。换句话说，**PyTorch 的实例化 = 声明结构 + 立即分配参数张量；sklearn 的实例化 = 声明"用什么算法 + 什么超参数"**。

---

## 三、本质差异：控制反转的方向

把上述观察归纳成一句话：**两边差别在于控制流归谁。**

| | sklearn | PyTorch |
|---|---|---|
| 用户定义什么 | 超参数 | 网络结构 + 前向传播 |
| 训练流程 | 框架固定，封装进 `fit` | 用户显式编写训练循环 |
| 控制流 | 框架拥有（你调 `fit`，框架接管） | 用户拥有（框架退化为基础设施） |
| 可学习参数 | 少，结构固定 | 多，结构由用户决定 |
| 抽象单位 | "模型" | "层 / 计算图" |

sklearn 是**框架控制反转**：你调用 `fit`，流程主权交给框架。PyTorch 是**用户控制流程**：框架只提供 `Tensor` 与 `autograd`，流程主权在你手上。

这解释了为什么深度学习要手写训练循环——不是做不到封装，而是**控制权被有意保留给使用者**。损失函数、优化器、学习率调度、梯度裁剪、混合精度、分布式策略，乃至 GAN 的对抗训练、强化学习的 rollout，差异极大，一个统一的 `fit()` 无法覆盖。

---

## 四、抽象层次是一条连续谱

把两个世界对立起来是简化说法。更准确的图景是：**抽象层次是一条连续谱，不是阵营对立**。

```
高抽象   sklearn fit/predict   ←→   Keras / Lightning / HF Trainer
             ↑                                ↑
        结构固定、流程标准              任务标准化后封装出现
             ↓                                ↓
低抽象   自定义 estimator/核/Pipeline ←→ 原生 nn.Module + 训练循环
```

决定默认抽象层的因素，是**"结构本身是否为创新点"**：

- 传统机器学习：算法结构是已知常量，创新点在特征工程与超参数 → 默认高抽象；
- 深度学习：网络结构本身就是研究变量 → 默认低抽象。

但两个方向都能往对面走。sklearn 提供 `BaseEstimator` / `TransformerMixin`，允许你写自定义估计器，角色上对标 `nn.Module`；传统 ML 也有丰富的"构建"手段——`Pipeline` 组合预处理、`SVC(kernel=my_kernel)` 自定义核、XGBoost 传入自定义目标函数、`statsmodels` 的公式化建模、`pgmpy` 手动定义贝叶斯网结构。所谓"传统 ML 不能构建"，只是默认结构封闭而已。

---

## 五、`fit` 在深度学习中的回归

一旦任务标准化，`fit` 风格的高层封装必然出现。近年的谱系大致如下：

| 层级 | 代表 | 简化程度 |
|---|---|---|
| 全封装 | Keras 3 `Sequential` | `model.fit(X, y, epochs=10)`，loss 都有默认值 |
| 训练器封装 | PyTorch Lightning `Trainer` | 只写 `training_step`，循环/多卡/混合精度全接管 |
| 任务封装 | HF `Trainer` | 微调 BERT 只需数行 |
| 一行式 | fastai `learn.fit_one_cycle(5)` | 内置学习率调度与迁移学习套路 |

其中两个案例尤其值得注意。

其一是 **Keras 3**：同一套 `fit` API 可以运行在 JAX / TensorFlow / PyTorch 三个后端之上。"`fit` 式体验"与"底层框架"彻底解耦。

其二是 **skorch**：它直接把 PyTorch 模型包装成 sklearn 估计器。

```python
from skorch import NeuralNetClassifier
net = NeuralNetClassifier(MyModule, max_epochs=10, lr=0.1)
net.fit(X_train, y_train)      # sklearn 风格
net.predict(X_test)
```

它甚至能直接塞进 `GridSearchCV` 与 `Pipeline`。这是前文所说"两边往对面走"最极致的体现：**PyTorch 的自由搭图与 sklearn 的控制反转，共存于同一行代码**。

需要强调的是，高层封装的"简单"建立在**任务标准化**之上。一旦涉及自定义 loss（多任务加权、对比学习）、训练流程本身是研究对象（GAN、RL、课程学习）、非常规数据流（图、稀疏、流式），或需要精细控制梯度（梯度累积、二阶优化），封装就开始漏风。此时 Lightning 让你覆写 `training_step` 与 `configure_optimizers`，Keras 让你写 `train_step`——**封装逐级让位**。高层 API 并非取代低层，而是"默认走高层，遇到例外往下钻"。

---

## 六、如何选择抽象层

判断依据不是"深度学习 vs 机器学习"，而是**创新点是否位于训练流程之中**：

- 创新点在**数据或结构**，训练流程标准 → 使用 `fit` 式 API，把节省的时间投入模型设计；
- 创新点在**训练过程** → 显式编写训练循环，高级 API 反而成为束缚。

---

## 结语

sklearn 的一行 `fit` 与 Lightning 的 `Trainer.fit`，本质上是同一个东西：**把控制流从用户手中收回**。二者的区别只在于收回的时机——前者收得早，因为算法天然固定；后者收得晚，需要等任务标准化之后才敢收。

因此，`fit` 在深度学习中的回归并不矛盾，它恰恰说明：**抽象层次是任务成熟度的函数**。任务越标准，封装越深；任务越前沿，控制权越需要留给使用者。理解这一点，比记住"哪个框架更简单"更有价值。