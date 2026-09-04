# Hex Path Optimizer — 六方向多步长扫描优化器

> A Hexagonal Multi-Step Scanning Optimizer for Non-Convex Low-Dimensional Problems

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Overview

针对**非凸低维参数空间**的优化器，核心思路来自对传统梯度下降两个问题的独立推导：

1. **梯度的局部性陷阱**：梯度只反映"无穷小位移"下的最优方向，不代表走固定步长 x 后依然是最优方向；
2. **贪心算法的能垒问题**：每步选当下最优，可能错过需要跨"能垒"才能到达的更优区域。

本优化器通过**六方向六边形扫描 + 多步长全扫 + 极小值标记 + 能垒路径重跑**来缓解上述问题。

---

##  核心思路 | Core Idea

### 1. 坐标空间固定步长（不是传统学习率）

传统梯度下降：`w_new = w_old - α * gradient`（α 是学习率，方向由梯度决定）

本优化器：`w_new = w_old + x * unit_direction`（x 是坐标空间欧氏距离，方向自己选）

方向是归一化单位向量，移动距离严格等于 x。

### 2. 六方向六边形采样

在当前点建 2D 投影底面（梯度方向为 x 轴，随机正交方向为 y 轴），生成 6 个方向：

```
          0° (梯度方向)
             ↑
        60° / \ 300°
           /     \
    120° ←●→ 240°
           \     /
       180° \ /  (反梯度)
             ↓
```

相邻方向夹角 60°，在底面上排成正六边形。每个方向都走固定坐标步长。

### 3. 多步长全扫

除了最大步长 base_step，还试几个更小的步长（默认 smaller_steps=[x/2, x/4, x/8]），每个步长都扫 6 个方向：

```
默认采样点数 = (1个大步长 + 3个小步长) × 6方向 = 24个点/每轮
自定义smaller_steps时相应变化
```

### 4. 标记极小值，但走大步长

- 小步长最优点 → 标记为**极小值候选**（可能掉进局部山谷）
- 实际下一步 → 走**大步长**的最优方向（防止局部最小）

### 5. 能垒路径重跑（证伪设计）

收敛后，取 loss 最小的 top 5 极小值点连线，沿连线 + 梯度混合方向**重跑一次**，对比两次 loss：

| 方案 | 路径 | Loss |
|---|---|---|
| 大步长贪心 | 每步选大步长最优方向 | L1 |
| 能垒路径 | 沿极小值连线 + 梯度混合 | L2 |

**如果 L2 < L1**：验证了"能垒路径假设"——跨极小值连线确实可能找到更优解。

### 6. 极小值即特征指纹

每个极小值点对应一组参数权重组合，代表一种可能的"体系解释"。分析极小值点的主导特征分布，可以探讨：
- 为什么存在这个极小值点？
- 这个极小值点下各种特征有什么分布或关系？

---

## 快速开始 | Quick Start

### 安装

```bash
git clone https://github.com/用户名/hex_path_optimizer.git
cd hex_path_optimizer
pip install -r requirements.txt
```

### 最小示例

```python
import numpy as np
from path_optimizer import HexPathOptimizer

# 定义一个非凸测试函数（双谷函数，有局部最小）
def loss_fn(w):
    x, y = w[0], w[1]
    return (x**2 + y**2 - 2)**2 + 0.1 * (x - y)**2

# 初始化优化器（loss_fn在构造函数传入）
opt = HexPathOptimizer(
    loss_fn=loss_fn,
    n_directions=6,           # 六边形方向数，默认6
    base_step=0.15,           # 最大步长x（坐标空间移动距离）
    smaller_steps=[0.075, 0.0375],  # 更小步长列表
    max_iters=30,
    tol=1e-4,
    seed=42
)

# 优化（verbose控制是否打印过程）
result = opt.optimize(w_init=np.array([2.0, -1.0]), verbose=True)

print(f"最终 loss: {result['best_loss']:.6f}")
print(f"最终参数: {result['best_w']}")
print(f"采样点总数: {len(result['all_samples'])}")
print(f"标记极小值数: {len(result['marked_minima'])}")
print(f"能垒路径重跑改善: {result['rerun_result']['improved']}")
```

### 直接运行自测

```bash
python path_optimizer.py
```

自测输出（双谷函数上，实际结果可能因机器浮点精度略有差异）：

```
=== 路径优化器自测 ===
[iter 0] loss=6.313489, 采样点=18
[iter 5] loss=0.342085, 采样点=18
[iter 10] loss=0.143727, 采样点=18
[重跑] 大步长最终loss=0.018365, 能垒路径重跑loss=0.012685, 改善=True
最终loss: 0.018365
采样点总数: 324
标记极小值数: 18
自测通过 ✅
```

---

## 项目结构 | Structure

```
hex_path_optimizer/
├── path_optimizer.py        # 核心算法（HexPathOptimizer类 + 自测）
├── DESIGN.md                # 设计思路说明（心路历程）
├── README.md                # 本文件
├── requirements.txt         # 依赖（numpy）
├── LICENSE                  # MIT License
└── .gitignore               # Git忽略规则
```

---

## API 说明 | API

### HexPathOptimizer

```python
HexPathOptimizer(
    loss_fn,                # 必传：接收参数向量w返回loss的函数
    n_directions=6,         # 六边形方向数，默认6（60度一个）
    base_step=0.1,          # 最大步长x（坐标空间移动距离）
    smaller_steps=None,     # 更小步长列表，默认[x/2, x/4, x/8]
    max_iters=50,           # 最大迭代次数
    tol=1e-4,               # 收敛阈值
    seed=42                 # 随机种子，保证可复现
)
```

### 主要方法

| 方法 | 说明 |
|---|---|
| `optimize(w_init, verbose=True)` | 主优化循环，返回结果字典 |
| `analyze_minima_features(feature_names)` | 极小值特征分析（体系指纹），feature_names必传 |
| `get_landscape_data()` | 获取 loss 地形图数据（采样点+loss值） |
| `_rerun_along_minima_path(w_final)` | 第二阶段：沿极小值连线重跑对比 |

### 返回结果字典（optimize的返回值）

```python
{
    'best_w': np.ndarray,            # 最终最优参数
    'best_loss': float,              # 最终 loss
    'path': np.ndarray,              # 大步长贪心路径
    'loss_history': List[float],     # 每轮 loss
    'all_samples': List[dict],       # 所有采样点（含方向/步长/loss/角度）
    'marked_minima': List[dict],     # 标记的极小值点
    'rerun_result': {                # 能垒路径重跑结果
        'rerun_loss': float,
        'improved': bool,
        'rerun_path': np.ndarray,
        'n_minima_used': int
    }
}
```

---

##  算法流程 | Algorithm

```
每步迭代：
┌─────────────────────────────────────────┐
│ 1. 算当前点梯度方向 a（中心差分）         │
│ 2. 建2D投影底面（a为x轴，随机正交为y轴）  │
│ 3. 生成6方向六边形（0°,60°,...,300°）     │
│ 4. 对base_step + smaller_steps分别扫6方向 │
│    （默认4个步长，共24个采样点/轮）       │
│ 5. 小步长最优点 → 标记极小值候选          │
│ 6. 大步长最优点 → 实际下一步方向          │
│ 7. 记录所有采样点                        │
└─────────────────────────────────────────┘
收敛后：
┌─────────────────────────────────────────┐
│ 8. 取top5 loss最小的极小值点连线          │
│ 9. 沿连线(70%)+梯度(30%)混合方向重跑      │
│ 10. 对比 L1(大步长) vs L2(能垒路径)      │
│ 11. 极小值特征分析 = 体系指纹             │
└─────────────────────────────────────────┘
```

---

## 适用场景 | Use Cases

- **非凸低维参数优化**（参数维度 < 20 效果最佳）
- **需要可解释优化过程**的场景（不是黑盒，全程可可视化）
- **多极小值探测**：不止要最优解，还要知道有哪些次优解及其特征
- **催化剂/材料体系分析**：极小值点 = 不同体系解释

---

## 诚实声明 | Honest Disclaimer

本优化器不是"全新发明"，各组件在已有文献中均有出现：
- 坐标空间固定步长 ≈ normalized gradient descent
- 六方向采样 ≈ 2025 Aquila Optimizer 的六边形搜索
- 多步长比较 ≈ line search / trust region

**本工作的价值在于**：
1. **完整组合**：六方向 × 多步长 × 极小值标记 × 能垒路径重跑 × loss地形图，整套组合未发现完全相似工作；
2. **可证伪设计**：通过大步长贪心 vs 能垒路径两次重跑对比，验证能垒假设；
3. **跨领域连接**：极小值点用于体系特征指纹分析。

**创新性不是本工作最关注的，独立推导和工程实用性才是。**

---

##  许可证 | License

MIT License — 详见 [LICENSE](LICENSE)

---

##  设计心路 | Design Notes

详细设计思路（独立推导过程）见 [DESIGN.md](DESIGN.md)。

##  其他问题 | Other issues
如使用，请遵照引用规则，详见Design.md

---

##  贡献 | Contributing

欢迎 Issue 和 PR。如果发现算法问题或有改进建议，欢迎留言讨论。

---

*本项目为个人学习实践中的思考性工作。如有高度相似或完全撞车的早于本项目的工作，欢迎告知，将及时补充引用说明。*
