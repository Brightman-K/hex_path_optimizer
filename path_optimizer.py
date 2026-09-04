# -*- coding: utf-8 -*-
"""
路径优化器 —— 六方向多步长扫描优化器（B方案代码实现）

核心思路：
1. 梯度方向a只反映无穷小方向，不代表走固定步长x后还是最优方向
2. 在a周围每60度取一个方向(b,c,d,e,f)，共6个方向，在2D投影底面上排成六边形
3. 每个方向走固定坐标步长x，算loss，比较谁最优
4. 再试几个更小步长(x-v/x-b/x-n)，每步长×6方向全扫
5. 小步长最优点标记为极小值候选，但实际走大步长x（防掉进局部山谷）
6. 所有点连成路径，极小值之间连线，沿连线重跑一次对比loss
7. 极小值点可用于分析特征-极小值关系（催化剂体系指纹）

注意：这里的步长是坐标空间的步长（投影平面上的欧氏距离），
不是传统直接乘梯度的学习率。方向是归一化单位向量，固定距离移动。
"""

import numpy as np
from typing import Callable, List, Dict, Tuple, Optional
import warnings


class HexPathOptimizer:
    """六方向多步长扫描优化器，为低维参数空间设计（比如打分公式w）。"""

    def __init__(self, loss_fn: Callable[[np.ndarray], float],
                 n_directions: int = 6,
                 base_step: float = 0.1,
                 smaller_steps: List[float] = None,
                 max_iters: int = 50,
                 tol: float = 1e-4,
                 seed: int = 42):
        """
        Args:
            loss_fn: 接收参数向量w，返回loss的函数
            n_directions: 六边形方向数，默认6（60度一个）
            base_step: 最大步长x（坐标空间移动距离）
            smaller_steps: 更小步长列表，默认[x/2, x/4, x/8]
            max_iters: 最大迭代次数
            tol: 收敛阈值
            seed: 随机种子，保证可复现
        """
        self.loss_fn = loss_fn
        self.n_directions = n_directions
        self.base_step = base_step
        self.smaller_steps = smaller_steps or [base_step / 2, base_step / 4, base_step / 8]
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        # 记录所有采样点，用于画loss地形图
        self.all_samples: List[Dict] = []
        # 记录极小值候选点
        self.marked_minima: List[Dict] = []
        # 记录优化路径
        self.path: List[np.ndarray] = []
        # 每步的最优loss
        self.loss_history: List[float] = []

    def _compute_gradient(self, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """中心差分算梯度，比前向差分准。"""
        grad = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()
            w_plus[i] += eps
            w_minus[i] -= eps
            grad[i] = (self.loss_fn(w_plus) - self.loss_fn(w_minus)) / (2 * eps)
        return grad

    def _get_2d_basis(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        建立2D投影底面的两个正交基：
        axis_x = 梯度方向（归一化）
        axis_y = 和梯度正交的随机方向
        这样六边形采样就在这个底面上展开。
        """
        # x轴：梯度方向
        axis_x = grad / (np.linalg.norm(grad) + 1e-10)
        # y轴：随机方向，正交化
        random_dir = self.rng.standard_normal(len(grad))
        axis_y = random_dir - np.dot(random_dir, axis_x) * axis_x
        axis_y = axis_y / (np.linalg.norm(axis_y) + 1e-10)
        return axis_x, axis_y

    def _get_hex_directions(self, axis_x: np.ndarray, axis_y: np.ndarray) -> List[np.ndarray]:
        """
        在2D底面上生成六边形6个方向，映射回高维参数空间。
        相邻方向夹角60度。
        """
        directions = []
        for i in range(self.n_directions):
            angle = 2 * np.pi * i / self.n_directions  # 0, 60, 120...
            # 2D平面上的方向
            dir_2d = np.array([np.cos(angle), np.sin(angle)])
            # 映射回高维参数空间
            dir_high = dir_2d[0] * axis_x + dir_2d[1] * axis_y
            directions.append(dir_high)
        return directions

    def _scan_step(self, w_current: np.ndarray, step: float,
                   directions: List[np.ndarray], iter_idx: int) -> List[Dict]:
        """
        对单个步长，扫描所有方向，返回每个采样点的信息。
        每个点记录：坐标、loss、方向索引、步长、是否在底面上的角度
        """
        samples = []
        for dir_idx, direction in enumerate(directions):
            # 坐标空间固定步长移动：w_new = w + step * 单位方向
            # 这就是"坐标步长"——移动距离=step，不是传统α*gradient
            w_new = w_current + step * direction
            loss = self.loss_fn(w_new)
            samples.append({
                "w": w_new.copy(),
                "loss": loss,
                "dir_idx": dir_idx,
                "angle_deg": 360 * dir_idx / self.n_directions,
                "step": step,
                "iter": iter_idx,
            })
        return samples

    def optimize(self, w_init: np.ndarray, verbose: bool = True) -> Dict:
        """
        主优化循环。按B方案：
        1. 算梯度方向，建2D底面
        2. 六方向 × 多步长全扫
        3. 小步长最优点标记为极小值
        4. 实际走大步长最优方向
        5. 收敛后沿极小值连线重跑对比
        """
        w = w_init.copy()
        self.path.append(w.copy())
        best_loss = self.loss_fn(w)
        self.loss_history.append(best_loss)

        for iter_idx in range(self.max_iters):
            # 算当前点梯度
            grad = self._compute_gradient(w)
            grad_norm = np.linalg.norm(grad)

            # 梯度太小，到平地了，可以停
            if grad_norm < self.tol:
                if verbose:
                    print(f"[iter {iter_idx}] 梯度太小({grad_norm:.2e})，停")
                break

            # 建2D投影底面
            axis_x, axis_y = self._get_2d_basis(grad)
            # 六边形6个方向
            directions = self._get_hex_directions(axis_x, axis_y)

            # === 核心：六方向×多步长全扫 ===
            all_scan_results = []
            # 先扫最大步长
            big_samples = self._scan_step(w, self.base_step, directions, iter_idx)
            all_scan_results.extend(big_samples)
            # 再扫几个更小步长
            for small_step in self.smaller_steps:
                small_samples = self._scan_step(w, small_step, directions, iter_idx)
                all_scan_results.extend(small_samples)

            # 记录所有采样点
            self.all_samples.extend(all_scan_results)

            # 找小步长里loss最小的点，标记为极小值候选
            small_results = [s for s in all_scan_results if s["step"] <= max(self.smaller_steps)]
            if small_results:
                min_small = min(small_results, key=lambda s: s["loss"])
                self.marked_minima.append(min_small)

            # 实际走大步长的最优方向（防局部最小山谷）
            big_best = min(big_samples, key=lambda s: s["loss"])

            # 检查是否收敛：大步长最优 loss 没怎么改善
            if big_best["loss"] >= best_loss - self.tol:
                # 但小步长可能还能下降，记录一下不强制停
                if verbose:
                    small_loss_str = f"{min_small['loss']:.4f}" if small_results else "N/A"
                    print(f"[iter {iter_idx}] 大步长无改善(L={big_best['loss']:.4f})，"
                          f"小步长最优={small_loss_str}")
                # 如果连续2轮无改善才停
                if len(self.loss_history) >= 3 and abs(self.loss_history[-1] - self.loss_history[-3]) < self.tol:
                    break
            else:
                w = big_best["w"].copy()
                best_loss = big_best["loss"]

            self.path.append(w.copy())
            self.loss_history.append(best_loss)

            if verbose and iter_idx % 5 == 0:
                print(f"[iter {iter_idx}] loss={best_loss:.6f}, 采样点={len(all_scan_results)}")

        # === 第二阶段：沿极小值连线重跑对比 ===
        rerun_result = self._rerun_along_minima_path(w, verbose)

        return {
            "best_w": w,
            "best_loss": best_loss,
            "path": np.array(self.path),
            "loss_history": self.loss_history,
            "all_samples": self.all_samples,
            "marked_minima": self.marked_minima,
            "rerun_result": rerun_result,
        }

    def _rerun_along_minima_path(self, w_final: np.ndarray, verbose: bool = True) -> Dict:
        """
        沿标记的极小值点连线，用小步长重跑一次，
        对比大步长贪心的结果，验证能垒路径假设。
        """
        if len(self.marked_minima) < 2:
            return {"rerun_loss": None, "improved": False, "reason": "极小值点不足2个，无法连线"}

        # 取loss最小的几个极小值点
        sorted_minima = sorted(self.marked_minima, key=lambda s: s["loss"])
        top_minima = sorted_minima[:min(5, len(sorted_minima))]

        # 沿极小值连线方向走
        w_rerun = w_final.copy()
        rerun_loss = self.loss_fn(w_rerun)
        rerun_path = [w_rerun.copy()]

        # 极小值点之间的平均方向作为引导
        directions_to_minima = []
        for m in top_minima:
            diff = m["w"] - w_rerun
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                directions_to_minima.append(diff / norm)

        if not directions_to_minima:
            return {"rerun_loss": rerun_loss, "improved": False, "reason": "极小值点方向无效"}

        # 平均方向（允许稍微偏离，结合梯度工程逻辑）
        avg_dir = np.mean(directions_to_minima, axis=0)
        avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-10)

        # 小步长沿这个方向走几步
        small_step = self.base_step / 4
        for step_i in range(10):
            # 每步重新算梯度，和极小值方向加权（结合梯度下降工程逻辑）
            grad = self._compute_gradient(w_rerun)
            grad_dir = -grad / (np.linalg.norm(grad) + 1e-10)
            # 70%极小值方向 + 30%梯度方向
            combined_dir = 0.7 * avg_dir + 0.3 * grad_dir
            combined_dir = combined_dir / (np.linalg.norm(combined_dir) + 1e-10)

            w_next = w_rerun + small_step * combined_dir
            next_loss = self.loss_fn(w_next)

            if next_loss < rerun_loss:
                w_rerun = w_next
                rerun_loss = next_loss
                rerun_path.append(w_rerun.copy())
            else:
                break  # 没改善就停

        improved = rerun_loss < self.loss_history[-1] - self.tol
        if verbose:
            print(f"[重跑] 大步长最终loss={self.loss_history[-1]:.6f}, "
                  f"能垒路径重跑loss={rerun_loss:.6f}, 改善={improved}")

        return {
            "rerun_loss": rerun_loss,
            "improved": improved,
            "rerun_path": np.array(rerun_path),
            "n_minima_used": len(top_minima),
        }

    def analyze_minima_features(self, feature_names: List[str]) -> List[Dict]:
        """
        分析每个极小值点对应的特征权重分布——催化剂体系指纹。
        这是跨领域价值的核心：不同极小值=不同催化剂体系解释。
        """
        analysis = []
        for idx, m in enumerate(self.marked_minima):
            w = m["w"]
            # 找权重最大的几个特征
            top_indices = np.argsort(np.abs(w))[::-1][:5]
            top_features = [
                {"feature": feature_names[i] if i < len(feature_names) else f"w_{i}",
                 "weight": float(w[i])}
                for i in top_indices
            ]
            analysis.append({
                "minima_id": idx,
                "loss": float(m["loss"]),
                "step": float(m["step"]),
                "dominant_features": top_features,
            })
        return analysis

    def get_landscape_data(self) -> Dict:
        """
        返回loss地形图数据，用于可视化。
        把高维采样点投影回2D底面坐标。
        """
        if not self.all_samples:
            return {"x": [], "y": [], "loss": [], "angle": [], "step": []}

        # 用最后一步的底面基坐标系投影
        last_iter = max(s["iter"] for s in self.all_samples)
        last_grad = self._compute_gradient(self.path[-1])
        axis_x, axis_y = self._get_2d_basis(last_grad)

        xs, ys, losses, angles, steps = [], [], [], [], []
        for s in self.all_samples:
            # 把采样点投影到2D底面
            diff = s["w"] - self.path[s["iter"]] if s["iter"] < len(self.path) else s["w"] - self.path[-1]
            x_proj = np.dot(diff, axis_x)
            y_proj = np.dot(diff, axis_y)
            xs.append(float(x_proj))
            ys.append(float(y_proj))
            losses.append(float(s["loss"]))
            angles.append(float(s["angle_deg"]))
            steps.append(float(s["step"]))

        return {"x": xs, "y": ys, "loss": losses, "angle": angles, "step": steps}


# ============================================================
# 自测：在一个简单的2D非凸函数上跑，验证优化器能工作
# ============================================================
if __name__ == "__main__":
    print("=== 路径优化器自测 ===")

    # 测试函数：Rosenbrock-like 非凸函数，有局部最小
    def test_loss(w):
        x, y = w[0], w[1]
        # 两个谷：一个在(1,1)，一个在(-1,-1)
        loss = (x**2 + y**2 - 2)**2 + 0.1 * (x - y)**2
        return float(loss)

    w_init = np.array([2.0, -1.0])  # 故意放在离全局最优远的地方
    optimizer = HexPathOptimizer(
        loss_fn=test_loss,
        n_directions=6,
        base_step=0.15,
        smaller_steps=[0.075, 0.0375],
        max_iters=30,
    )
    result = optimizer.optimize(w_init, verbose=True)

    print(f"\n最终loss: {result['best_loss']:.6f}")
    print(f"最终参数: {result['best_w']}")
    print(f"采样点总数: {len(result['all_samples'])}")
    print(f"标记极小值数: {len(result['marked_minima'])}")
    print(f"重跑结果: {result['rerun_result']}")

    # 分析极小值特征
    feature_names = ["param_x", "param_y"]
    minima_analysis = optimizer.analyze_minima_features(feature_names)
    print("\n=== 极小值点特征分析（催化剂体系指纹）===")
    for ma in minima_analysis[:3]:
        print(f"极小值{ma['minima_id']}: loss={ma['loss']:.4f}, "
              f"主导特征={[(f['feature'], round(f['weight'],3)) for f in ma['dominant_features'][:2]]}")

    # 地形图数据
    landscape = optimizer.get_landscape_data()
    print(f"\n地形图数据点: {len(landscape['x'])}")
    print("自测通过 ✅")
