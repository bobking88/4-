# 18 维证据的后验可重建性审计

## Material Passport

- 类型：事后只读代数与数值核验，不训练新模型。
- 来源：已登记的 `abmp_frozen_visual_probe_v1` 两个 Fold 0 内层子集，共 680 行，以及原拟合侧标准化状态。
- Verification Status: VERIFIED，完整只读计算已重放，保存 JSON 与重放结果精确一致。
- 输出：`outputs/theory/abmp_evidence_reconstruction_v1/diagnostic_summary.json`。
- 结果 SHA-256：`6d3edf080937bbcc54a3b870e0fd175fa8e766811b7eddb9114e01a46310ca7a`。
- 边界：不加载图片或视觉特征矩阵、不读 outer/Fold 1/2、不更新头或骨干、不修改正式报告。

## 1. 需要修正的解释

原记录为保证区间端点可测，将总条件状态写为 `(E18,p,L)`，并提醒有限 E18 头与理论 oracle 不同。代码检查表明：**当前固定校验配置下，p 和 L 本来就是 E18 的确定函数**。因此总条件状态的写法可以保留，但不能暗示额外拼接 p、L 提供了 E18 缺失的统计信息。

`build_projection_evidence` 的零起始列序为：

| 列 | 内容 |
|---|---|
| 0:4 | 固定融合校验后的四类概率 |
| 4:8 | OOS 路由校验后的四类概率 q |
| 8:12 | 两个四类后验的绝对差 |
| 12 | OOS 候选门 |
| 13、14 | 含钛、金属校验的目标概率 |
| 15、16、17 | 两个后验熵与 JS 散度 |

实际加载器同时返回 OOS 校验前 p、矛盾量 L；这并不表示它们独立于 E18。旧线性/MLP 从 E18 学得较差，只能说明当前有限表示、函数类、预算和优化表现，不证明原后验信息被 E18 删除，更不证明 H 的真实条件互信息为正。

## 2. 解析逆映射

设固定校验参数为阈值 1/2、两强度均 1，E18 中的校验输出为 v_Ti、v_M。定义

$$
L(E)=[1-2v_{Ti}]_+ + [1-2v_M]_+,
\qquad s(E)=e^{-L(E)}.
$$

由于两个 v 都在 [0,1]，有 `0<=L<=2`、`exp(-2)<=s<=1`。给定原概率 p，当前校验输出为

$$
q_T=\frac{s p_T}{1-(1-s)p_T},\qquad
q_j=\frac{p_j}{1-(1-s)p_T}\quad(j\ne T).
$$

因此令 `d=s+(1-s)q_T`，即可逐行恢复

$$
\boxed{p_T=\frac{q_T}{d},\qquad
p_j=\frac{s q_j}{d}\ (j\ne T).}
$$

证明：第一式交叉相乘给出 `q_T=p_T[s+(1-s)q_T]`；再代回归一化分母得 `1-(1-s)p_T=s/d`，于是得到非目标逆式。逆后验和为 `(q_T+s(1-q_T))/d=1`。正尺度保证 d>0；当前正向分母至少 exp(-2)，代码中的 epsilon 下限不激活。

这不是一般意义上的新可逆网络理论，而是当前确定校验的显式逆映射。若改为 s=0、只保留 argmax、删除校验输出、量化或未知参数，结论不自动成立。

## 3. 信息与有限网络的区别

在固定配置、精确实数公式下，`p=f(E18)`、`L=g(E18)`，所以

$$
\sigma(E18,p,L)=\sigma(E18),
\qquad I(Y;H\mid E18,p,L)=I(Y;H\mid E18).
$$

这里是确定函数的条件信息等价关系，不是从 680 行估计出的互信息。实际 CSV 与浮点逆映射只做数值容差核验，不声称有限精度数据完全可逆。

视觉探查对 E18 只做逐列标准化 `E_std=(E18-mu)/a`，每列 a>=1e-8，不做 E 侧 PCA 或删列；故精确实数层面可以由 `a*E_std+mu` 恢复 E18。填零 H 的 E 组也没有丢掉这些 E 列。

但是，有限线性头或有限 MLP 不保证精确实现包含指数、乘法和除法的逆映射。从均匀初始化重新学习后验与直接保留 `log(p)` 路径，虽不改变可获得的信息，仍改变函数族、优化起点与归纳偏置。下一轮锚点对照应明确定位为**确定性重参数化和强对照**，不能把显式 p/L 当成新的感知证据，也不能把普通残差校准本身包装为创新。

## 4. 下一步实验的约束

本核验不启动待批准实验。下一轮若获批准，应分别报告信息条件、输入表示、有效参数和优化收敛；把同一原后验锚点给 E 与 E+H 两组，而不是让视觉组独占强基线。H 的价值仍需超过同信息、可靠求解的概率对照；角色网络的价值还要超过同 H 的普通残差并通过干扰误入约束及后续独立验证。

不能因为本次解析重建就撤销原探查的全部输出、修改其预登记筛查或把原始配对下降改称没有发生。应保留观察，同时修正其理论解释。主线 B 的网络创新目标尚未验证，C 仍是后续方向。

实际核验结果：拟合/停止各 340 行，p 的最大重建残差分别为 3.33067e-16 和 2.22045e-16，L 的重建残差均为 0；正向往返、概率单纯形和标准化逆变换残差均不超过 2.22045e-16，epsilon 下限均未激活。10000 个随机合成单纯形与 4 个顶点的最大逆映射残差为 2.22045e-16。输入、已登记代码、标准化状态与正式报告摘要已核验，前后源快照一致。代码没有使用标签做推断，也没有优化器步骤。

## 5. 完整只读重放代码

在研究工作树根目录执行。已有结果时仅精确比较，不覆盖。读取标准化状态是为了核验 E 的逆变换，不加载 `frozen_features.pt`、原图片或任何待训练新头。

```python
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, "scripts")
from run_frozen_visual_probe import DEFAULT_OUTPUT, ROOT, configure_runtime, snapshot
from run_tc_oos_rsg_experiments import _file_sha256, _write_json
from hrgv_network import apply_residual_target_verifiers

configure_runtime()
summary_path = DEFAULT_OUTPUT / "development_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
protocol = json.loads((DEFAULT_OUTPUT / "registered_protocol.json").read_text(encoding="utf-8"))
data, source = snapshot(ROOT, protocol)
assert source == summary["source_snapshot"]
assert all(_file_sha256(ROOT / "scripts" / name) == digest for name, digest in summary["code_sha256"].items())
assert _file_sha256(DEFAULT_OUTPUT / "preprocessing.pt") == summary["artifact_sha256"]["preprocessing.pt"]
state = torch.load(DEFAULT_OUTPUT / "preprocessing.pt", weights_only=True, map_location="cpu")
assert bool((state["e_scale"] >= 1e-8).all())

def inverse_verified(q, scale):
    denominator = scale + (1-scale)*q[:, :1]
    assert bool((denominator > 0).all())
    return torch.cat([q[:, :1]/denominator, scale*q[:, 1:]/denominator], 1)

rows = []
for name, subset in zip(protocol["subsets"], data, strict=True):
    e = subset["evidence"]
    q = e[:, 4:8]
    contradiction = torch.relu(1-2*e[:, 13:14]) + torch.relu(1-2*e[:, 14:15])
    scale = (-contradiction).exp()
    p = inverse_verified(q, scale)
    standardized = (e-state["e_mean"])/state["e_scale"]
    recovered_e = standardized*state["e_scale"]+state["e_mean"]
    replay = apply_residual_target_verifiers(p, e[:, 13:14], e[:, 14:15])
    result = {"subset": name, "rows": len(e), "evidence_dimensions": e.shape[1],
              "p_reconstruction_max_abs_residual": float((p-subset["p"]).abs().max()),
              "L_reconstruction_max_abs_residual": float((contradiction-subset["contradiction"]).abs().max()),
              "verification_roundtrip_max_abs_residual": float((replay-q).abs().max()),
              "standardization_inverse_max_abs_residual": float((recovered_e-e).abs().max()),
              "simplex_max_abs_residual": float((p.sum(1)-1).abs().max()),
              "scale_min": float(scale.min()), "scale_max": float(scale.max()),
              "epsilon_clamp_active_count": int((1-(1-scale)*subset["p"][:, :1] < torch.finfo(e.dtype).eps).sum())}
    assert all(result[key] <= 1e-12 for key in result if key.endswith("residual"))
    assert result["epsilon_clamp_active_count"] == 0
    rows.append(result)

generator = torch.Generator().manual_seed(20261004)
synthetic_p = torch.cat([torch.softmax(torch.randn(10000,4,generator=generator,dtype=torch.float64),1), torch.eye(4,dtype=torch.float64)],0)
v = torch.rand(len(synthetic_p),2,generator=generator,dtype=torch.float64)
scale = (-(torch.relu(1-2*v[:, :1])+torch.relu(1-2*v[:, 1:]))).exp()
synthetic_q = apply_residual_target_verifiers(synthetic_p, v[:, :1], v[:, 1:])
synthetic_residual = float((inverse_verified(synthetic_q,scale)-synthetic_p).abs().max())
assert synthetic_residual <= 1e-12
_, source_after = snapshot(ROOT, protocol)
assert source_after == source
result = {"status": "READ_ONLY_EVIDENCE_RECONSTRUCTION_AUDIT", "reference_summary_sha256": _file_sha256(summary_path),
          "source_snapshot": source, "subsets": rows, "synthetic_rows": len(synthetic_p),
          "synthetic_seed": 20261004, "synthetic_max_abs_residual": synthetic_residual,
          "numeric_tolerance": 1e-12, "new_training": False, "new_images_loaded": False,
          "visual_feature_matrix_loaded": False, "formal_report_unchanged": True, "outer_manifest_read": False,
          "interpretation": "With fixed positive verifier scale, p and L are deterministic functions of E18 in exact arithmetic. Explicit posterior anchor changes representation/optimization, not statistical information; finite-precision residuals are reported. No conditional MI estimate or network novelty proof."}
path = ROOT / "outputs/theory/abmp_evidence_reconstruction_v1/diagnostic_summary.json"
if path.exists():
    assert json.loads(path.read_text(encoding="utf-8")) == result
    print("EXACT_REPLAY_VERIFIED")
else:
    _write_json(path, result)
for row in rows:
    print(row)
print("SYNTHETIC_RESIDUAL", synthetic_residual)
```
