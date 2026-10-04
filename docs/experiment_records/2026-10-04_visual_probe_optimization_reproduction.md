# 视觉探查优化诊断重放

这是事后只读诊断，不是新增训练。只能读取已登记的两子集、冻结特征、预处理状态和 24 个已选头；原协议与代码及每个 artifact SHA 先核验。完整结果重放要求相同本地权重/缓存和 Python/PyTorch 环境。

运行下面完整代码（在研究工作树根目录）不会更新模型。结果首次保存，再次执行仅比较精确一致，不覆盖旧诊断。交叉熵与存储 softmax-log NLL 的计算路径不同，实测最大损失重放残差为机器精度量级；本诊断核验容差为 1e-12。该事后损失容差不改变原训练/特征的精确重放要求。

```python
import json, sys, math
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, "scripts")
import frozen_visual_probe as probe
from run_frozen_visual_probe import DEFAULT_OUTPUT, ROOT, configure_runtime, snapshot
from run_tc_oos_rsg_experiments import _file_sha256, _write_json, state_dict_sha256

configure_runtime()
output = DEFAULT_OUTPUT
summary_path = output / "development_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
protocol = json.loads((output / "registered_protocol.json").read_text(encoding="utf-8"))
data, source = snapshot(ROOT, protocol)
assert source == summary["source_snapshot"]
assert all(_file_sha256(output / name) == digest for name, digest in summary["artifact_sha256"].items())
assert all(_file_sha256(ROOT / "scripts" / name) == digest for name, digest in summary["code_sha256"].items())
cache = torch.load(output / "frozen_features.pt", weights_only=True, map_location="cpu")
state = torch.load(output / "preprocessing.pt", weights_only=True, map_location="cpu")
blocks = probe.apply_preprocessing(data[0]["evidence"], cache["fit"].double(), state)
y = data[0]["labels"]
diagnostic_lambda = 0.0001
runs = []
for registered in summary["runs"]:
    architecture, arm, seed = registered["architecture"], registered["arm"], registered["seed"]
    folder = output / f"{architecture}_{arm}_seed{seed}"
    head = probe.build_classifier(82, architecture, protocol["budget"])
    head.load_state_dict(torch.load(folder / "best_head.pt", weights_only=True), strict=True)
    head.eval()
    before = state_dict_sha256(head.state_dict())
    assert before == registered["state_sha256"]
    x = probe.make_input(blocks, arm, seed+100000)
    logits = head(x)
    nll = F.cross_entropy(logits, y)
    parameters = tuple(head.parameters())
    gradients = torch.autograd.grad(nll, parameters)
    g = torch.cat([value.detach().reshape(-1) for value in gradients])
    theta = torch.cat([value.detach().reshape(-1) for value in parameters])
    ridge_gradient = g + diagnostic_lambda*theta
    q = logits.detach().softmax(1)
    derivative = float(((q*logits.detach()).sum(1)-logits.detach()[torch.arange(len(y)), y]).mean())
    eps = 1e-6
    finite_difference = float((F.cross_entropy(logits.detach()*(1+eps), y)-F.cross_entropy(logits.detach()*(1-eps), y))/(2*eps))
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    loss_replay_residual = abs(nll.item()-history[registered["best_epoch"]-1]["fit_nll"])
    assert loss_replay_residual <= 1e-12
    row = {"architecture": architecture, "arm": arm, "seed": seed, "selected_epoch": registered["best_epoch"],
           "fit_nll": float(nll.detach()), "loss_replay_residual": loss_replay_residual, "data_gradient_max_abs": float(g.abs().max()), "data_gradient_l2": float(g.norm()),
           "explicit_ridge_diagnostic_gradient_max_abs": float(ridge_gradient.abs().max()),
           "logit_scale_derivative_at_1": derivative, "logit_scale_finite_difference": finite_difference,
           "logit_scale_derivative_residual": abs(derivative-finite_difference),
           "registered_fit_temperature_beta": registered["calibration"]["beta"],
           "fit_nll_epoch25_minus_epoch30": history[24]["fit_nll"]-history[29]["fit_nll"],
           "stop_nll_epoch25_minus_epoch30": history[24]["stop_nll"]-history[29]["stop_nll"]}
    if architecture == "linear":
        augmented = torch.cat([x, torch.ones(len(x),1,dtype=x.dtype)],1)
        smoothness = float(torch.linalg.matrix_norm(augmented, ord=2).square()/(2*len(x)))+diagnostic_lambda
        gradient_square = float(ridge_gradient.square().sum())
        row.update(explicit_ridge_diagnostic_value=float(nll.detach()+diagnostic_lambda/2*theta.square().sum()),
                   smoothness_upper_bound=smoothness,
                   explicit_ridge_suboptimality_lower_bound=gradient_square/(2*smoothness),
                   explicit_ridge_suboptimality_upper_bound=gradient_square/(2*diagnostic_lambda))
    assert before == state_dict_sha256(head.state_dict())
    runs.append(row)
assert max(r["logit_scale_derivative_residual"] for r in runs) < 1e-8
result = {"status": "READ_ONLY_OPTIMIZATION_DIAGNOSTIC",
          "reference_summary_sha256": _file_sha256(summary_path),
          "source_snapshot": source, "runs": runs,
          "diagnostic_lambda": diagnostic_lambda,
          "objective_boundary": "Explicit L2 ridge objective is a post-hoc diagnostic, NOT the original AdamW objective. No optimizer step, no new fitted model, no new image reads.",
          "new_training": False, "new_images_loaded": False, "original_heads_unchanged": True,
          "formal_report_unchanged": True, "outer_manifest_read": False,
          "claim_boundary": "Fit gradients and convex linear diagnostic gaps cannot prove independent generalization or a visual information advantage. No MLP global optimum certificate."}
path = ROOT / "outputs/theory/abmp_visual_probe_optimization_v1/diagnostic_summary.json"
if path.exists():
    assert json.loads(path.read_text(encoding="utf-8")) == result
    print("EXACT_REPLAY_VERIFIED")
else:
    _write_json(path, result)
print("HEADS",len(runs))
print("GRAD_MAX_RANGE", min(r["data_gradient_max_abs"] for r in runs), max(r["data_gradient_max_abs"] for r in runs))
print("SCALE_DERIV_RANGE", min(r["logit_scale_derivative_at_1"] for r in runs), max(r["logit_scale_derivative_at_1"] for r in runs))
print("NEGATIVE_SCALE_DERIVATIVES", sum(r["logit_scale_derivative_at_1"]<0 for r in runs))
print("FIVE_EPOCH_FIT_IMPROVEMENTS", sum(r["fit_nll_epoch25_minus_epoch30"]>0 for r in runs))
print("FIVE_EPOCH_STOP_IMPROVEMENTS", sum(r["stop_nll_epoch25_minus_epoch30"]>0 for r in runs))
print("LINEAR_GAP_LOWER_RANGE",min(r["explicit_ridge_suboptimality_lower_bound"] for r in runs if r["architecture"]=="linear"),max(r["explicit_ridge_suboptimality_lower_bound"] for r in runs if r["architecture"]=="linear"))
print("DERIVATIVE_MAX_RESIDUAL",max(r["logit_scale_derivative_residual"] for r in runs))
for r in runs:
    if r["arm"] in ("E","EH"):
        print(r["architecture"], r["arm"], r["seed"], "gradient",r["data_gradient_max_abs"],"scale_derivative",r["logit_scale_derivative_at_1"],"fit_last5",r["fit_nll_epoch25_minus_epoch30"],"stop_last5",r["stop_nll_epoch25_minus_epoch30"])
```
