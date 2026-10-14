# IC-RFA-B v1 的 v2 真实机制实验：完整执行、未观察到主点角色净纠错

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent run + saved-result descriptive validation。
- Origin Date: 2026-10-14，Asia/Shanghai；Version Label: `ic_role_mechanism_v2_results`。
- Verification Status: `REAL_EXECUTION_AND_SAVED_ACCEPTANCE_VERIFIED_NO_PRIMARY_ROLE_DECISION_INCREMENT`。
- 人类批准项：`["request_user_input_async","call_RHuePD3CQpoUvp4t88mzQnSM",0]`，回答“批准一次固定协议的真实机制实验”。
- 运行授权登记：`3dd2fd985fd3de938e0f61418547b5cce373e120`；机器协议SHA-256：`332dd67fec71b123a0b88eea1d0c255e34af11d251f9f48f672065d78679852c`。
- 源码仍为修复快照da176fc；网络/损失/矩阵不改，模型仍为IC-RFA-B v1。本记录不是正式技术报告修订、独立测试或数学原创认证。

## 1. 执行与接受证据

北京时间2026-10-14 **10:38:48 至 11:17:07**，单次入口累计 **2299秒（38分19秒）**，低于7200秒上限。原生工具session81026最终退出码0，parent PID34924。无自动重试、续跑、扩网格、重新计时探查或空闲验收。

| 项目 | 已保存证据 |
| --- | --- |
| 总体终态 | [whole_experiment.json](../../outputs/training/ic_role_mechanism_v2/development_fold_0/whole_experiment.json)：COMPLETE，无error |
| 训练阶段 | [stage_state.json](../../outputs/training/ic_role_mechanism_v2/development_fold_0/stage_state.json)：MECHANISM_COMPLETE，108条轨迹、3次T0、112个阶段任务 |
| 完整固定检查点 | 每轨迹0/1/2/5/20/50/100/200/400，共972个保存检查点通过公开核验 |
| 最终接受 | [final_acceptance/evidence.json](../../outputs/training/ic_role_mechanism_v2/development_fold_0/final_acceptance/evidence.json)：delivery_verified=true、theory_violation_count=0 |
| 临时分析 | [analysis/evidence.json](../../outputs/training/ic_role_mechanism_v2/development_fold_0/analysis/evidence.json)：delivery_verified=false，不作为最终接受；metrics/comparisons/diagnostics与最终文件逐项相同 |
| 监督与释放 | 112个阶段任务和最终公开验收均COMPLETE，监督记录alive=false、请求释放；阶段父级和最终父级释放证据完整 |

最终公开验收只进行一次，在现有监督器的独立子进程内、受从入口起剩余7200秒预算约束；没有新增优化、PCA/T0拟合或轨迹。保存产物接受后，只用标准库读取JSON做算术汇总，不再次调用公开验证器或模型前向。

## 2. 主问题的直接答案

冻结主点为100步，首要比较C-S-R相对同修复、同参照及同预算的C-S-U。每个种子的三内折合并成340条既有fit OOF记录，目标/Ti/G/M分母分别为66/130/97/47。

| seed | 改变预测数/340 | 目标新增 N+ | 目标损失 N- | 净纠错 | NLL(R)-NLL(U) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261002 | 0 | 0 | 0 | 0 | -0.000007107725 |
| 20261003 | 0 | 0 | 0 | 0 | -0.000023422106 |
| 20261004 | 0 | 0 | 0 | 0 | -0.000014814126 |

三个种子的340张图像逐图argmax均相同，不只是总正确数碰巧相等。因此目标召回、Ti/G/M误入目标数、Accuracy、Macro F1和混淆矩阵在首要配对中均相同。平均NLL差为 **-0.000015114652**，只表示概率值存在微小变化；没有观察到主点决策纠错收益，不作显著性或选矿收益宣称。

两者目标正确数均为37/66（召回56.06%）；Ti误入目标为17/13/14，G均为8，M均为9，种子顺序如上。目标新增/损失相对T0锚点均为1/1、净值0；这与本节R/U配对的0/0不同，不能混淆参照对象。

配对定义保留：
\[
N_+=\sum_{i:y_i=T}\mathbf1[\hat y_i^R=T,\hat y_i^U\ne T],\quad
N_-=\sum_{i:y_i=T}\mathbf1[\hat y_i^R\ne T,\hat y_i^U=T],\quad
\Delta N_T=N_+-N_-.
\]

## 3. 全部控制，不择优展示

下表均为固定100步，三个种子的描述性均值；目标正确数按20261002/03/04排列。它们不是独立测试集或工业分选指标，不能直接替换此前1284张测试集的基线结果。

| 单元 | NLL均值 | Accuracy % | Macro F1 % | 目标正确数/66 |
| --- | ---: | ---: | ---: | --- |
| Z-O-U | 0.798849 | 68.04 | 65.86 | 37/37/37 |
| Z-O-R | 0.798849 | 68.04 | 65.86 | 37/37/37 |
| Z-S-U | 0.799095 | 68.14 | 65.97 | 37/38/37 |
| Z-S-R | 0.799101 | 68.14 | 65.97 | 37/38/37 |
| C-O-U | 0.798573 | 67.84 | 65.69 | 37/37/37 |
| C-O-R | 0.798569 | 67.84 | 65.69 | 37/37/37 |
| C-S-U | 0.798749 | 68.04 | 65.84 | 37/37/37 |
| C-S-R | 0.798734 | 68.04 | 65.84 | 37/37/37 |
| HC-S | 0.798749 | 68.04 | 65.84 | 37/37/37 |
| FC-S | 0.797982 | 67.65 | 65.35 | 36/36/34 |
| C-S-U-noaux | 0.798655 | 67.84 | 65.69 | 37/37/37 |
| C-S-R-noaux | 0.798633 | 67.84 | 65.69 | 37/37/37 |

C-S-R与HC-S在三个种子的逐图决策也均相同；HC-S只有1476可训练标量及1476函数参照，C-S-R为3024+3024。不能只引用比宽头FC-S多1/1/3个目标正确的结果来证明角色增量，因为同修复均匀路由和普通读出已得到同样目标决策；FC-S的NLL比较也不是三个种子一致方向。

noaux R/U在主点同样没有任何预测变化或目标净纠错。未把400步或其他较好检查点改成主点；400是预算，不是收敛或全局最优证明。旧Z/C比较属于整块参数化变化，不能单独归因于初始化。全部逐种子、配对和机制摘要分别见：

- [36组指标](../../outputs/training/ic_role_mechanism_v2/development_fold_0/summary/unit_metrics.csv)：目标Precision/Recall/F1、NLL、Brier、ECE、全部原分母及锚点变化。
- [12单元均值及样本标准差](../../outputs/training/ic_role_mechanism_v2/development_fold_0/summary/unit_summary.csv)：标准差只描述优化种子变化，不当作独立样本不确定性。
- [21组固定配对](../../outputs/training/ic_role_mechanism_v2/development_fold_0/summary/paired_summary.csv)：逐图改变数、目标新增/损失/净值及Ti/G/M误入新增/移除。
- [108轨迹机制摘要](../../outputs/training/ic_role_mechanism_v2/development_fold_0/summary/mechanism_summary.csv)及[派生摘要身份](../../outputs/training/ic_role_mechanism_v2/development_fold_0/summary/result_summary.json)。

## 4. 理论机制证据及不能推导的结论

P1的中心化关系仍是
\[
\delta(h;\theta_0)=0,\qquad
\left.\partial_\theta\delta(h;\theta)\right|_{\theta_0}
=\frac1{16}\left.\partial_\theta A(h;\theta)\right|_{\theta_0}.
\]
冻结参照不求导，因此函数值恒等不必使参数微分为零。真实核验检查全部起点锚点一致性（绝对容忍度1e-12）、参照冻结/无别名/哈希，以及前五次数据/正则/总梯度的完整保存来源。C-S-R初始整体数据梯度范数范围0.029792--0.058977；这不是每个参数块均有非零损失梯度的保证。Z组的整体读出数据梯度也非零，不能写成“旧网络完全不能学习”。

P2的同权重关闭适配比较为
\[
|\log q_y-\log q_y^{\rm off}|\leq
(\bar\alpha+\bar\beta)\varepsilon_f L_Fb^2.
\]
972检查点的保存/公开数值核验没有界违规。C-S-R主点的内折验证保护证书比例范围61.40%--70.80%；这是同权重off的裕量保护，不是正确率或目标召回，也不等同于T0锚点或另训U。单分支移除的主点最大log概率漂移范围0.000556--0.002050；有非零扰动不代表有纠错。

P3只限共同零适配起点的标量分支块：
\[
J_{\rm main}=GJ_{\rm aux},\qquad K_{\rm main}=GK_{\rm aux}G.
\]
C-S-R的Gram关系最大残差为4.879e-19，初始主分支数值秩139--175、辅助分支223--227；C-S-U对应主秩155--182。数值秩采用固定绝对阈值1e-10，不是精确代数秩，不能由此宣称角色路由提高了秩或全局收敛。Z-S-U的相应初始分支核为零；C取消了这一特定起点阻断，但主点仍没有角色净纠错。

**因此本轮支持“已登记结构关系与实现的有限数值一致性”，不支持“原创理论已获独立证明”或“理论性质已转化为识别收益”。** 中心化、L2-SP、瓶颈及Gram同余均已有前例；独立工程复查也未审核数学新颖性。可证伪分析框架可以保留，候选效果必须保持负结果。

## 5. 描述性解释的11项边界检查

| 风险 | 本次检查/保留限制 |
| --- | --- |
| Simpson反转 | 首要配对逐种子、逐图及各类别决策均相同；未做摄影者/来源分层，不能外推分层泛化 |
| 生态谬误 | 只描述当前图像及代理类别，不推导整矿石/生产线效果 |
| Berkson/选择偏倚 | 340fit是已筛选且已暴露样本，保留选择偏倚及外推限制 |
| Collider控制偏倚 | 未新增事后控制变量；Z/C整块变化仍不能拆成单因素因果 |
| 基率忽略 | 同时保留66/130/97/47分母及目标Precision/Recall，不将类别Precision当精矿品位 |
| 均值回归 | 不用锚点错分改善的叙述替代同修复R/U对照；无主点增量 |
| 幸存者偏倚 | 本轮既定108轨迹均完成，旧v1失败与旧负结果不删除 |
| 多重寻找 | 全12单元、7配对x3seed保留，不挑宽头、seed或较好点；不做推断检验 |
| 分叉路径 | 100主点/矩阵/分析预登记，不事后换主点、扩网格或续跑 |
| 相关到因果 | 初始核非零、梯度较大和微小NLL差不证明纠错或泛化因果 |
| 反向因果 | 工程介入顺序固定；矿物学、工业过程和资源回收的因果解释不在本数据范围 |

11/11按可用证据检查；“保留限制/未检验”不是已排除风险。未计算p值、显著性或独立样本置信区间，三seed/重复内折不作独立确认。

## 6. 决定、保护与可复现归档

按预登记规则，**停止IC-RFA-B v1的效果晋级**：不读stop、不全fit重训、不扩网格、不追加新数据/种子寻求收益，不把无主点增量写成正方法效果。不因运行成功或有数学公式就宣称论文创新已验证。后续若继续研究，应先审核这份负结果及科学解释，再另行决定理论诊断/新规格；本记录没有批准新实验、网络修改或报告整合。

本轮源码、网络、损失、矩阵、冻结规格及正式报告不改。主项目/工作树唯一正式DOCX哈希仍为 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`，六张用户既有PNG不暂存、不覆盖。

GitHub只归档本记录、导航、轻量终态/manifest、临时及最终JSON证据、CSV摘要和最终日志。111个.pt文件约5.70GB保留本地，不上传原始特征、逐状态权重或原图；仓库不因此具备无需合法本地数据的完整复现能力。执行器/内折数据合同/分析源和授权提交与既有登记相同，派生CSV仅对接受后的JSON取字段、计数和mean/stdev(ddof=1)，无额外前向、拟合或验证重放。

关键原始字节SHA：whole `fed0778d02914e273a7fd2fab381a255ceb9837224bb81c7f8c5a22c7e8189bd`；stage `47d62905fc4fe13e097f5706e2f8844b6151de045be9f1f50a05a69dad5e8b5e`；最终evidence `28104aae46f7f1d7e011c8d6ff8d4ca357ceb9d03a557264984a35b80187206b`。更多字节身份在派生摘要中保留，不覆盖原始产物。

本次单次运行许可已消费。Task8的执行和一次公开验收完成，研究决定为不晋级；正式报告、独立新颖性评价和论文创新目标仍未完成，不把这一局部交付当作整个研究目标完成。
