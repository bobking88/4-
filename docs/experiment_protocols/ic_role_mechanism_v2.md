# IC-RFA-B v1 修复后机制实验：v2 执行预登记

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Origin Mode: plan / preregistration。
- Origin Date: 2026-10-13；Version Label: `ic_role_mechanism_v2`（执行登记版本，网络仍为 IC-RFA-B v1）。
- Verification Status: `DESIGN_REGISTERED_REAL_UNAUTHORIZED_INDEPENDENT_REVIEW_PENDING`；GitHub 登记身份以承载本文的提交及远端回读为准。
- 人类批准仅为 **“批准新协议登记，暂不运行”**，问题项 `["request_user_input_async","call_b7aaa1e4e1764728b80858311c0bf2a6",0]`。不是新的真实运行许可。
- [机器协议](ic_role_mechanism_v2.json)为 `DRAFT_NOT_AUTHORIZED`，`human_approved=false`，运行批准身份和 UTC 均为 null。
- 独立工程复查尚未完成。作者自审和此前限定合成通过，不替代独立审核、数学查新或真实效果验证。

## 1. 本次变更与历史边界

只重新登记修复后的完整源码身份、独立协议和全新输出目录；网络、损失、输入、控制矩阵、优化器、检查点、分析和累计预算均不改。本次不读取真实输入内容或重新计算其文件哈希，不拟合、不推理真实状态、不重做计时或空闲验收、不修改正式报告。

[v1 设计](ic_role_mechanism_v1.md)保留历史；其中登记时“未运行/目录不存在”的文字不是当前历史状态。[旧单次授权](../experiment_records/2026-10-13_ic_role_mechanism_authorization.md)已在一次失败尝试中消费，[失败记录](../experiment_records/2026-10-13_ic_role_mechanism_v1_failure.md)及其四份原始证据不覆盖、不续跑。失败发生于首个 prepare 的活动状态门，完成训练轨迹及温度拟合均为 0，不是模型效果负结果。

[修复记录](../experiment_records/2026-10-13_ic_role_state_publication_repair.md)登记了派发前写回 RUNNING 的一行修复、三项 RED/GREEN 回归和既有 96 项限定合成检查。本轮不重跑这些测试，也不据此宣称全流程成功。

## 2. 固定身份与范围

| 项目 | 本次登记 |
| --- | --- |
| 已修复源码快照 | `da176fc5d247affde7b2ceb766aa5a5727aac658` |
| 源码快照树 | `b2dbba7714ef359c44dd085ac5160a0ca35e43e0` |
| 目标分支 | `codex/theory-aware-report` |
| 执行闭包 | 机器协议列出的 34 文件工作树字节 SHA-256 |
| 新执行器 SHA-256 | `423fad79bee6d5a351589319cb03b98a8097b958615a9b9e1fd274ef921f302a` |
| 34 源码哈希表规范 JSON SHA-256 | `89aa86065729315e0714491ba9394cf1f5b62814cdc789127d2c647a03859a3a` |
| 冻结规格 SHA-256 | `35894a722b27fd5bc7e02a23d1e87047b5e5acdb0762a5ed99dae6bab7f5183f` |
| 新独占输出 | `outputs/training/ic_role_mechanism_v2/development_fold_0`，登记时不存在，本次不创建 |

相对旧机器协议，34 个源码中仅 `run_ic_role_mechanism.py` 的身份变化，其余 33 个逐项相同。模型家族字符串仍为 `ic_rfa_b_v1`，符合现有执行器严格 schema；不增加可执行 JSON 顶层字段，不开发新执行入口。

设计依据为[冻结规格](../superpowers/specs/2026-10-12-identity-centered-role-adaptation-design.md)及[实施计划](../superpowers/plans/2026-10-12-identity-centered-role-adaptation-implementation.md)。本新协议替代 v1 的执行身份、授权和输出目录；科学设计及结论限制不替代、不放宽。

| 固定输入（继承身份，本次未读） | SHA-256 |
| --- | --- |
| `outputs/training/lc_role_adaptation_v2/development_fold_0/fit_raw.pt` | `d1edbe1f87977ce0a91639bc8cb455708d4c2e17d53340c48d11d2e8b2a568b1` |
| 同目录 `inner_assignment.json` | `e603ba0b603bc903f82649dd19edaebb69f7e171be0dec2092d69048c5469927` |

未来另行获准时才核对输入字节和内容。仅现有已暴露 340 个唯一 fit 分组，T/Ti/G/M=66/130/97/47；既有内折 0/1/2 和行序不重抽。输入 p4、H1280、E18、contradiction、labels、records 不增加。每内层训练折拟合 PCA64（full SVD、载荷规范符号、不 whitening）、h64/E23 标准化及正全局 T0，标准差下限 1e-8；验证折不参与。E23=E18+log(p)4+contradiction1，不是元素分析信息。禁止全 fit 重训及 CNN 重训。

本轮是既有专家特征条件下的 fit-only 机制探索，不是整个视觉系统独立交叉验证或确认性效果实验。不读 stop、原图、新分组、组合缓存或独立测试。B 为当前主线，阶段条件化 C、品位、回收率、物料流与送检经济成本继续后置。

## 3. 冻结网络、损失与运行矩阵

Windows-10-10.0.26200-SP0，现有 `.venv-training`，Python 3.11.5，torch 2.12.1+cu130；实际 CPU float64、单线程、确定性算法。包版本保持 numpy 2.4.6、scipy 1.17.1、scikit-learn 1.9.0、threadpoolctl 3.6.0、matplotlib 3.11.1，不安装或升级依赖。

每支 64-4-64、SiLU/Tanh、上投影无偏置；共享读出 87-16-4，FC-S 为 87-34-4。C 有 3024 可训练标量及 3024 冻结函数参照；HC 为 1476+1476，FC 为 3132+3132，FC 不是精确同容量。参照函数接收原始 h/e，不接收适配后输入，不更新且无可写存储别名。

\[
\delta_j(h)=\frac{A_j(h)-A_{j,0}(h)}{16},\quad
g_j=4\bar p_T\bar p_j=bw_j,\quad b=4t(1-t),\quad
\widetilde h=h+\sum_jg_j\delta_j(h).
\]
\[
s=F_\theta(\widetilde h,e)-F_{\theta_0}(h,e),\qquad
L_S=L_{\mathrm{NLL}}+0.10L_{\mathrm{pair}}
+\frac{0.001}{2}\|\theta-\theta_0\|_2^2.
\]

U 仅用 b/3 均匀路由；O 正则惩罚 theta，S 惩罚 theta-theta0；noaux 将 pair 权重设为 0。alpha_max=beta_max=0.5，epsilon_f=1，三辅助角色均权 1/3。使用未加权四分类 NLL，全批次 Adam，lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0。其余初始化、随机流、log-space 合同和有界输出按冻结规格。主 q argmax 决策，平分固定 T/Ti/G/M 顺序，不另调阈值。

固定单元顺序：

`Z-O-U, Z-O-R, Z-S-U, Z-S-R, C-O-U, C-O-R, C-S-U, C-S-R, HC-S, FC-S, C-S-U-noaux, C-S-R-noaux`。

每单元依次 fold 0/1/2，每折依次 seed 20261002/20261003/20261004，共 **108 条轨迹**，最多 **3 次训练折温度拟合**。先准备三个内折，再按矩阵执行，最后临时分析及一次最终推理验收；不再全 fit 重拟合或筛选后重训。完整交付每轨迹须完成 400 次更新；检查点 0/1/2/5/20/50/100/200/400，**100 为固定主要描述点**，400 是预算而非收敛证明。禁止早停、选最佳点/seed、扩大网格、调度、增强、BN、Dropout、STE、伪标签或额外温度。

108 轨迹字典的规范 JSON SHA-256 固定为 `8ae883853d9b82ac2ee936632a0900a43318a7117c76fe6a1d6c61655a9c1e89`。规范化为 `json.dumps(value, sort_keys=True, separators=(',', ':')).encode()`。Z/C 比较是整块参数化变化，不能解释为单一初始化因果；HC/FC 在主输出计算 pair loss，无三辅助视图。

## 4. 理论核验与效应判据

| 机制 | 冻结证据与解释边界 |
| --- | --- |
| P1 恒等起点 | 初始 q/锚点、冻结参照身份、真实参数微分、前五次更新前完整数据/正则/总梯度及布局；恒等值不等于零微分，梯度非零不保证纠错 |
| P2 双预算保护 | 逐样本特征及概率包络、谱范数、主 b²/辅 b 证书、同权重 adaptation-off 及保护覆盖；不保证真标签、召回或实用覆盖 |
| P3 起点标量分支核 | 初始 Jmain=G Jaux、Kmain=G Kaux G 与奇异值，数值秩阈值 1e-10；仅共同零适配起点/分支块，不推广为全参数核或全程收敛定理 |
| 角色真实增量 | 各检查点分支范数/余弦、参数距离、固定权重逐分支移除；仅推理消融，核迹或梯度较大不等于角色收益 |

主路径同权重关闭适配的界仍为
\[
|\log q_y-\log q_y^{\rm off}|\leq
(\bar\alpha+\bar\beta)\varepsilon_f L_F b^2,\qquad
L_F=(1+e^{-1})\|W_2\|_2\|W_{1,h}\|_2.
\]
辅助路径 b² 换为 b；界容忍度 1e-10+1e-10*abs(bound)，起点恒等绝对容忍度 1e-12。off 不是 T0 或重新训练的 U。零梯度、真实奇异核或不足的保护覆盖如实保留，不换种子、抬门值或加扰动；P3 条件不满足时不报最小特征值下界。

主要配对 **C-S-R vs C-S-U**。其余既定配对为 noaux R/U、C-S-R/HC-S、C-S-R/FC-S、C-S-U/Z-S-U、C-S-R/C-O-R、Z-S-R/Z-O-R。保留全部 12 单元、三种子、三折合并的 340 行 OOF 及 T0 同行锚点，不挑较差对照。

报告 NLL、Accuracy、Macro F1、Brier、固定 15 桶 ECE、完整混淆矩阵和逐图预测；目标召回/正确数分母 66，Ti/G/M 误入目标分母 130/97/47。目标配对新增、损失与净纠错为
\[
N_+=\sum_{i:y_i=T}\mathbf1[\hat y_i^R=T,\hat y_i^U\ne T],\quad
N_-=\sum_{i:y_i=T}\mathbf1[\hat y_i^R\ne T,\hat y_i^U=T],\quad
\Delta N_T=N_+-N_-.
\]
同时报告 Ti/G/M 误入目标的新增/减少，三 seed 方向逐项公开；三 seed 与重复内折不充当独立显著性证据。无自动选模或晋级阈值。没有一致角色增量就保留负结果、停止晋级；仅 NLL 改善但目标损失不称选矿收益；400 步和合成核验不证明收敛、效果或原创理论。

本轮没有 P/E 凸强对照、无界控制、来源外/真正未暴露确认或独立数学审核，不能称全面优于所有替代。正探索结果最多支持另写效果协议，不自动读 stop、改报告或扩网格。中心化、L2-SP、低秩瓶颈及 Gram 同余不能各自宣称为首次创新。

## 5. 7200 秒全流程监督与交付

未来一次入口的累计硬上限 **7200 秒**含远端登记、输入核验、准备、全部训练、梯度/Jacobian/核诊断、临时分析、导出及最终公开推理验收。每任务最多 7200 秒且受剩余累计预算限制，心跳 15 秒；持续接电、复用现有原生监督器、临时防待机及退出释放。旧 6 小时/拟合额度、v1 失败目录与许可均不继承。

历史合成估算 3767.654 秒来自 `f933153387b39a4ff414e02ad2281249dc41d891`（含 1.5 倍系数）。b412dfc 的校验加固和 da176fc 的状态发布未重新计时；旧数值不是本源码精确测量或成功保证。本次不重测。真实额外开销、临时和最终两次推理成本可能导致超时，不删控制、略诊断或延长预算。

断电、监督中断、心跳/累计超时、子进程残留、电源请求/释放失败、身份错误、缺类、非有限/概率下溢、参照改变、界违规或诊断缺失，均失败即停、不重试、不续跑。不重复空闲验收、不主动休眠、不永久改电源计划。

未来按冻结 `run_stage(..., 'fit-mechanism')` 执行 3 个准备任务、108 轨迹和 1 个临时分析任务，共 112 个受监督任务。阶段文件、消费锁、packet/task_started、日志、权重/准备结果、artifact_manifest、analysis 等只写新独占目录。原始张量及权重不上传 GitHub，轻量状态/指标/如实结果另行获准后同步。

临时分析 `delivery_verified=false`。仅阶段 MECHANISM_COMPLETE、108/3 计数、全部身份与父/子释放正确后，公开 `verify_mechanism_delivery` 才能进行一次保存状态的前向核验，不优化、不重拟合。现有公开函数单独调用不会给自身限制剩余时间；必须复用 `run_supervised_job` 在独立子进程内执行，父级 `WindowsAwakeGuard(api=NativeEnvironmentAPI)` 持续监督，时限为本次入口 7200 秒减去所有已用时间。最终子进程不另建电源请求；父级负责保护并释放。`final_acceptance`、`whole_experiment.json` 记录总计时及终态。剩余时间或证据不足即失败，不将临时结果当完整交付，也不修改源码绕过。

## 6. 尚未完成的前置事项

独立工程复查先前因额度限制未返回结论，详见[复查证据](../../outputs/ic_role_state_publication_qa_v1/review_evidence.json)。本协议按人类要求明确保留缺口，不再次派发复查，也不写“独立审核通过”。后续复查如何完成、发现问题如何处理，应另行决定；如果源码变化，需再次登记身份。

**本次没有获准的运行命令，不调用 run_stage、worker、真实加载器或最终交付验证器。** 当前机器草稿会在授权门拒绝。后续真实单次运行仍须直接的人类明确批准、记录真实问题项身份及录入 UTC，将授权 envelope 重新登记并核对 exact bytes 后才可执行；不能把本次登记批准、旧已消费批准或自动继续当作真实批准。

任务 8 整体未完成。未暴露数据确认、stop 评价、独立数学/新颖性评价及正式报告整合均不在本次范围。主项目与工作树唯一正式报告只读 SHA-256 均为 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`，六张用户既有图改动不覆盖或暂存。
