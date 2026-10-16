**发现**
以下均为静态审阅结论，反例未执行；按严重性排序。

1. **[P1] 磁盘上的 RUNNING 被当成活跃父监督，陈旧 packet 可脱离原累计预算执行。**  
   [run_msb_role_mechanism.py:265](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/run_msb_role_mechanism.py:265) 只核对状态、active_job、协议摘要及 packet 哈希，没有验证父进程存活或原始截止时间。触发路径：父进程在208-217行发布活动任务后、实际派发前终止，留下尚无 `task_started.json` 的合法 packet；之后直接进入 worker，现有检查仍可通过。280行新建的 child guard 重新起算，而训练所用 `stage_elapsed_seconds()` 只返回时间，并不自行实施7200秒上限。原预算依赖仍存活的父监督。因此，“必须受活跃父监督、不可续跑、累计预算不重置”尚未被完整强制。这里确认的是静态接纳路径，不是声称已实测所有 Windows 硬杀或孤儿行为。

2. **[P2] prepared 训练记录没有严格限定为 projector_fit，非 fit 标记可能通过最终验收。**  
   [msb_role_data.py:48](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/msb_role_data.py:48) 仅拒绝 `projector_stop`，缺失或其他 `tc_subset` 值均可通过。静态反例：某折训练行改为其他 scope，其他身份、标签、张量及登记摘要相应一致；另外两折该行的验证记录仍标为 fit。105-110行跨折比较不包含 `tc_subset`，分析端严格 fit 检查又只作用于验证预测行，因此该训练 scope 不一致可贯穿优化及最终验收。哈希保证登记字节一致，不能替代 fit-only 语义检查。未据此判断真实 prepared 已存在污染。

3. **[P2] 三折联合身份验收未接入拟合前的三个 load 任务。**  
   [run_msb_role_mechanism.py:285](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/run_msb_role_mechanism.py:285) 每次只验证单折，load 结果仅返回 `validated=True`；随后即进入75次拟合。完整 `validate_fold_collection()` 到 [analyze_msb_role_mechanism.py:365](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/analyze_msb_role_mechanism.py:365) 才调用。三份各自合法、但验证集重复或跨折身份不一致的登记文件，可以先完成拟合，最早在其后的 analysis 或 final-verify 被拒绝。终态通常仍会失败，不属于虚假晋级，但会先消耗单次拟合授权，未落实联合 OOF 身份验收先于优化的边界。

**优点与数学核对**
未发现规格式(1)-(21)在其明确假设内的实质推导错误；核心网络和损失静态实现与规格一致。

- 固定参照使用独立 clone buffers，随 `state_dict` 保存；训练逐步检查参照摘要，最终回读严格载入并核对初始化参照。既有测试包含 MAIN 一步更新、不同 seed 构造后的 save/load，以及同形消融初始化一致性。
- MAIN 为三个独立 rank4 分支，1083当前参数加1083固定参照；SHARED-BIL 的1107参数及块嵌入确实包含 MAIN，MLP 为1095参数。未见新增 CNN 骨干改动；e23没有被实现成新物理模态。
- 内部见证 `s=±atanh(.5)`、目标胜出下界 `.125`、非目标间下界 `.25`、相对能量 `.25`及有限常偏置 SP 代价成立。同输入冲突标签和高置信正确被翻转的反例均保留，不能推出联合可学习或只修错不伤对。
- 边界 bias 导数、单样本 pair 梯度下界及 `.75I` Jacobian 的条件正确；它们不保证批梯度、完整 NTK、所有参数块梯度或优化收敛。log-odds 重构和漂移界正确，共同归一化仍会改变其他绝对概率。
- 混合 Hessian 只排除指定仿射或可分加性 score，不排除一般 MLP／共享双线性网络；ADD 也没有被误称为消除全部非线性交互。
- 四个凸目标、BIAS 的 MAIN 子模型等价、正 ridge、`µ=.001`证书及200 Newton／4201目标评估边界基本一致。证书仍依赖有限精度梯度计算，不是舍入误差受控的形式化证明；REPAIR4-COMP不是完整原方法复现。
- 损失归一化、FIX1仅改输出半径、NOAUX仅去 pair 权重正确。100步逐 seed 七参照门、误入与 Macro-F1 约束、12证书要求、ADD独立门及禁止400步补救均已体现。
- 实数内部性质与 float64 饱和／下溢分开处理。调度算术为75拟合、80任务；最终615次验证回读加12次训练证书检查，共627次核验，不是615次总计算。

**覆盖与缺口**
审阅范围为 `41840cd493760e81fd67ceed61bc9d7481aced24` 至 `125527749ca1d5cd7201275018a7aab4cc6fc6c9`；限定源码／测试在 `3d48cd4` 后无差异。失败审核记录未作为审核证据。

完整读取八个新源码模块、八个对应测试模块及 `msb_synthetic_helpers.py`；读取 accepted 规格、实施计划、原生合成验收记录和获准作者 ledger。复用代码仅查看六个允许文件中与 prepared 格式、监督及电源合同有关的内容。

读取全部七项允许证据：`synthetic_tests.log`、`conditional_theory_checks.json`、`control_plane_summary.json`、`source_identity.json`、`legacy_source_preservation.json`、`protected_artifact_preservation.json`及 `assets/architecture_metadata.json`。后两项是历史保持记录，不代表本轮重新核验当前保护文件。

既有日志记载83项通过：data7、model9、controls10、audit13、training9、runner15、analysis16、figure4。覆盖包含短合成轨迹、实际100更新 MAIN mask 记录、四类实际合成凸解及证书复算；75任务元数据 fixture 不是75次真实拟合或400步训练证据。

重要缺口包括：陈旧 RUNNING／失去父监督的接纳路径、非 fit 训练 scope、拟合前跨折联合验收均无对应端到端拒绝覆盖；完整真实协议路径、全矩阵及原生硬杀／孤儿／电源异常行为未验收。其余 OS 行为仍属未判断，不以模拟绿灯证明安全。

本轮没有重跑测试、Python／模块导入、前向、求解、拟合、预处理、计时／空闲／电源验收或外部搜索；没有读取真实 `.pt`、prepared／fit／stop／dataset、矿物图像或缓存。未打开六张用户 PNG，未改文件、index 或 HEAD，未派子代理。真实数据身份、已有专家泄漏程度、来源独立性、图像导出视觉质量、性能成本和外部泛化均未独立判断。

**分别裁决**
- **(a) 数学／实现一致性：** 核心条件数学与模型、损失基本一致；上述执行及身份合同缺口使整体工程验收不能判通过。
- **(b) 后续协议登记准备度：** 当前提交不宜作为“审核通过、可冻结运行”的候选登记；问题先记录，修复及任何 Task9 工作另需批准。本报告不授权执行。
- **(c) 原创性／效果：** 均未证明。本轮未查新、未读真实数据、未运行真实矩阵；既有83项合成绿灯不能替代优化收敛、角色增益或独立确认。

