# MSB 终结修复：首次验收器导入失败

日期：2026-10-17。开发授权：`call_2a665eb0bae044b38917a4aa34f5127a`。
批准范围为提交 `4a6da08` 的原生 Tasks1--4 与全模拟合成验收。
状态：`HARNESS_IMPORT_FAILED_NO_VALID_RED_NO_PRODUCTION_REPAIR`。

## 实际发生

已新增固定阶段合成守卫入口，并在 runner 测试中补充五项终态发布反例。尚未修改任何生产源码。

执行一次：
`tests/run_msb_terminal_repair_acceptance.py --phase task1-red`。
[原始摘要](../../outputs/msb_terminal_repair_synthetic_v1/task1-red/acceptance_summary.json)为 FAILED，tests_run=0，failures=0，errors=0，skipped=0，harness_error 非空。
这是测试载入前的验收器错误，不是“零失败通过”，也不是有效的 T1 缺陷红灯。

调用链为 runner -> check_lc_native_environment -> run_lc_role_adaptation -> lc_role_adapter -> lc_role_data -> sklearn.model_selection。
已安装 sklearn 的 _distributor_init.py 预装载 vcomp140.dll / msvcp140.dll，首个 ctypes.WinDLL 被守卫拦截并抛出异常。记录 unmocked_boundary_calls=[WinDLL]；被拦截调用未执行。
torch 必要装载已经按获批 bootstrap 在守卫前完成，不能把“本轮无原生进程验收”误写为解释器/数值运行库完全没有 OS 装载活动。

未运行测试用例、项目 NativeMSBAPI 构造/tick、真实 Popen、原生电源/空闲验收；未打开真实 prepared/fit/stop、张量、图片或权重，真实拟合为零。
失败后只读核对安装包加载器源码及当前 diff，没有补跑测试或调用真实 DLL 接口来探查。

## 原件保留

- [完整失败日志](../../outputs/msb_terminal_repair_synthetic_v1/task1-red/synthetic_tests.log)。
- [运行前直接捕获源码及哈希](../../outputs/msb_terminal_repair_synthetic_v1/task1-red/source_snapshot.json)，包含当时验收器、两项测试及两项生产源码原文。不是事后重建。
- 该目录不覆盖、不移动或继续写入新结果；新增反例仍是未执行状态。

Task1 未完成；Tasks2--4 未开始。生产代码、网络、公式、损失、矩阵、既有证据和正式报告均未改，不宣称四项 P2 已修复。

## 最小调整建议（尚未授权、未实施）

1. 将已安装 sklearn 的必要数值运行库装载与既有 torch bootstrap 同列，单独记录版本/范围；项目模块导入及测试仍在 WinDLL/Popen/时钟/电源/线程守卫内，不能把项目 Native API 移到 bootstrap。
2. 保留原七个 phase 与固定模块，仅把证据目录改为 phase 内递增且独占的 attempt-001/002/003。每阶段最多三次，包含已有失败；不加任意模块或任意输出参数，禁止覆盖、删除、迁移失败原件。原 task1-red 的直接文件视为该阶段第一尝试，后续新尝试另建子目录。
3. 这些仅是合成开发的输入隔离/证据修正，不增加真实任务、训练重试或扩大网格；新许可后才修验收器、重新取得有效 red，再继续已批准 Tasks1--4。
4. 若仍触发守卫或达合成尝试上限，保留证据并停止，不把项目 native API 放行取得绿灯。

当前先归档失败事实并请求这项有限调整；不擅自改已审核规格/计划或复用已消费目录。原生无数据验收、独立复查和真实实验仍分别另批。
