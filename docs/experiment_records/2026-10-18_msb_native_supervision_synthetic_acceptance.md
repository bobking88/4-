# MSB原生无数据验收驱动：原生方式实施与限定合成验收

日期：2026-10-18。代码/合成授权：`call_6961d4dac6b64d7aba8e538f01c5e097`。
状态：`AUTHOR_SYNTHETIC_ACCEPTED_NATIVE_NOT_RUN`。规格/计划基线：`a71d54821a482d50eb1fd182d4a5bcafe9219700`。

## 交付与范围

按已认可[规格](../superpowers/specs/2026-10-18-msb-native-supervision-acceptance-design.md)和[计划](../superpowers/plans/2026-10-18-msb-native-supervision-acceptance.md)原生实施Tasks1--3，未派代理。
仅新增三个代码文件：`scripts/check_msb_native_supervision.py`、`tests/test_check_msb_native_supervision.py`、`tests/run_msb_native_supervision_synthetic_acceptance.py`。
没有修改既有43项执行源码或10项测试/验收器，没有创建实际原生协议、没有原生进程/计时/空闲/电源验收，没有真实数据/原图/缓存/权重读取、真实训练或真实数据前向，正式报告不变。Tasks4--5未开始。

Task1提交c54fef2建立纯合同、严格门及守卫；Task2提交a6f5f81接入原MSBParentSupervisor/open_worker_scope及私有能力CLI；Task3与本记录同次归档，加入collector、只查询进程身份的observer、Git登记核对、四例接收与最终发布。
旧执行器获独立APPROVE不等于新验收驱动获独立批准；本轮只有作者自审及合成证据，新的独立复查仍另批。

## 最终合成结果

[最终回执](../../outputs/msb_native_supervision_synthetic_v1/final/attempt-002/acceptance_summary.json)和[原始日志](../../outputs/msb_native_supervision_synthetic_v1/final/attempt-002/synthetic_tests.log)记录：**202/202通过**，0失败、0错误、0跳过、0预期失败、0意外成功、0受限未模拟调用。
原157项名称完整保留，新模块45项；固定十模块计数为49、12、9、10、13、9、34、17、4、45。不是202次真实实验。
两次torch.load仅为专用合成scratch的round-trip；原九模块的合成张量数学、前向、梯度和小规模求解照常实际执行，不宣称所有计算都被mock。

原生构造、时钟/进程/句柄/电源/线程启动和网络接口在验收中均为fake或守卫禁止。已安装torch/sklearn必要数值bootstrap位于应用守卫之外，不能写成“没有原生库装载”。
NativeServices的真实构造未被执行；纯源文件/Git对象核对的单元测试只用临时伪源文件和fake响应，不是一次真实登记或原生运行。
main接线测试向内层注入skipped TestResult，其FAILED回执写在独立临时ROOT；这是反例输入，不是外层测试跳过或额外真实phase尝试。

## 全部尝试账本

每phase最多3次，含失败；七phase实际共14次，均保留独占原目录、运行前直接快照和日志，不覆盖或重命名。

| phase/attempt | 项数 | 失败数 | 错误数 | 实际状态 |
|---|---:|---:|---:|---|
| contract-red/001 | 9 | 7 | 0 | FAILED |
| contract-green/001 | 9 | 0 | 0 | PASSED |
| worker-red/001 | 19 | 10 | 0 | FAILED |
| worker-red/002 | 21 | 1 | 3 | FAILED |
| worker-red/003 | 21 | 1 | 0 | FAILED |
| worker-green/001 | 21 | 0 | 0 | PASSED |
| collector-red/001 | 34 | 13 | 0 | FAILED |
| collector-red/002 | 38 | 17 | 0 | FAILED |
| collector-red/003 | 41 | 1 | 0 | FAILED |
| collector-green/001 | 41 | 0 | 0 | PASSED |
| collector-green/002 | 44 | 3 | 0 | FAILED |
| collector-green/003 | 44 | 0 | 0 | PASSED |
| final/001 | 202 | 1 | 0 | FAILED |
| final/002 | 202 | 0 | 0 | PASSED |

contract-red的7项为新函数缺失断言失败，不是导入错误。worker-red/002的3项错误来自fake GRANT/RENEW错误复制HELLO的job_assigned=False；按原control_message修正fake为True，未放松生产协议。worker-red/003仍有真实CLI接线反例，其他20项已通过。
collector-red/003验证了观察者读取部分写入JSON时过早失败；修复为暂缓查询/ACK，仍受原case/480/ready-ACK期限约束，完整但非法元数据仍拒绝。

collector-green/001通过41项后，作者静态自审加入三个边界回归；/002的3个实际失败为本地化EOF、ACK读取跨5秒和观察锁清理。修复分别采用原生WinError109而非英文文本、读取后再次采样、1秒有界锁获取且不关闭仍在使用的句柄；/003通过44项。
最终组合又发现finish在停观察线程之前可能先进入同一采样锁；final/001中原157及此前44项通过，仅新回归失败。先确认停止线程，未确认则仅有界清理并失败，不再采样；final/002通过202项。
后续回归只使用剩余固定green/final额度，标签与失败原文如实保留；原RED/green已耗尽额度不重置，不增phase、不扩模块、不另派代理。不是另开目录重试已失败原生实验。

归档时完整暂存区diff-check仅报告collector-green/attempt-002原始日志第30行的测试输出行尾空白；为保存失败证据，原样保留，不清洗日志。源码及作者文档单独diff-check通过，此例外不是测试通过结论或生产源码格式放宽。

## 已实现的限定合同

- 四例顺序N1--N4、原MSB共同7200000ms预算、额外480秒整体窗口、90/45/75/45外层限额、每例一次；不运行原80-job synthetic入口，不假造0.25秒MSB超时。
- N1真实控制状态机的两次以上续租ACK与正常关闭；N2退出23/无DONE并区分缺关闭与原生broken-pipe原因；N3正常DONE后迟退由冻结租约拒绝；N4由固定故障线程在身份ACK后发起父exit31。
- observer只开查询/同步进程句柄，核对PID及创建时间，不复制Job或通信能力；必须有已分配、WORKING和初始ACK原始证据，再发送collector ACK，退出观察及句柄关闭不足均不通过。
- 根guard和既有LC wrapper的同线程parent链有实际接线反例；所有硬退出路径显式release未观测，不能补造。fake SystemExit仍会展开Python finally，因此它不证明真实os._exit后的清理或Job因果。
- 只允许固定原生协议路径、来源46项、精确新许可及active collector claim；当前文档/开发许可拒绝。许可是明确人类批准及登记工作流，不是靠黑名单进行任意ID语义或密码学认证。
- 严格合成门与main回执接线拒绝空/skip/expected/unexpected/守卫错误；最终写后重新核对期限，双重写失败保留两异常及非零传播，不宣称磁盘撤销已持久化。

## 保护、自审与取舍

[源码身份](../../outputs/msb_native_supervision_synthetic_v1/final/source_identity.json)：
旧43项执行源码及10项测试/验收器原始SHA-256全部匹配；新三文件与final/002的[直接快照](../../outputs/msb_native_supervision_synthetic_v1/final/attempt-002/source_snapshot.json)匹配。未来原生登记候选46项是旧43加新三文件，不改变真实MSB科学闭包。
原始字节SHA、Git blob及规范化LF口径分别保留，不互换。未生成真实协议SHA或假批准。

[保护和范围](../../outputs/msb_native_supervision_synthetic_v1/final/preservation_and_scope.json)：
20个指定保护文件匹配，含六张既有用户脏PNG和两处正式报告；旧规格/计划、157/146等历史及旧独立复查没有diff。
正式报告两处SHA均为`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。
接受的本轮规格/计划不改历史状态或复选框；实际进度由新记录及账本说明。原生协议路径及outputs/msb_native_supervision_v1仍不存在。

作者自审逐项对照Tasks1--3和五项Review Focus，检查原状态机接线、单次身份、同一累计起点、有限清理、去能力化日志、静态源身份、全部失败与后续回归。没有独立审阅者，不冒称独立通过。
显式授权的固定十模块范围优先于通用全库发现；当前禁止代理，故没有执行通用技能中的自动review派发。采用既有worktree、原生PowerShell/apply_patch账本，不运行不适配的MSYS辅助器；保留忽略的审计scratch，不为清理删除证据。
上述取舍会留下白名单外兼容性、真实原生构造/调度/管道/Job/线程、OS阻塞及崩溃一致性的未验证风险。有限通过不等于完整Windows正确性，尤其不能据本轮结论放行真实机制实验。

## 研究主线与下一闸门

本轮是恢复可靠实验条件的工程工作，不计为报告或论文的网络理论创新。B主线、C后续方向、MSB-RRE公式、损失、初始化、11单元矩阵及三种子七参照晋级门不变。
真正需要的证据仍是：角色非线性分支是否被利用、间隔缩放预算是否带来有效跨界、相对SHARED-BIL/CONCAT-MLP/充分求解凸参照是否有独立收益。340已暴露fit也不是未暴露泛化。
下一步建议另批一次独立只读工程复查，不重跑本轮测试；再按结果决定是否登记一次原生无数据协议及申请运行。真实MSB实验仍单独批准，未达晋级门保留NO_ROLE_PROMOTION，未取得科学结果不改正式报告的成果结论。
