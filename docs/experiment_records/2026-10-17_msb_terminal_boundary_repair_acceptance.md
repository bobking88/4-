# MSB终结边界修复：限定合成验收

日期：2026-10-17。基线：`4a6da08fbb3e81ab8a282968ea045f71eb727956`。
开发授权：`call_2a665eb0bae044b38917a4aa34f5127a`；有限验收器调整：`call_34d91895cd594efe92c80a1bf6f41138`。
状态：`AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING`。这是作者原生实施代码后的全模拟验收，不是原生OS验收。

## 交付与证据

| 任务 | 实施提交 | 限定结论 |
|---|---|---|
| Task1 | f53a854 | 单一成功发布，最终写入跨期撤销；失败记录写不出则非零，不谎报已持久化 |
| Task2 | e31837b | 父端预持有启动所有权；启动后异常仍回收同一进程，未知创建保守失败 |
| Task3 | a27101e | 显式CLOSE_REQUEST/CLOSE_GRANT/DONE，同锁写入与有界退出排空；v2身份及终结证据门 |
| Task4 | 本记录及同次归档提交 | 固定九模块总验收、身份保护核对及作者静态自审 |

[最终摘要](../../outputs/msb_terminal_repair_synthetic_v1/final/attempt-001/acceptance_summary.json)与[完整原始日志](../../outputs/msb_terminal_repair_synthetic_v1/final/attempt-001/synthetic_tests.log)：146项通过，0失败、0错误、0跳过，受限接口未模拟调用为0。

| 模块 | 实际通过数 |
|---|---:|
| 执行安全 | 45 |
| 数据合同 | 12 |
| 网络 | 9 |
| 强对照 | 10 |
| 条件理论审计 | 13 |
| 固定训练流程 | 9 |
| 执行器 | 27 |
| 结果分析 | 17 |
| 公式结构图 | 4 |
| 合计 | 146 |

原121项测试名均保留；新增25项。80 job与75拟合的元数据/编排只使用fake与小合成张量，绝非完成真实实验。两次torch.load仅从专用合成临时目录读回测试张量，未读项目prepared、fit、stop、原图或权重。
三次load联合验收先于任何拟合；非法三折到此停止，拟合调用为零。63+12拟合元数据、615+12最终核验预算和7200秒原累计期限未改。

## 失败保留与验收器边界

[阶段开发记录](2026-10-17_msb_terminal_repair_development.md)逐项保留红绿结果。task1-red旧直接目录计第一次尝试：sklearn数值库导入被WinDLL守卫阻止，0项；attempt-002为未使用的torchvision依赖装载被阻止，0项。二者都不算有效红灯。获准的torch/sklearn数值bootstrap不包含torchvision；其模型/预处理用调用即失败的测试导入替身隔离，未放开kernel32或实际CNN调用。

task1-red/attempt-003才取得有效27项中5失败的反例，之后27/27绿灯；该红灯阶段三次额度已耗尽。Task2为32项中6失败后32/32；Task3为42项中13失败（含子案例），再以45项中6失败固定身份门，最后45/45。final/attempt-001为本次146/146。所有原件均保留，未覆盖、搬移或将历史失败改为通过。

七个固定phase各最多3次独占尝试，详见[尝试清单及保护核对](../../outputs/msb_terminal_repair_synthetic_v1/final/preservation_and_scope.json)。正常并发时序由确定性fake调度，不执行原生线程、Popen、WinAPI、系统电源/空闲验收或计时探查。torch/sklearn数值运行库的必要bootstrap单独披露，不能据此称为“无任何原生库装载”。

每次运行均先直接捕获当时源码；[最终快照](../../outputs/msb_terminal_repair_synthetic_v1/final/attempt-001/source_snapshot.json)的五项原始文件哈希均与验收后当前文件一致。source文本为读取时换行规范形式，sha256对应当时原始文件字节；二者不是同一哈希输入。旧v1验收器重建来源及早期时钟模拟偏差保留原有说明，不被本次绿灯覆盖。

## 身份保护与作者自审

[源码身份](../../outputs/msb_terminal_repair_synthetic_v1/final/source_identity.json)记录43项闭包，仅`msb_execution_safety.py`与`run_msb_role_mechanism.py`两项生产源码变化。其余41项、旧34项、网络/强对照/条件审计/结构图四项科学模块以及data/training/analysis均保持原哈希。10项保护文件全部匹配，包括六张用户PNG、两处正式报告及原协议/规格。科学规格、两轮修复规格、接受的计划及83/121项历史证据没有修改。

作者静态自审检查四项原P2、两向在途续租、旧请求/重放拒绝、退出独立于父读DONE、精确期限、无额外成功尾写、所有权清理和未来授权门。没有把拟议测试所在文件名当成功能要求：两个身份/下一job反例位于固定安全模块的ExecutionIdentityTests，完整runner仍纳入final。未删旧测试断言来迁就fake。

实施采用原生PowerShell及apply_patch账本；MSYS辅助器无法初始化时未重试或升级权限探查。既有工作树复用，未派发独立代理。上述替代与测试位置决定完整列于开发账本及本记录，若其隔离假设有误，影响的是覆盖充分性，不能靠合成绿灯排除。

这是作者自审，弱于新的独立复查。未发现需要在本阶段追加实施的问题；原生句柄、Job、跨线程竞争、真实文件I/O和本机进程退出仍未验证，不宣称生产可用。

## 科学结论与下一闸门

三个rank4分支、1083当前参数和1083固定参照、`R_ref=2*abs(m)+0.25`、原损失/优化器/初始化、11单元、三折/三种子、400步上限及100步主点、七参照晋级门均不变。没有新PCA/T0，没有训练/评价现有340分组，也没有修改正式技术报告。

本次只能修复“能否可信地执行既定检验”的工程准备条件，不能证明MSB-RRE有效、理论原创性成立、泛化提高或具备工业分选能力。理论创新的真实证伪实验仍未开展。

下一步建议单独批准一次独立只读工程复查。原生无数据验收、新真实协议登记和真实运行仍分别另批；本次未自动启动任何一项。未来执行身份是`msb_parent_bound_v2`并绑定本轮规格LF哈希，旧v1或文档/开发/复查许可不能充当真实授权。

复现入口为`tests/run_msb_terminal_repair_acceptance.py --phase final`，使用项目既有Python。当前phase已用attempt-001，不能把该命令当作未经授权可再次执行的建议；任何后续运行必须遵守剩余额度与相应人工批准，旧目录绝不复用。
