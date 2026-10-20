# MSB初始化异常保留最小修复规格

日期：2026-10-20。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
本轮授权：`call_754f23a7456b4c5cbe8e744ba0dd6114`，仅写本规格、实施计划及GitHub归档。
基线：`11f6ab75ea5482dd72c5f6c90aaba5e72c3af1f7`；受审源码仍为`90c017a931fc1c4f858194502b8cfb28ddf054b8`。
依据：[单次独立复查](../../experiment_records/2026-10-20_msb_native_acceptance_repair_independent_review.md)、[原R4合同](2026-10-18-msb-native-acceptance-repair-design.md:67)。

## 1. 目标与已定位根因

只闭合R4/P2：首次RUNNING发布后check失败，再发生根guard释放失败时，结构化errors/primary_errors丢失前一个原因。
[run_acceptance](../../../scripts/check_msb_native_supervision.py:928)先建立state，写registered_protocol及RUNNING并check，随后才进入现有body try；guard.__exit__可用释放异常替代正在传播的初始化异常。
这不是已确认的误报PASS或网络效果缺陷。当前REQUEST_CHANGES，未执行新的复现；静态证据和既有230项日志不能算修复已验证。
成功条件是保留原始异常对象及顺序，继续失败关闭、有限失败发布、实际CLI非零，且不改变授权、来源、N4或期限合同。

## 2. 方案比较与选择

- 采用A：将既有body try向上移至state字典成功建立后，覆盖两次初始化写入、首次check及原有case循环；复用既有except记录body原因，外层继续独立记录guard退出原因。无新公共接口或状态机。
- 不采用B：在外层递归挖掘__context__/__cause__。异常链可能含包装或被抑制内容，不能替代明确的业务/释放对象和稳定顺序。
- 不采用C：重写guard、通用错误聚合器或原生产监督器。超出这一项修复边界，并增加无关接口风险。

## 3. 精确实现边界（未来另批）

未来仅改三个既有文件：`scripts/check_msb_native_supervision.py`、`tests/test_check_msb_native_supervision.py`、`tests/run_msb_native_supervision_synthetic_acceptance.py`。
保留`run_acceptance(root: Path, protocol_path: Path, *, execute_native: bool, approval: dict, services=None) -> dict`和`publish_acceptance(output,state,binding,*,services,final_check=None)`签名。
state建立成功后立即进入原body try；registered_protocol写入、首次stage_state写入及紧随check全在其中。except仍追加`("body", error)`，不重新抛出让guard覆盖它；外层仍追加`("collector_scope_or_release", error)`。
按发生顺序保留独立错误：body在先、release在后；后续postflight若失败仍按原逻辑追加。不能从字符串重建原始对象或只依赖Python异常上下文。
不改已有失败发布次数：已有非PASS只尝试一次失败持久化；失败时AcceptancePublicationError保留全部primary_errors、首项primary_error兼容别名和独立persistence_error。
本修复不增加running_published字段、不把内存state存在伪装成磁盘RUNNING已写入。state建立前的授权/登记/来源拒绝仍原样传播且不创建输出；state已建立且输出已存在时的初始化写失败沿用既有有限失败收据策略。
释放失败不得补造power_request_released=True；初始化失败后不启动observer/controller、不重进guard、不重试写入或换目录。
未来NON_NATIVE_IDS追加已知只读复查ID `call_29684c57aed94222832ace218cd392be` 和本文档ID `call_754f23a7456b4c5cbe8e744ba0dd6114`，不把文档许可当原生运行许可。
保护计数不扩写为科学成果：原43执行源码及10测试/验收器均保护不变；新增三验收文件仍单独记录源码身份。

## 4. 五个精确反例与正例边界

1. `test_initial_running_check_failure_and_release_preserve_both_causes`：在真实合成JSON写入RUNNING完成后再注入AC=False、fake时钟+31000ms、原起点+480000ms三种故障，并使原guard的fake释放失败。要求errors前两项为body、collector_scope_or_release，前项类型/消息等于实际捕获的检查异常，case_records为空、launch/observe为0、失败终态和delivery_verified=False。
2. `test_initial_running_triple_failure_preserves_exception_objects`：在第1项AC模式叠加失败收据写入OSError。要求primary_errors按原始对象顺序保存check与release（用is断言），primary_error is primary_errors[0]，persistence_error is注入OSError，失败发布仅一次且failed_state不是PASS。磁盘原RUNNING可仍存在，不声称已撤销。
3. `test_initial_write_failure_and_release_preserve_both_causes`：分别在registered_protocol和首次RUNNING写入前抛出已知OSError，再使释放失败；原初始化异常不得丢失，不能声称未完成的RUNNING已持久化，不能启动case。
4. `test_initial_running_failure_cli_is_nonzero_and_clock_injected`：真实main调用路径使用fake服务和实际原guard+fake环境，初始stdlib monotonic显式注入并断言一次；首次RUNNING后的AC及释放双失败返回code=1、stdout JSON含两项原因、无case launch。
5. `test_pre_state_rejection_preserves_no_output_contract`：登记拒绝、来源拒绝及guard进入失败三种pre-state路径，保持原异常传播，输出目录不存在、write/launch/observe为0；本文两个新非原生ID在既有拒绝测试中追加子例，不新增运行权限。

夹具必须区分“写入前失败”和“写入成功后的check失败”。后者的局部write_json包装先调用原实际合成写函数、读回合成RUNNING元数据确认，再改变fake环境；不能仅在写入前抛异常冒充原窗口。
检查异常和释放异常用局部观察包装捕获真实抛出的对象后原样raise，保留实际guard/check调用；不mock成功结果、body错误收集器或判定器。
原230个完整测试名必须保留；新增固定五方法，预计final为235项，按实际日志核对，不预填PASSED。旧成功路径、旧N1联合错误、N4、postflight和发布检查由原固定十模块保留。

## 5. 有限未来合成验收

仅在开发与合成验收获批后，使用新根`outputs/msb_initialization_error_retention_synthetic_v1`和scratch `.superpowers/sdd/2026-10-20-msb-initialization-error-retention/test-tmp`；本轮二者均不创建。
只执行`collector-red`、`collector-green`、`final`三阶段，每阶段最多3次独占attempt（含任何失败），递增目录、不覆盖、不重置；耗尽停止。旧七阶段与全部失败证据不动。
验收器只改输出/scratch两路径；同一测试文件的隔离验收门夹具只同步其预期路径。PHASES可选项、十模块、bootstrap、严格接收门和所有已有守卫不变，不执行其余四阶段。
隔离验收门子例仍可能调用fake contract-green并故意得到FAILED/skip；它不是新根下额外阶段，不冒充实际跳过或重试。
受限Windows/进程/时钟/电源/线程启动/网络接口使用原fake/guard；特别保留全局monotonic守卫和CLI入口注入。只允许既有torch/sklearn数值bootstrap和获批后的合成计算，不读真实输入或启动原生进程。
最终必须原230名+五新名全在，0失败/错误/跳过/预期失败/意外成功/已守卫未模拟调用，三文件匹配运行前直接快照，旧保护身份及原件不变。仪器计数只证明覆盖范围，不抹除历史collector-green/001时钟缺口。
本规格/计划/复查授权均不授权原生无数据验收、真实前向/拟合、PCA/T0或任何新的科学实验，不扩大7200秒科学协议预算。

## 6. 保持的合同与后续门

原43+10、科学网络/损失/公式/实验矩阵、正式报告、六张已有脏PNG和全部历史记录不改；原480秒、30秒gap、2秒漂移、N1--N4参数和原生产7200秒边界不改。
作者合成接收与独立复查分开；再次独立复查、原生登记/验收及真实机制实验仍分别另批，不自动派代理或续跑旧失败目录。
本轮仅文档起草及静态自审，未实施、未运行，R4仍PARTIAL/UNFIXED。本修复是诊断证据完整性工程，不是MSB-RRE的网络理论创新或独立收益验证；科学晋级门和NO_ROLE_PROMOTION规则不变。
