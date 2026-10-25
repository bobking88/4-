# MSB 固定 v3 命名空间适配：代码与有限合成验收

日期：2026-10-25。状态：**AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING**。

开发授权：`call_ed3402ece50249469f240dc0b9cb744a`，答复“认可文件，批准代码与有限合成验收（推荐）”。批准规格/计划提交：`5ff02a45b547f07410368adce7be0d959b92aa86`。

## 1. 范围与实际变化

按[规格](../superpowers/specs/2026-10-25-msb-native-namespace-v3-design.md)及[计划](../superpowers/plans/2026-10-25-msb-native-namespace-v3.md)在既有独立工作树原生会话内实施，仅改原三个指定源码/测试文件：

- driver新增两个固定路径常量，更新FIXED协议名/输出引用与三处派生路径；worker仍从FIXED派生前缀。
- driver测试新增固定8个方法和独立v3夹具/helper；旧116个方法名不变，旧正文只改11处正向命名空间字面值及2处harness输出预期。
- 原合成验收器只更换output/scratch两处字面路径，模块、bootstrap、通过谓词和守卫不变。

新的固定协议路径为`docs/experiment_protocols/msb_native_supervision_v3.json`，固定原生输出为`outputs/msb_native_supervision_v3/acceptance_once`。**这两个位置没有创建或登记**；当前仅源码适配。旧v2协议与失败原件保持原样。

原`msb_parent_bound_v2`、CPython启动契约、N1-N4、480秒累计上限、各例上限、接电、单次与失败关闭策略均未改。权限策略、源码闭包、网络、损失、公式、实验矩阵与正式报告未改。

## 2. RED、GREEN 与 final

| 阶段 | 实际目录 | 测试数 | 结果 |
|---|---|---:|---|
| collector-red | attempt-001 | 124 | 新增8项因v3缺接线而断言失败，旧116项通过 |
| collector-green | attempt-001 | 124 | 124/124通过 |
| final，原十模块 | attempt-001 | 285 | 285/285通过 |

三个阶段各仅一次实际入口调用，没有重试或扩模块。每阶段错误、跳过、预期失败、意外成功、未模拟边界调用及harness异常均为零；RED的8项失败为计划要求的缺行为证据，不能将其写成通过。

RED时driver原始SHA256仍为`30ff7c45947108b80fd9c867d6ff8113590dd372fbcdbf4fd143e3d94123ae83`。新增8项和helper在RED之后冻结；GREEN之后至final/当前源码相同。旧正向夹具在GREEN阶段才迁移，未全局替换新测试中的旧版本负例。

final完整名集合精确等于旧277项加固定8项，共285个唯一名称。全量合成仍仅有原两次scratch合成张量读取，没有真实输入、真实拟合或科学实验。既有267及277项旧证据均未重写。

## 3. 验证内容

固定8项覆盖：新身份/类型与46项闭包、旧/混合/任意命名空间拒绝、四例case与worker前缀、本地协议路径、远端v3内容与原始/blob身份、独占目录和旧历史哨兵、完整模拟入口的claim/累计绑定/顺序及失败关闭。

作者另以旧driver文本的精确替换结果核对当前Git blob，确认除两个常量和五处命名空间替换外没有其他driver行为改动；当前blob为`ffbe6613eff5cc3567402e568672bd9d5ec7f285`。旧测试前缀逐字比较及冻结helper哈希验证均已完成，而非仅查看绿色数量。

当前三个文件原始SHA256：

| 文件 | SHA256 |
|---|---|
| `scripts/check_msb_native_supervision.py` | `d09a97dbff082f8081a74a96b84ecb2be1ad2cc16a33f8167b62c490f047ea78` |
| `tests/test_check_msb_native_supervision.py` | `8c973d98742ae074b7fda7d2b74d7a88d1e936d4b62505a0fd7c74879437650d` |
| `tests/run_msb_native_supervision_synthetic_acceptance.py` | `ebc386d2611e373130a2d3e6c1eba2ac9bb2247f8a4b2407198c33b9f0b8e745` |

原56项保护源码中53项不变；旧原生46项中43项不变，变化仅为批准的三个文件。保护基线延续443条记录并保留重复项；除三个文件和README外，不允许既有记录变化。六张预存脏图、两处正式报告及旧原生原件保持原始哈希。

## 4. 原件与执行记录

证据根：`outputs/msb_native_namespace_v3_synthetic_v1`。保留三阶段summary/source_snapshot/log、6份入口stdout/stderr（包括3份空stderr）、授权、来源身份、作者核对和前后保护记录。

`raw_evidence_manifest.json`包含16份原件身份：阶段9份、入口6份、归档时ledger快照1份。分别记录原始SHA256、仅用于Git比较的LF规范化SHA256及Git blob SHA1；没有改写原日志的换行。公开ledger是Step8/9完成前的时间点快照，最终作者核对和远端回读状态以本记录及后续外部核验为准。

准备阶段曾有不涉及runner的补丁/输出解析工具错误，以及归档清单脚本的路径拼接错误；都在调用合成入口之外处理，没有被当作RED或额外试验，未覆盖任何验收原件。实际runner始终为三个attempt-001。

依照人类授权，未调用技能的额外baseline/task-done测试、代理复查或清理；这是范围裁定，不是独立复查已完成。私有workspace、原件及所有旧失败目录保留。

另有一项格式小问题保留：新增测试文件末尾多一个空行，Git默认whitespace检查报告该项。为保持RED后冻结的新增helper以及GREEN/final源码身份，本轮不改它；仅对该文件显式排除blank-at-eof检查，仍检查行尾空白和缩进，其余作者文件使用默认检查。这不改变或重跑测试，不宣称默认whitespace检查全通过。

## 5. 当前结论与下一步

本轮结论仅为：**固定v3路径适配的作者合成验收通过，独立工程复查待做。**

没有Windows原生验收结果，也没有新的真实机制实验；原始trace操作系统故障原因仍未完全解释。测试通过不证明真实调度、文件替换、释放、证据交付或MSB-RRE科学收益已验证。旧源码的独立APPROVE不自动覆盖本次新代码。

建议下一步仅对本次固定提交作独立只读工程复查，检查三文件差异、八项反例和285项既有日志，不重跑测试或读科学数据；需另行批准。新的执行协议登记、单次原生验收及真实fit-only机制实验继续分别另批。

这一适配不是论文理论创新。研究主线仍须验证间隔缩放预算的纠错与损伤条件、角色非线性证据的非等价贡献、相对强对照的收益与代价；不因工程验收而预写网络优越性或工业应用结论。
