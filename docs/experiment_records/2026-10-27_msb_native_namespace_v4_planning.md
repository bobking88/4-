# MSB 固定v4命名空间：规格与计划登记

日期：2026-10-27。授权 `call_6912f128ec5a4fb0a8e847242f3c76cf`，答复“采用固定v4，批准仅编写规格与计划（推荐）”。文档基线d2fa040，源码仍固定已审8175134。

本轮交付[规格](../superpowers/specs/2026-10-27-msb-native-namespace-v4-design.md)和[单工作包实施计划](../superpowers/plans/2026-10-27-msb-native-namespace-v4.md)。状态 **DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN**，不是开发或运行批准。

用户已选择固定v4，放弃当前不必要的通用配置执行器。未来生产driver只需改变三个固定命名值，原harness只换两处证据路径；共享读取器、监督器、权限策略及科学网络均不改。v4不是新算法版本。

为获得明确RED而不靠旧夹具报错，拟追加4个独立v4方法，保留原297个完整方法名，未来driver140/final301仅目标。RED时旧正向v3夹具暂不动，新增4项独立使用v4；有效RED后才按清单迁移旧字面值。旧harness预期共4处，不是只有此前容易漏掉的两处。

静态AST清单发现31个相关字面值：4项RED前harness预期、26项有效RED后旧namespace迁移、1项原样保留的额外未知字段负例。见[迁移清单](../../outputs/msb_native_namespace_v4_plan_v1/fixture_migration_inventory.json)。清单只是未来要求，没有应用于源码。新增4项/helper、harness和4处预期在有效RED后冻结；不虚称整个测试文件跨RED/GREEN字节冻结。GREEN/final来源仍须相同。

未来仅原三个阶段、每阶段最多3次独占尝试（含失败），保留原件，原生接口全模拟，不扩模块、bootstrap或scratch张量例外；无效RED即停，不自动重试。真实v4协议与原生输出目录始终不在此工作包生成。

本轮只用隔离标准库AST、JSON解析及文本/哈希静态核对，未导入项目、调用项目函数或运行测试/前向/拟合/计时/探查/原生验收，没有读取真实科学输入、派代理或改正式报告。588条前保护记录保留重复项，涵盖旧源码、协议、合成/原生失败原件、六张预存脏图、两处报告和复查/兼容性档案；仅新增本轮文档与元数据。作者自检不是独立复查。

证据见[授权](../../outputs/msb_native_namespace_v4_plan_v1/authorization.json)、[前保护](../../outputs/msb_native_namespace_v4_plan_v1/protection_before.json)、[后保护](../../outputs/msb_native_namespace_v4_plan_v1/protection_after.json)、[文档核对](../../outputs/msb_native_namespace_v4_plan_v1/document_check.json)。原生监督契约、N1–N4、480秒累计、case limits、接电/单次/失败关闭、46/56来源集合及其他43/53源码保持不变。

请先审核这两份新文件，开发与有限合成验收尚须独立批准；独立工程复查、可执行协议登记、单次原生验收和真实机制实验仍分别另批。工程适配不构成网络理论创新，当前没有新MSB-RRE科学收益证据，也不更新正式报告或预写阳性结论。
