# MSB-RRE v1 原生实现与合成验收

实际验收日期：2026-10-15。文件名沿用2026-10-14规格/实施计划日期。
授权：`call_978b15d460804fa2982cb750a1d4b4ad`，仅Tasks1--8原生代码与合成验收。
状态：`NATIVE_SYNTHETIC_ACCEPTED_PENDING_INDEPENDENT_REVIEW`，不是新网络真实效果结论。

## 1. 交付及证据

已实现八个新模块：prepared身份合同、中心化角色网络、目标函数与四个凸强对照、条件性质审计、固定轨迹、单次监督执行器、逐图汇总/最终复算、Python公式结构图。

限定联合验收命令：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test*msb*.py' -v
```

八个模块共83项测试通过；[完整原生日志](../../outputs/msb_role_synthetic_v1/synthetic_tests.log)保留逐测试结果。
各模块数目依次为data7、model9、controls10、audit13、training9、runner15、analysis16、figure4。
另一次[模拟控制面验收](../../outputs/msb_role_synthetic_v1/control_plane_summary.json)完成80个模拟任务；真实数据读取=false、真实神经拟合=0、真实凸拟合=0、真实交付复算=false。
该模拟不调用原生电源API，不是新的接电、空闲、超时清理或操作系统可靠性验收。

[源码身份](../../outputs/msb_role_synthetic_v1/source_identity.json)含42个执行源码SHA-256（旧34+新8）；它不是新的真实运行协议或授权。
[旧源码保持记录](../../outputs/msb_role_synthetic_v1/legacy_source_preservation.json)证明34个旧源码未变。
[保护文件记录](../../outputs/msb_role_synthetic_v1/protected_artifact_preservation.json)证明两份正式报告、六张用户既有PNG、旧协议及旧IC规格均保持原哈希。

## 2. 理论与网络证据的准确含义

候选网络不改EfficientNet-B0主干，在冻结视觉h64和既有预测证据e23之上建立三个独立rank4分支：

\[
f_{\theta_j}(h,e)=a_j^T[\tanh(U_jh+u_j)\odot\tanh(V_je+v_j)]/\sqrt4+\zeta_j,
\quad s_j=f_{\theta_j}-f_{\theta_j^0}.
\]

\[
m_j=\ell_T-\ell_j,\quad R_j=2|m_j|+0.25,\quad d_j=R_j\tanh(s_j),
\quad q=\operatorname{softmax}[\ell_T,\ell_{Ti}-d_{Ti},\ell_G-d_G,\ell_M-d_M].
\]

共1083个可训练参数，另存1083个固定初始参照参数；推理仍计算两条参照路径，不宣称零额外成本。
e23为E18、log(p_base)4及contradiction1的既有标准化证据，不是新增物理模态。

[条件性质数值验收](../../outputs/msb_role_synthetic_v1/conditional_theory_checks.json)记录：

| 条件性质 | 本轮数值证据 | 不支持的延伸结论 |
|---|---|---|
| 各类单图内部见证 | 125组构造间隔×4类=500点；最小胜出间隔约0.125、负类间约0.25；相对能量0.25 | 不是500张矿物图，不是全样本联合可学习或分类一致性证明 |
| 配对边界可控 | 有限差分与自动微分最大差约4.51e-9；所测边界导数最小0.25，满足规格保守下界0.1875 | 不是整批梯度下界、全参数条件数或全局收敛保证 |
| 规范化偏置Jacobian | 见证处为0.75I | 不是多样本NTK正定或参数可识别 |
| 非线性跨输入交互 | 构造混合Hessian与解析式误差0 | 只排除指定仿射/可加形式，不能排除一般MLP或共享双线性头 |
| 限制与反例 | 同输入冲突标签最多正确1个；高置信正确锚点可被见证翻转；配对平局可能不是全局胜者 | 不允许声称扩大预算只修错、不损害正确样本 |

以上是既有条件推导的有限数值验收，不以测试代替数学证明或正式查新。
已实现精确共享双线性块嵌入；共享对照包含MAIN。bias-only子模型也满足部分理论性质，因此公式成立本身不能证明独立视觉角色分支具有增益。
仅置换和符号等确切对称性足以说明参数并非唯一；不得把一般连续U/V缩放当作tanh网络的严格不变性。

## 3. 强对照与效果证伪实现

拟议矩阵仍为11单元、63神经轨迹+12凸拟合=75次拟合、0次新PCA/T0拟合、100步固定主点。当前未执行该真实矩阵。
MAIN、SHARED-BIL、CONCAT-MLP、ADD、FIX1、UNB-SCALE、NOAUX及四个凸对照均按冻结规格实现。
BIAS3-CVX、LIN-UNB、CLASSWISE4、REPAIR4-COMP的合成实际求解及独立重算证书通过；凸解不是三个种子的独立重复证据。
REPAIR4-COMP是受控竞争特征对照，不是原REPAIR论文全方法的复现。

主点晋级须对SHARED、MLP、BIAS、LIN、CLASSWISE、COMP、T0七参照逐一满足：三个种子每个目标净纠错至少+1/66、Ti/G/M各误入目标数不增、Macro F1不降，且12个凸证书有效。否则`NO_ROLE_PROMOTION`。
相同argmax但NLL变化、只赢某一种子、偏置或共享对照打平均不得晋级；不以400步结果补救100步主点。
Hadamard必要性另受ADD对照门约束；ADD去除显式乘积不等于消除所有非线性交互。预算和辅助损失分别报告，不拼成未经检验的因果创新。
这是描述性开发门，不是显著性检验、功效设计或独立确认。

最终复算的执行计数通过合成调度验证：567神经检查点验证前向+12凸验证前向+36主点关支路前向=615；另有12次训练侧凸梯度证书核验，共627次前向/梯度核验。后续协议必须分别登记两类，不再笼统写成“总共615次”。
本轮逐张量复算覆盖真实计算的短合成轨迹及四类合成凸解；完整615+12真实矩阵尚未执行。调度计数测试含明确标识的元数据fixture，绝不是75次真实训练或400步训练证据。

## 4. 执行边界与修复记录

执行器只提供synthetic及单独授权的fit-mechanism入口。先发布RUNNING再派发，独占输出、旧/代码许可拒绝、失败即停、剩余累计预算、父子释放证据和失败退出码均有回归。
分析/最终复算通过本地一次性上下文绑定既有活跃child guard，不创建新的电源请求或独立验收时钟；无监督API先拒绝再开文件。
本轮修复了共享偏置缩放、凸求解最后一步证书、空检查点、失败退出码、释放证据、过早协议路径读取、worker终态、跨模块losses字段与分析监督绑定等合成暴露的问题。没有借此改变网络、损失、单元矩阵或重跑旧失败目录。
仍未原生验证进程硬杀后孤儿/陈旧RUNNING的所有OS情形；不能把模拟测试称为这些情形已被完全证明安全。

## 5. 结构图与作者自审

[PNG候选图](../../outputs/msb_role_synthetic_v1/assets/msb_rre_v1_architecture.png)、[可编辑SVG](../../outputs/msb_role_synthetic_v1/assets/msb_rre_v1_architecture.svg)、[PDF](../../outputs/msb_role_synthetic_v1/assets/msb_rre_v1_architecture.pdf)和[结构元数据](../../outputs/msb_role_synthetic_v1/assets/architecture_metadata.json)均在新目录。
Python/matplotlib生成；300dpi PNG、SVG文本、PDF TrueType。已目视检查PNG及由Python调用现有Poppler渲染的[PDF预览](../../outputs/msb_role_synthetic_v1/assets/pdf_preview.png)。自动文字越框及连线穿框均为空；旧图和DOCX未覆盖。

作者按41840cd..3d48cd4全分支差异及实施计划五项Review Focus自审：固定参照、可达/伤害反例、共享包含与凸对照、prepared/OOF合同、监督发布与剩余预算均有针对性测试。未新增确认的阻断代码问题；这不是独立审核通过。
独立工程/数学复查、新协议登记、真实fit机制实验、未暴露确认及正式报告更新均未完成。

## 6. 执行裁决与下一步

1. 不跑全库discover：以明确数据隔离优先，只验收八个新合成模块，不能宣称全库回归；代价是旧库更广泛兼容性未覆盖。
2. 独立审核单独待批：先做作者自审，代价是缺少新视角审核，不能据此批准真实运行。
3. 为主点关支路记录增加一次实际合成100更新轨迹，不用2步状态冒充100步；代价为小量额外单元测试计算。
4. 在615验证回读之外独立重算12个训练侧凸证书；不重放优化。代价为12次额外前向/梯度核验，真实协议必须显式计入。
5. 分析API新增既有guard的一次性上下文绑定；代价是未来调用绑定不合规时直接拒绝，避免无监督最终回读。

下一步只申请一次新的独立只读工程/数学复查及结论归档。若获准，先检查上述源码、数学边界、83项既有日志及真实运行边界；不重跑测试/前向/拟合、不读取真实张量或原图。问题先记录，修复另议。当前不创建真实MSB协议，不启动真实实验，不改正式报告。
