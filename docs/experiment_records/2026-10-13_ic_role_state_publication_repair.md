# IC-RFA-B 执行状态发布修复与合成验收

## Material Passport

- Origin Skill / Mode: systematic-debugging / test-driven-development / synthetic acceptance。
- Origin Date: 2026-10-13；Version Label: ic_role_state_publication_repair_v1。
- Verification Status: `SCOPED_SYNTHETIC_VERIFIED_REAL_UNVERIFIED`。
- 当前批准项：`["request_user_input_async","call_b5eec178b6d34f319532e1984755dafb",0]`；人类回答“批准修复与合成验收”。
- 修复基线：`99f163df58ca76cfacc047818a2611a209abb2d8`，原失败证据保持不变。
- 仅修复执行器与合成验收；不继承旧真实运行授权，不创造新真实协议或结果。

## 1. 最小生产改动

[执行器](../../scripts/run_ic_role_mechanism.py)只增加一行 `_json(output/"stage_state.json", state)`：在内存 status/state_history 进入 RUNNING 后、生成/派发任何任务前关闭写入状态文件。真实 worker 的活动状态检查不放宽；写入失败仍走既有失败终止路径，禁止派发。

网络、损失、路由、初始化、预算、控制矩阵、检查点、输入范围、最终验证公式与既有监督器均未改。34文件执行闭包中仅执行器 SHA 改变：

| 项目 | SHA-256 |
| --- | --- |
| 旧执行器（获准失败版本） | `2afedbd43e9c7820c2e5473b6b3132ae4f169d99642c80f3b9aaebb9cc5a6f3b` |
| 本次修复执行器 | `423fad79bee6d5a351589319cb03b98a8097b958615a9b9e1fd274ef921f302a` |

没有改写原 v1 实验目录或旧协议身份；当前源码刻意不再匹配已消费的 v1 真实协议。

## 2. RED 到 GREEN

先在[现有执行器测试](../../tests/test_run_ic_role_mechanism.py)增加三个回归，再修改生产代码：

| 回归 | 修复前观察到的失败 | 修复后行为 |
| --- | --- | --- |
| 实际合成 worker 读取父阶段文件 | 首个 worker 拒绝 PREPARING，阶段 FAILED | 112个同进程合成控制平面任务全部通过真实 worker 状态/消费锁逻辑；无优化/温度拟合 |
| RUNNING 写入失败 | 先派发一次任务，之后写入才失败 | 首次发布失败即 FAILED，未建立 runs/，不派发 worker |
| 原生合成父子启动/失败停止 | 真正子进程在活动阶段门检查失败，第二任务未到达 | 一个短原生合成子进程成功；第二派发注入无子进程的 FAILED，第三不派发，进程退出/防待机释放 |

三回归在生产改动前确实失败，RED 3/3；修复后 IC-RFA 合成组63/63通过，其中三个回归均通过。原生边界在RED/GREEN各使用一个短合成子进程，是功能验证，不是预算计时或360秒空闲验收；没有运行旧计时入口。

IC-RFA命令：

```powershell
& 'D:/成信工科研/人工智能选矿/.venv-training/Scripts/python.exe' -X utf8 -W error -m unittest discover -s tests -p '*ic_role*.py' -v
```

旧纯合成回归只加载 test_lc_role_data、test_lc_role_adapter、test_lc_role_audit、test_lc_execution_safety 四模块，33/33通过。初次按包名直接调用缺少 tests/ 搜索路径，27项通过、1个 lc_role_fixtures 导入错误；改用下列加载方式后通过，没有改旧源文件/测试。该调用问题不隐去，也不误算为生产修复。

```python
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
modules = ["test_lc_role_data", "test_lc_role_adapter",
           "test_lc_role_audit", "test_lc_execution_safety"]
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules)
result = unittest.TextTestRunner(verbosity=2).run(suite)
```

以上96项是静态限定的合成检查，不是全仓库测试声明。未运行可能打开真实数据的裸测试发现。合成数据测试可能使用随机张量、临时 torch 状态和短步数优化，但不读既有真实 fit/stop 或矿物图片，不占用或冒充108条真实轨迹。测试终端耗时不是新的真实预算估算。

## 3. 证据与完整性

- [RED证据](../../outputs/ic_role_state_publication_qa_v1/red_evidence.json)：工具末段，含完整三项失败摘要；默认控制台编码导致路径乱码，如实保留，不能称完整控制台日志。
- [GREEN证据](../../outputs/ic_role_state_publication_qa_v1/green_evidence.json)：63项最终摘要及工具末段；前段进度在其他工具块返回，未伪装成完整日志。
- [旧回归结果/调用修正](../../outputs/ic_role_state_publication_qa_v1/legacy_evidence.json)：33项结构化摘要，非完整终端转录。
- [完整性核验](../../outputs/ic_role_state_publication_qa_v1/integrity_evidence.json)：除执行器外33个执行源码未变，失败目录四个文件原始字节未变，正式报告未改。

当前源码对旧真实协议的 validate_protocol 校验在反序列化之前拒绝，返回 `Registered execution source changed.`；此检查只读协议/源文件，fit加载器和torch.load哨兵调用数均0，未读取真实特征。

主项目与工作树唯一正式报告均保持 SHA-256 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。六张用户已有PNG改动不暂存、不覆盖。无原生空闲验收、预算计时、真实重跑、stop评价或新图像读取。

## 4. 复查与剩余边界

已请求一次独立只读工程复查，但代理因调用额度限制终止，未返回任何审核结论，现已关闭。不写“独立审核通过”，也未重派代理或绕过限制。详见[复查状态](../../outputs/ic_role_state_publication_qa_v1/review_evidence.json)。独立复查缺口仍保留，作者自审不能替代它。

作者按审核模板检查一行生产差异与三个回归，未发现新的作用域内代码问题：RUNNING文件写入关闭后才派发，异常进入已有FAILED路径，worker门检查未放宽，旧授权因源码身份变化被拒绝。作者自审不核准合并或真实全流程；未判断理论新颖性、数学独立性或矿物效果。原生功能检查依赖Windows接电，非Windows跳过；本机本次没有跳过。

本次证据支持“首个worker启动的状态发布合同已修复，并保持失败停止约束”，不证明完整真实准备/108条轨迹/最终验收在7200秒内成功，更不证明角色净纠错或理论创新有效。旧合成计时仍保留原源码身份，本轮不更新估算。

下一步须先登记修复后的源码身份、独立新协议/输出目录，再单独取得真实单次运行批准。不得将旧失败目录、旧已消费许可或本次修复批准当作重跑许可；正式报告和论文效果整合仍须真实证据及后续授权。
