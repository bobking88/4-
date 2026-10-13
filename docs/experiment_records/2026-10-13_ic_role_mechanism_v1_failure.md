# IC-RFA-B v1 单次机制实验执行失败

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent run。
- Origin Date: 2026-10-13；Version Label: ic_role_mechanism_v1_failure。
- Verification Status: `EXECUTION_FAILED_BEFORE_FEATURE_DESERIALIZATION`。
- 授权登记提交：`1ee406c6b2cb53a00d066b4f47de0ae1cef16783`；执行源码为 b412dfc，34 文件未改。
- 本次授权已消费，目录不可续跑；没有矿物效果数据，不做统计解释或成功声明。

## 1. 单次执行事实

| 项目 | 记录 |
| --- | --- |
| 开始 UTC | 2026-10-13T05:32:48.751025+00:00 |
| 结束 UTC | 2026-10-13T05:33:23.009229+00:00 |
| 全流程耗时 | 34.265 秒；注册硬上限 7200 秒 |
| 整体 / 阶段终态 | FAILED / FAILED |
| 已启动任务 | 一次 prepare_fold0，exit_code=1，约 5.860 秒 |
| 完成神经轨迹 / 温度拟合 | 0 / 0；108 条训练轨迹尚未启动 |
| PCA / 温度 / 神经优化实际调用 | 均未到达；在 worker 活动阶段检查失败，计算消费锁未创建 |
| 最终推理验收 | 未启动，没有接受 provisional 分析 |
| 自动重试、续跑 | 无 |

本轮仅用冻结 run_stage 入口。前置/登记阶段读取了规定 fit/assignment 文件字节作 SHA 校验，但未反序列化特征、准备内折、读取 stop、原图或组合缓存。real_data_opened=false 不是单独的访问审计；结合调用顺序、失败堆栈及 task_started.json/result.pt 缺失，支持计算未开始的判断，不冒充操作系统级访问追踪。

## 2. 原因与证据

worker 报错：`ValueError: Worker requires an active exclusive stage.`

[执行器](../../scripts/run_ic_role_mechanism.py)先把 PREPARING 写入 stage_state.json，随后 run_stage 只在内存将 status 改为 RUNNING 并添加历史项，未在派发首个 worker 前写回。worker 从磁盘读取旧 PREPARING，严格活动阶段检查因而失败；发生在 task_started.json 创建、工作线程电源请求及 load_fit_bundle 前。

不是随机数、模型初始化、标签、超时或接电问题。最终 state_history 含内存中的 RUNNING，不证明首个 worker 启动前该状态已正确发布。失败终态由异常路径写回，旧失败证据不手工改成成功。

[执行器测试](../../tests/test_run_ic_role_mechanism.py)的 simulate 用 mock 替换 run_supervised_job，未让真正 worker 读取父进程新发布的状态；独立 worker 测试则预先手写 RUNNING。这解释了合成控制平面验收未发现该边界缺陷。93项合成通过是历史记录，不能据此声称父子进程端到端启动已通过。本次诊断仅查日志和源码，没有新增/运行修复测试。

## 3. 清理与完整性

父 PID=14016、首 worker PID=36800，退出后只读进程查询均不存在。stage_state 的 alive=false，任务 attempts=1；父阶段和任务监督器的临时系统防待机请求均 acquired=true/released=true。最大轮询间隔约0.063秒，最大暂停差约0.012844秒，无监督中断证据；未改永久电源计划。

最终目录只有以下四个生成文件，无结果张量、检查点、预处理状态、分析或 final_acceptance：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| [stage_state.json](../../outputs/training/ic_role_mechanism_v1/development_fold_0/stage_state.json) | 2156 | `978aa7001668ce569f20cbade0c66bde9e297d89317810ee0fb2805aedf75d01` |
| [whole_experiment.json](../../outputs/training/ic_role_mechanism_v1/development_fold_0/whole_experiment.json) | 595 | `672d704f1d2784e6328645c69a407b6d21160d0d31dcc7010d7cb821f64a8ea1` |
| [prepare packet](../../outputs/training/ic_role_mechanism_v1/development_fold_0/runs/prepare_fold0/packet.json) | 486 | `aac7ecd821dfe65f1d05b02b5e5953afb6b19932c7ea5ab2483add7707da7032` |
| [worker.log](../../outputs/training/ic_role_mechanism_v1/development_fold_0/runs/prepare_fold0/worker.log) | 538 | `672bf5a8bb4d9acf6af2967656bbbee7b55e34b3e2540f4354661593361c516e` |

34个冻结源码 SHA 仍与获准协议一致。主项目/工作树唯一正式报告均保持 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。六张用户 PNG 改动未覆盖、未暂存；旧负结果/失败目录不改。

归档仅为这四个原始证据文件在根 .gitattributes 添加 -text 换行保留规则，沿用已有归档模式，避免 Git clean 改变原生 SHA；不修改四个文件内容或任何冻结源码。GitHub 只提交这些轻量证据及文档状态，不上传张量/权重。

## 4. 结论与下一步边界

本次只能得出“真实执行入口存在状态发布缺陷并按约定停止”，不能判断 C-S-R 效果，也不是角色方法/理论机制的负结果。没有100步指标、纠错数或初始核数据，不编造图表或创新效果。

推荐另行批准**仅执行安全修复与合成验收**：先用合成回归复现首个派发时磁盘状态错误，再最小修复 RUNNING 发布顺序，验证父子边界、失败停止/消费锁及完整纯合成回归；网络、损失、网格不改，不读真实输入、不续跑本目录、不训练、不改正式报告。本次仅提议，尚未实施。

任何真实重跑须修复验收后重新登记源码、协议及独立输出目录，并另获人类单次运行批准。本轮许可不继承，不自动开启新的机制/完整效果或 stop 实验。
