# 投稿工作稿限定引用核对

日期：2026-11-18。输入为本目录英文正文、补充材料和已归档中文稿的引用；基线整理提交为 `edcb714`。本次只收口已有文献及已采用的Focal Loss、AdamW原始出处，不搜索新的网络候选，不构成系统综述、全文重新评审或首创性鉴定。

## 1. 本轮修订

1. 正文保留既有1–10号引用，补入11号Focal Loss和12号AdamW；补充材料增加SRef2/SRef3。合计15个编号位置、13篇去重文献。
2. ResNet引用从2015年作者预印本统一为CVPR 2016正式版本，补正式页码。两个年份对应不同版本，并非把预印本年份认定为错误。
3. MineralImage5k的Crossref记录列作者为Andrey Kuznetsov；正文由无依据的`A.V.`改为该记录支持的`A.`，BibTeX使用完整名字。
4. 区分高层次综合的多粒度程序表示、翻译路由、来源域混合路由与本研究的真实标签条件两专家概率后悔。外部论文的保证不能直接移用为本网络保证。
5. 新增[references.bib](references.bib)，供后续排版使用；不添加未核实的DOI、页码或结果数值。

## 2. 文献身份与主张对照

| 正文/补充编号 | BibTeX键 | 已核对版本与主来源 | 本稿使用及限制 |
|---|---|---|---|
| 1 | `wu2022` | [Wu等，Minerals 2022, 12(11), 1338](https://www.mdpi.com/2075-163X/12/11/1338)，DOI `10.3390/min12111338` | 伴生矿物图像的多标签识别背景；不是2024年论文，不将其结果与本稿四角色准确率直接比较。 |
| 2 | `ji2024` | [Ji等，Minerals 2024, 14(6), 627](https://www.mdpi.com/2075-163X/14/6/627)，DOI `10.3390/min14060627` | 已有矿物OOD工作；不能据此把本稿闭集或摄影者留出评价称为未知矿物识别。 |
| 3 | `nesteruk2023` | [出版社记录](https://www.sciencedirect.com/science/article/pii/S0098300423001188)，Computers & Geosciences 2023, 178, 105414；[Crossref元数据](https://api.crossref.org/works/10.1016/j.cageo.2023.105414) | 原始标本零样本识别与描述基准；核对题名、作者、卷号、文章号与DOI，未声称在同一数据协议上复现。 |
| 4 | `zhang2025` | [Zhang等，Future Internet 2025, 17(12), 553](https://www.mdpi.com/1999-5903/17/12/553)，DOI `10.3390/fi17120553` | 铁矿图像多视图进化融合先例；不同任务和采集条件，不构成本文直接性能对照。 |
| 5 | `jacobs1991` | [作者发表记录](https://www.cs.toronto.edu/~hinton/mixex.html)、[作者论文](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)，Neural Computation 1991, 3, 79–87 | 专家混合与门控已有基础；本稿不声称发明门控。 |
| 6 | `li2025` | [AAAI正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/34033)，2025, 39(17), 18476–18484；DOI `10.1609/aaai.v39i17.34033`；[作者预印本](https://arxiv.org/abs/2410.19225) | 高层次综合中的分层专家与多粒度表示，不是矿物种类到角色的直接先例；使用正式2025版本。 |
| 7 | `liang2025` | [ACL Anthology正式记录](https://aclanthology.org/2025.acl-long.1040/)，ACL 2025, 21433–21445；DOI `10.18653/v1/2025.acl-long.1040` | 翻译中的任务引导与上下文路由；不能仅凭本任务不同就断言网络全新。 |
| 8 | `dann2025` | [NeurIPS正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/39c60dda48ebf0a2e5dda52ce08eb5c8-Abstract-Conference.html)，2025, 38；DOI `10.52202/085713-1356` | 未知来源域混合下的路由与后悔分析。其学习设置、参照和保证不同，本稿不继承其定理。 |
| 9 | `tan2019` | [PMLR正式记录及BibTeX](https://proceedings.mlr.press/v97/tan19a.html)，ICML 2019, PMLR 97, 6105–6114 | EfficientNet基础骨干出处；不是本项目结构创新。 |
| 10 | `he2016` | [CVF会议记录](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)，CVPR 2016, 770–778 | ResNet正式版本，用于基线及迁移骨干出处；不再留“会议元数据待核对”。 |
| 11 / SRef2 | `lin2017` | [CVF会议记录](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)，ICCV 2017, 2980–2988 | Focal Loss基础出处；本地加权分类变体不作为新损失贡献，引用不能补证历史超参数。 |
| 12 / SRef3 | `loshchilov2019` | [ICLR/OpenReview记录](https://openreview.net/forum?id=Bkg6RiCqY7)、[作者版本及发表说明](https://arxiv.org/abs/1711.05101)，ICLR 2019 | AdamW解耦权重衰减出处；首个预印本为2017，正文采用2019会议版本。不能由论文推定本项目未知的学习率等设置。 |
| SRef1 | `niculae2017` | [NIPS会议记录](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2d1b2a5ff364606ff041650887723470-Abstract.html)，2017, 30 | 标准正则化概率映射背景；不是本稿sigmoid解析解或熵正则化的首创证明。 |

## 3. 来源可访问性与核对边界

- 直接读取了MDPI、PMLR、ACL、NeurIPS/NIPS官方页面以及相关作者公开页面。
- ScienceDirect直连返回403；使用该出版社被索引的摘要和Crossref REST返回的元数据交叉核对。Andrey Kuznetsov来自Crossref完整作者记录，未擅自增加中间名。
- CVF直连返回403，AAAI和OpenReview直连遇到防自动化页面；对应条目使用索引中的官方会议记录/正式PDF元数据，必要时辅以作者预印本的发表说明。不将访问失败写成“已阅读全文”。
- 未引用第三方分区或非官方论文摘要，不以搜索摘要代替数学证明审查。13篇均已匹配到可识别的原始记录，但本轮没有重新通读13篇全文，也没有证明检索范围穷尽。
- BibTeX只登记已核实的身份字段，未强行补足所有期刊格式细节。最终编号由排版稿及补充文件各自管理，不能因为两个文件重复引用Focal/AdamW而重复建立文献实体。

## 4. 对创新表述的约束

本轮支持的是更准确的相关工作定位，不是新增实验或创新成功证据。应保留：

- 可写：特定矿物角色任务中的跨粒度路由监督、路由后悔的条件分解、机制效果与分类效果的区分。
- 不可升级为：首次提出专家门控、首次提出后悔路由、发明熵正则化软目标，或证明新网络比普通残差结构表达能力更强。
- 仍未建立：稳定的总体分类优势、工业分选收益、独立现场泛化及更强的新网络结构创新。
- 未验证候选继续放附录或后续方向；本轮不重新开启训练、筛选和原生执行验收。

## 5. 待完成的收稿工作

作者仍需核准引用的语义及作者事实，之后完成可编辑公式的投稿排版和逐页检查。历史配置缺口、旧目标召回标准差冲突仍按[配置追溯](archive_trace_2026-11-17.md)披露，不因补齐引用而视为解决。本次不修改正式技术报告。

## 6. 文稿静态检查结果

本轮检查了固定格式BibTeX的13个唯一键、作者/题名/年份/刊会名/URL字段，以及正文12项、补充3项的题名和键映射。正文引文首次顺序为1–12，补充SRef1–3均被引用。此为字段与映射检查，不宣称BibTeX已编译成功。

本机Pandoc未能启动：PowerShell调用返回标准输出配置异常，经Python子进程调用返回`WinError 623`（系统DLL重定位）。未安装依赖、修改运行环境或继续尝试编译。文稿静态检查首个命令构造中存在替换字符串转义问题，修正后检查通过；这不是项目代码或科学实验失败，不改写为首次全程成功。

通过的检查还包括：正文R1–R12与中文源一致，补充R9a–R9e仅翻译分段条件文字；原表10行数值匹配中文源，表5七行28个数值匹配历史汇总，12个来源哈希匹配。摘要170词、关键词6个，图序1–4、表序1–5，最终38处本地Markdown链接存在。中文正文和两处正式报告哈希均未改变。检查未读取预测、权重或矿物原图，也没有项目测试、训练或科学指标重算。

尚未验收：BibTeX编译、Word/LaTeX可编辑公式排版、最终PDF逐页可读性、作者声明。不能把本记录称为完整投稿包验收。
