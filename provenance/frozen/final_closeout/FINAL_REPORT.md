# ChainPrint 最终实验收口报告

状态：`COMPLETE_FINITE_CLOSEOUT_NO_AI_RETRY`

日期：2026-09-25

本报告只汇总本轮冻结协议下的有限证据。没有修改论文正文或正式主表，没有重训模型，没有新增权重更新，也没有为了填补缺失支持而复制、替换或重新生成样本。

## 1. 最终结论

1. 严格 matched 的 `G=8` 常规查询接口已补齐。对共同完整覆盖的 `K={2,3,4,6,8}`，ChainPrint 的 Top-1 和 MRR 均为 `1.000`，180/180 个独立源均可评估，dataset-stratified source bootstrap 95% CI 均为 `[1.000, 1.000]`。
2. 该结果关闭了此前 ChainPrint 使用 `G=4`、比较方法使用 `G=8` 的接口不一致，但不产生“更多示例继续改善性能”的证据：`K=2` 已饱和，至 `K=8` 无可测斜率。
3. ChainPrint 与 Joint fitting 在共同 K 范围完全持平，二者配对差为 0。因此本轮不支持“ChainPrint 优于 Joint fitting”，只支持它们在该有限固定设计上均达到上限。
4. 最强非 Joint fitting 外部方法是 Editprint。ChainPrint 相对 Editprint 的 Top-1 配对优势在共同 K 范围为 `+0.6500` 至 `+0.6611`，所有 95% CI 均严格大于 0。
5. AI 检索缺口不能通过合法只读恢复补齐：9072/9072 个计划请求均已历史调用，其中 8004 个完成、1068 个为未变更安全检查器拒绝（1036 个唯一请求哈希）；没有从外部缓存发现精确且摘要可验证的命中。新的 editor 调用预算冻结为 0。
6. 因此 AI 检索的 Behavior error、best-in-library error 和 Regret 仍不可定义；现有 verified Top-k utility 继续保留，但不得重命名为 Regret。

## 2. 严格 G=8、K=8 截面

所有方法在同一 3 个数据集、180 个独立源、16 个候选以及同一 `G=8, K=8` 接口上汇总。置信区间使用冻结的 10,000 次 dataset-stratified source bootstrap；随机基线为解析值。

| 方法 | Top-1 | 95% CI | MRR | 95% CI |
|---|---:|---:|---:|---:|
| ChainPrint | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| Joint fitting | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| Editprint | 0.3389 | [0.2722, 0.4111] | 0.5362 | [0.4850, 0.5879] |
| TSOM++ | 0.0667 | [0.0333, 0.1056] | 0.2388 | [0.2052, 0.2746] |
| TMFNet | 0.0722 | [0.0389, 0.1111] | 0.2203 | [0.1867, 0.2576] |
| SeqFakeFormer++ | 0.0500 | [0.0222, 0.0833] | 0.2139 | [0.1830, 0.2472] |
| TransDetect | 0.0600 | [0.0487, 0.0749] | 0.2075 | [0.1932, 0.2241] |
| Random | 0.0625 | exact | 0.2113 | exact |

## 3. K 曲线与覆盖边界

共同严格范围仅为 `K={2,3,4,6,8}`：每个 K 均有 180/180 个完整源。

| K | ChainPrint Top-1 | ChainPrint MRR | ChainPrint - Editprint Top-1 | 配对 95% CI |
|---:|---:|---:|---:|---:|
| 2 | 1.0000 | 1.0000 | +0.6556 | [0.5889, 0.7222] |
| 3 | 1.0000 | 1.0000 | +0.6611 | [0.5944, 0.7278] |
| 4 | 1.0000 | 1.0000 | +0.6556 | [0.5889, 0.7222] |
| 6 | 1.0000 | 1.0000 | +0.6500 | [0.5778, 0.7167] |
| 8 | 1.0000 | 1.0000 | +0.6611 | [0.5889, 0.7278] |

`K=10` 与 `K=12` 仅有 90/180 个注册支持前缀，固定分母结果为 Top-1=0.5、MRR=0.5；`K=16` 为 0/180。它们是覆盖诊断，不是纯粹的 K 学习效应，不能与共同范围混成一条性能曲线。

## 4. AI 支持机制与检索闭环

既有机制确认资产保持原判定 `COMPLETE_WITH_UNEVALUABLE`：计划 572 个唯一请求，569 个允许的新调用中 559 个接受、10 个被未变更安全检查器拒绝，另有 3 个历史精确拒绝按合同阻断。可完整评估的 85/85 个长链候选对在默认与替代配置下保持相同赢家，且均选择真实候选。该结果只支持“机制存在”的有限存在性结论，不支持其对六个目标普遍成立。

本轮只读恢复审计进一步确认：

- planned/called：9072/9072；never-called：0；unknown/started：0；
- COMPLETE：8004；UNCHANGED_CHECKER_REFUSED：1068；
- 唯一历史拒绝请求：1036；
- 精确可接受外部缓存命中：0；摘要验证命中：0；
- 终止决定：`NO_LEGAL_RECOVERY_CALLS_ALL_REQUESTS_ALREADY_CALLED_AND_REFUSALS_IMMUTABLE`。

完整 8-probe 候选覆盖最多只达到 3/12 个源；仍缺 3020 个 replay atoms、26 个冻结排名不完整。因此本轮没有构造或推断未定义的 Regret。

## 5. 工程失败与协议边界

两个工程失败均被保留：v1 在载入原协议时遗漏 `Path(...)` 转换；v2 在排名辅助函数中重复传入 `method_complete`。两者都在模型、查询和评分结果落盘前失败，未形成科学数据，也未覆盖既有资产。v3 只修复接线错误并完成 1440 条记录，其中 1080 条完整、360 条固定支持前缀失败。

严格运行记录：

- records SHA256：`a7d7f29e294b5a459eeddf7bd428e6bcfa0db35be2076aa2d4d304b75e9e9f2e`
- protocol SHA256：`ed2b734d8c8441fba85aaaae065b54b8a60e42c42a32f34d184b6f4455d097e3`
- run contract SHA256：`4362395c07433d43e0435004eb9bf97f678939b04ec7077e33c32fbb4c7284ce`
- support binding SHA256：`6a33270fbde63ca6bf33cf35952bde20e49613fa04e01f1cb84d462608dd1c31`

## 6. 证据等级与禁止外推

该闭环是固定模型、固定支持池、固定源集合和固定候选设计下的有限确认。它不直接证明开放世界未知操作族泛化、跨数据分布 SOTA、真实部署鲁棒性，亦不支持由 `K=10/12/16` 的覆盖失败推断学习行为。所有论文级表述必须保留这些边界。

## 7. 机器可读资产

- 聚合摘要：`aggregate_v1/SUMMARY.json`
- 绘图源数据：`aggregate_v1/K_CURVE_PLOT.csv`
- 覆盖表：`aggregate_v1/K_CURVE_COVERAGE.csv`
- 配对差与 CI：`aggregate_v1/K_CURVE_PAIRED_SOURCE_DIFFS.csv`
- 逐查询记录：`aggregate_v1/K_CURVE_PER_QUERY.csv`
- 最终工作簿：`aggregate_v1/FINAL_RESULTS.xlsx`
- AI 只读恢复审计：`ai_retrieval_readonly/RECOVERY_AUDIT.json`
- AI 机制确认：`../ai_support_selected_mechanism_confirmation_20260925_v3_engineering/`

远端正式根：`/home/yz/chainprint_t1_20260917/final_experiment_closeout_20260925_v1`

