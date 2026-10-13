# Research Engine TODO

状态：Contract 输出、Card review 和 DataLoop 已实现；Data review 之后的冻结、Feat、Sig、Algo 仍未实现。

这份文件只记录当前有效的设计任务。历史讨论、失败方案和旧实现保存在仓库外部的 agent log，不在这里重复维护。

当前先用 `research-ci-lab` 的 SK Hynix proxy case 验证 Contract → DataLoop 的最小闭环。

不继承旧 `research-ci` 的代码、API、数据模型或兼容约束，也不设版本迁移计划。

## 状态标记

- **决定**：方向已经确定
- **待讨论**：需要继续推演
- **进行中**：正在实现
- **完成**：有验证证据
- **后置**：当前阶段不处理
- **废弃**：旧设计，不再使用

每一项任务都要写清楚：目的、输入、产出、完成条件、验证方式和依赖。

## 0. 当前边界

- **决定**：Contract 是整个研究任务的大边界，内容保持简单而稳定。
- **决定**：Contract 不预先定义 label、outcome、horizon、target、收益门槛或具体方法。
- **决定**：原始数据源保持自己的频率，每个来源是一张独立表。
- **决定**：Agent 负责读取来源、拼接和处理，形成数据大表。
- **决定**：Engine 负责校验阶段协议、执行外部组装的 eval、保存 sidecar 证据和冻结产物；不修改主表。
- **决定**：所有 eval 都基于当前阶段主表只读执行。中间列、布尔值、指标、失败记录和无业务输出都属于 eval 运行空间或 sidecar，不写回主表。
- **决定**：四个插槽属于每个阶段内部，按该阶段的任务目的生成。
- **决定**：Data 阶段的四个插槽只判断数据正确性，不使用 outcome。
- **决定**：Feat 阶段只读取冻结后的 Data 产物。
- **决定**：label、target、signal 在后续阶段逐步探索。
- **决定**：Algo 阶段处理仓位、执行、成本和经济结果。

### 0.1 文件夹边界

- **决定**：`research-ci/src/research_ci/` 只保留协议模型、Engine 执行、Card 渲染、workflow 和 CLI；不把具体研究 case 放进 Engine。
- **决定**：`research-ci-lab/lab/<case>/` 由外部 Agent 自由维护 Contract、raw、Data builder、阶段 eval 和探索文件。
- **决定**：`research-ci-lab/.rci/runs/<run>/` 是运行交接边界，按 `contract/`、`data/`、后续阶段目录保存 Card、主表和 sidecar 证据，并保存 `workflow.json`。
- **决定**：Card 是给人和 Agent review 的可读入口，Parquet、JSONL 和 manifest 是可追溯证据；主表和 eval sidecar 分开保存。
- **完成**：当前最小目录已按上述边界实现，CLI 不要求把外部 case 复制进 Engine。

## 1. Contract

### 1.1 最小结构

- **完成**：当前第一版提供 `mission`、`scope`、`stages` 和 `invariants` 的最小结构。
- **待讨论**：继续收敛任务目的（mission）的最小表达。
- **待讨论**：继续收敛研究范围（scope）：对象、数据边界和时间边界。
- **待讨论**：定义全局硬约束：PIT、禁止未来信息、可复现性、来源权限和版本规则。
- **待讨论**：定义阶段策略：哪些阶段启用、顺序如何、哪些阶段可以跳过。
- **待讨论**：定义证据要求：每个阶段必须产出哪些表、检查记录和冻结信息。
- **待讨论**：定义最终 failure policy 的表达，但不把具体收益指标提前塞进 Contract。

### 1.2 Contract eval

- **决定**：Contract eval 只做结构检查。
- **待讨论**：检查必填项是否存在、阶段依赖是否闭合、约束是否冲突、任务是否能生成下一阶段。
- **废弃**：Contract eval 计算 outcome、label、MDE、power、target 或收益判定。

## 2. 公共表结构

### 2.1 原始来源表

- **完成**：当前 Data builder 可以读取外部 Agent 保存的 raw JSON，并保留来源原始频率。
- **待讨论**：统一最小字段：`inst`、`event_time`、`available_at`、`vintage`。
- **待讨论**：值列、属性列、单位、时区和来源版本的声明方式。
- **待讨论**：同一来源多表的命名和来源身份。
- **待讨论**：原始频率如何保留，禁止为了方便提前压成统一频率。

### 2.2 数据大表

- **完成**：当前 Data builder 使用 Polars 将两个 candle 来源和 funding 来源拼成一张面向研究任务的数据大表。
- **待讨论**：数据大表的行粒度和主键，至少要明确 `t`、`inst`。
- **待讨论**：case 自己声明的分组列、组合键和多腿列，例如 `symbol`、`long`、`short`。
- **待讨论**：列的可用时间、年龄、填充方式和来源血缘如何表达。
- **待讨论**：哪些列由 Engine 保留，Agent 不得覆盖。

### 2.3 Eval 规则与结果

- **决定**：Agent 在外部 `lab/` 组装每个阶段的 `EvalPlan`；Engine 执行 `Polars Expr`、主表已有 column 或受控 runner。
- **决定**：Engine 不把 eval column 写入阶段主表；需要的中间列只存在于 scratch frame，结果写入 eval sidecar。
- **完成**：Engine baseline 已加入通用缺失值扫描，记录 Polars null、浮点 NaN/非有限值，以及字符串 `nan`、`NAN`、`none`、`None`、`null` 形式的缺失值。
- **待讨论**：通用缺失值扫描当前以 `record` 方式进入 Data Card；哪些字段的缺失应升级为 `stop`，由阶段规则明确声明。
- **待讨论**：通用底线规则的固定名字和命名空间。
- **待讨论**：时间、身份、有限值、重复、连接、来源可用性、PIT、年龄和单位等规则。
- **待讨论**：eval 的输出类型：无业务输出、布尔汇总、指标、失败行和完整证据。
- **待讨论**：单行 flags、整张表结果和 run-level 结果如何统一保存。

### 2.4 Eval sidecar

- **决定**：Eval sidecar 至少记录 `stage`、`slot`、`rule_id`、`status`、`output_kind`、`failure_policy` 和运行详情；不强制每条 eval 都有 `row_key` 或业务结果列。
- **待讨论**：失败行、聚合指标、fixture 结果和 scratch 中间数据的独立产物格式。
- **待讨论**：定义处理记录、重跑记录和版本之间的关联。

## 3. Eval 协议

### 3.1 四个插槽

- **决定**：Slot 1 = `baseline`，由 Engine 提供通用底线规则。
- **决定**：Slot 2 = `library`，由 Agent 从 Engine eval library 选择增量规则。
- **决定**：Slot 3 = `case`，由外部 Agent 写任务特定的正确性规则，是研究假设和潜在 alpha 语义的入口。
- **决定**：Slot 4 = `case_eval`，用有效样本和反例验证 Slot 3 的规则自身。
- **决定**：Engine 必须保证四个 slot 都有声明、来源、执行记录和最终状态；slot 可以没有业务输出，但不能静默缺失。
- **决定**：四个 slot 固定按 `baseline -> library -> case -> case_eval` 执行。
- **完成**：第一版提供 `EvalPlan`、四个 slot 的顺序校验、`Expr`、column、runner 和 case fixture 的最小执行协议。
- **完成**：第一版结果写入 eval report sidecar，主表保持不变。
- **完成**：Engine 自动填充一个 `engine.baseline.general_missing_values` 底线记录；外部 baseline 规则可以追加，但不再要求 Agent 重复声明底线。
- **待讨论**：四个插槽的统一输入、输出类型、空适用集语义和失败状态。
- **待讨论**：哪些检查可以产生 treatment，哪些只能阻断或记录。

### 3.2 正确性与方法论

- **决定**：正确性 eval 回答“数据或产物是否正确”。
- **决定**：方法论回答“正确的产物是否支持研究目的”。
- **决定**：两者不能混成一个默认 eval。
- **后置**：统计显著性、null world、预测力、经济价值和收益判定。

### 3.3 Eval 生命周期

- **决定**：Eval 结果至少区分 `passed`、`failed`、`error` 和显式 `not_applicable`；不能用没有列或空值表示没有执行。
- **待讨论**：声明、执行、失败、处理、重跑和冻结的状态机。
- **待讨论**：`output_kind` 如何覆盖 `none`、`summary`、`metric`、`findings` 和完整证据。
- **待讨论**：case eval 的输入是否只能来自当前阶段产物。
- **待讨论**：eval 变更后如何使下游产物失效。
- **待讨论**：eval 结果如何进入卡片、账本和 manifest。

## 4. Data 阶段

- **完成**：Agent builder 提供 `build_main_table()`（兼容 `build()`），可在 DataLoop 中执行。
- **完成**：DataLoop 输出 `main.parquet`、`exploration.json`、四 slot `report.jsonl`、`card.md` 和 `manifest.json`。
- **决定**：Engine 在 Data 主表上运行外部 `EvalPlan`，不向 Data 主表加入 eval 列。
- **完成**：当前 case 已经实际运行 baseline、library、case、case_eval 四个插槽。
- **完成**：Contract 和 Data 都会生成带稳定 `card_id` 的 Markdown Card；当前阶段必须由 human 或 agent 显式 approve，下一阶段才会解锁。
- **完成**：每个 run 保存 `workflow.json`，记录阶段状态、Card、审批来源、时间和消息；重新生成上游 Card 会锁住下游阶段。
- **待讨论**：审批是否需要身份凭证、签名或外部 review 系统链接。
- **待讨论**：数据问题的处理方式：标记、屏蔽、删除、拆分或要求人工处理。
- **决定**：任何处理都生成新表版本，不原地修改冻结数据。
- **待讨论**：处理后哪些检查必须重新执行。
- **待讨论**：Data freeze 的内容：数据大表、eval 结果、处理记录、来源版本、manifest 和 hash。
- **废弃**：Data 阶段生成 outcome、label、收益、MDE 或 power。

## 5. Feat 阶段

- **决定**：Feat 只能读取 Data freeze 的数据大表。
- **决定**：Feat 主表只保存 key 和 feature columns；用于 IC、ICIR、累计 IC 的 target、label、horizon 和评估上下文保存为显式的 sidecar，不偷偷混入特征主表。
- **决定**：Feat 的 Engine baseline 只判断特征产物是否正确、可复现、可因果计算和可被下游读取；不因为某个特征的 IC 高低改变 correctness verdict。
- **待讨论**：Feat builder 的最小接口，是返回一张宽表，还是返回 feature table 加显式 evaluation context；需要支持多个 feature、多个 horizon 和多个 target。
- **待讨论**：PCA 的拟合窗口、标准化方式、缺失值处理和是否允许使用全样本；PCA 只能先作为结构诊断，不能直接当作 alpha 结论。
- **待讨论**：IC 统计的横截面/时间序列语义、分组列、最小样本数、年化频率和序列相关处理。
- **待讨论**：多列特征的输入输出接口。
- **待讨论**：特征列的名字、来源、lookback、可用时间和版本元数据。
- **待讨论**：Feat 必选 eval 列。
- **待讨论**：Feat 四个插槽的默认检查和 case eval。
- **待讨论**：覆盖、缺失、稳定性、重复、冗余和实时可计算性等方法论。
- **后置**：依赖 label 或 outcome 的特征效果评价，必须作为显式下游任务接入。
- **待讨论**：通过后的特征表、特征池和方法论证据如何冻结。

### 5.1 Feat Engine baseline 候选

- **待讨论**：schema、feature name、dtype、key 和时间列正确，禁止重复列和意外覆盖 Data columns。
- **待讨论**：输出 key 与 Data key 的一对一对齐，重复 key、缺行、多行和排序变化的处理。
- **待讨论**：数值列的 null、NaN、正负无穷、字符串 sentinel、全常数和近常数检查。
- **待讨论**：feature coverage、按时间和 instrument 的缺失分布、最小有效样本量。
- **待讨论**：重复运行一致性、输入顺序不变性、主表不被 builder 原地修改。
- **待讨论**：point-in-time replay：在历史 cut 上重算 feature，验证结果不能读取 cut 之后的数据。
- **待讨论**：feature provenance：输入列、lookback、available time、单位、变换和 builder hash。

### 5.2 Feat Engine eval library 候选

以下方法先作为通用诊断候选；除 baseline 正确性外，默认输出 metric/findings，不直接批准或否定 alpha。

- **待讨论**：coverage、missingness、有效样本数和按时间/instrument 的 coverage heatmap。
- **待讨论**：分布、分位数、均值、标准差、偏度、尾部、异常值和跨窗口 drift。
- **待讨论**：横截面 dispersion、tie rate、rankability、time-series autocorrelation 和 decay。
- **待讨论**：feature-feature Pearson/Spearman correlation、相关簇、冗余和有效维度。
- **待讨论**：PCA explained variance、累计 explained variance、effective rank、loadings concentration 和窗口稳定性。
- **待讨论**：横截面 Pearson IC、Spearman/rank IC、分组 IC、IC 均值、标准差、ICIR 和正 IC 比例。
- **待讨论**：累计 IC 曲线、终值、斜率、最大回撤和 block/bootstrap 区间；累计 IC 不单独作为通过条件。
- **待讨论**：quantile/bucket spread、单调性、top-bottom spread 和分位数组间稳定性。
- **待讨论**：rank autocorrelation、rank turnover、feature turnover 和潜在执行负担；经济成本留给 Algo。
- **待讨论**：不同 horizon 的 IC decay、不同时间窗口、instrument/group/regime breakdown。
- **待讨论**：permutation/null world、block bootstrap、有效独立样本数和 multiple-testing/FDR。
- **待讨论**：每个 library method 声明适用的数据形状和最小 group size；不满足时输出显式 `not_applicable`，不能用空值伪装成结果。

### 5.3 Feat metrics 的边界

- **决定候选**：PCA 评价特征结构，IC/ICIR/累计 IC 评价 feature 与显式 target 的关系；两者不能混成一个无定义的“PCA IC”。
- **待讨论**：label/target 的生成和冻结方式；Engine 只执行已声明的评估上下文，不替 Agent 猜 target。
- **后置**：PnL、交易成本、容量、仓位、滑点和执行质量归入 Algo，不在 Feat 的通用 IC 里偷偷完成。

## 6. Sig 阶段

- **后置**：定义 target、label 和 horizon 的探索接口。
- **后置**：定义训练数据、验证数据、时间隔离和防泄漏规则。
- **后置**：定义 signal 输出表和元数据。
- **后置**：定义 Sig 自己的四个插槽。
- **后置**：定义 signal 方法论、基准和样本外证据。

## 7. Algo、Seal 和 Forward

- **后置**：signal 到仓位、订单和执行结果的转换。
- **后置**：成本、滑点、容量、风险和消费者接口。
- **后置**：Algo 自己的四个插槽。
- **后置**：Seal、Forward、重新打开阶段和最终证据规则。

## 8. Freeze、血缘和审计

- **待讨论**：每阶段产物的内容 hash 和存储格式。
- **待讨论**：上游产物变化后下游如何自动失效。
- **待讨论**：来源、表、eval、处理和阶段产物的血缘关系。
- **待讨论**：追加式账本是否保留，以及它只记录哪些事实。
- **待讨论**：人工 review 看到的卡片和底层证据如何对应。

## 9. 验证顺序

- **决定**：先验证 Contract 结构，再验证公共表结构。
- **决定**：先完成 Data，再开始 Feat。
- **决定**：先完成四个插槽和 eval 协议，再添加统计方法论。
- **决定**：在表结构和阶段边界稳定前，不做大规模性能优化。
- **待讨论**：每一阶段的已知答案样本和最小端到端 case。

## 10. 后置问题

- 具体统计检验和 null world。
- 事件独立性、beta 中和和多重检验。
- 大规模内存和分批执行。
- 具体研究 case 的收益门槛。
- `research-ci-lab` 的接口和迁移。
