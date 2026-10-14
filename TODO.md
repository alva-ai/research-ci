# Research Engine TODO

状态：Contract、DataLoop 和 FeatLoop 已实现；Feat Card 正在本地 review。Sig 的阶段边界已经确定，`SignalSet`、SigLoop、Sig Card 和 Sig 四槽仍待实现；Algo 尚未实现。

这份文件只记录当前有效的设计任务。历史讨论、失败方案和旧实现保存在仓库外部的 agent log，不在这里重复维护。

当前用 `research-ci-lab` 的 SK Hynix proxy case 验证 Contract → DataLoop → FeatLoop 的最小闭环。

不继承旧 `research-ci` 的代码、API、数据模型或兼容约束，也不设版本迁移计划。

## 状态标记

- **决定**：方向已经确定
- **待讨论**：需要继续推演
- **进行中**：正在实现
- **完成**：有验证证据
- **待实现**：协议或行为已决定，但代码和可运行证据尚未完成
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
- **决定**：Feat 阶段只读取冻结后的 Data 产物，生成可复用、point-in-time 的度量和状态表示。
- **决定**：label、target 和 horizon 在 Sig 阶段显式探索和声明；它们不属于 Feat 主表。
- **决定**：Sig 阶段把 feature 转成可供交易决策消费的预测、排序、方向或强度。规则、加权组合、矩阵方法和训练模型都是 Sig 的实现方式。
- **决定**：不增加独立的顶层 Model 阶段；模型是 Sig 的可选 backend。
- **决定**：Algo 阶段把已批准的 signal 转成资金分配、仓位、订单和执行，并处理止损、止盈、杠杆、风险、成本、滑点、容量和经济结果。
- **决定**：产生 signal score 的 feature/model/ensemble 权重属于 Sig；产生资本、资产或仓位配置的权重属于 Algo。

### 0.1 文件夹边界

- **决定**：`research-ci/src/research_ci/` 只保留协议模型、Engine 执行、Card 渲染、workflow 和 CLI；不把具体研究 case 放进 Engine。
- **决定**：Engine 的 `contract.py` 只负责 Contract 协议和结构校验；外部任务内容仍由 `lab/<case>/contract.py` 提供。
- **决定**：`eval/protocol.py`、`eval/runner.py` 和 `eval/baseline.py` 分别负责四槽协议、执行器和 Engine 底线。
- **决定**：`eval/library/` 按方法论组织通用方法；`eval/defaults/` 按 stage 提供方法提议；两者不混为同一层。
- **完成**：当前已建立 `quality`、`structure`、`association` 三个最小方法族，并为方法声明适用 stage 和输入形状。
- **完成**：方法库已按方法论扩展为 quality、structure、association、information 和 stability；包含 coverage、dispersion、PCA、VIF、互信息、IC、ICIR、累计 IC、分桶、序列稳定性和分布漂移等可复用诊断。
- **决定**：`research-ci-lab/lab/<case>/` 由外部 Agent 自由维护 Contract、raw、Data builder、阶段 eval 和探索文件。
- **决定**：`research-ci-lab/.rci/runs/<run>/` 是运行交接边界，按 `contract/`、`data/`、后续阶段目录保存 Card、主表和 sidecar 证据，并保存 `workflow.json`。
- **决定**：Card 是给人和 Agent review 的可读入口，Parquet、JSONL 和 manifest 是可追溯证据；主表和 eval sidecar 分开保存。
- **完成**：当前最小目录已按上述边界实现，CLI 不要求把外部 case 复制进 Engine。

### 0.2 四阶段语义

- **决定**：Data = 当前研究能够确认的事实和来源对齐结果。
- **决定**：Feat = 从事实生成的、可复用且不依赖某个具体交易决策的度量和状态表示。
- **决定**：Sig = 对未来或相对机会的预测、排序、方向和强度，以及形成这些输出所需的目标相关筛选与组合。
- **决定**：Algo = 根据 signal 做资金分配、风险约束和执行。
- **决定**：同一个 feature 可以服务多个 signal；同一个 signal 也可以被多个 algo 消费。阶段产物不能通过原地修改上游主表来隐式耦合。

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
- **完成**：`EvalContext` 支持显式加载标签、target、horizon 等外部 Parquet sidecar；context runner 可以按声明的 key 连接评估，不把这些列写回主表。
- **完成**：CLI 提供 `rci eval methods` 方法目录，Agent 可以按 stage 和方法族选择通用方法，不需要把库代码复制到 lab。
- **完成**：Engine 自动填充一个 `engine.baseline.general_missing_values` 底线记录；外部 baseline 规则可以追加，但不再要求 Agent 重复声明底线。
- **完成**：`DataSet` / `FeatureSet` 声明当前 Data、Feat 表的 key、列和列组；外部 Agent 通过 `LibrarySelection` 选择方法、范围、参数和理由，Engine 负责校验、展开和执行。
- **决定**：当前 `FeatureSet.targets` 和 `EvalContext` 已能连接 target sidecar，但这是低层过渡能力。最终由待实现的 `SignalSet` 声明 target、horizon、feature 输入和 signal 配方，Feat 默认方法不再承担 target 相关评价。
- **完成**：每条执行记录保存 `execution_mode` 和 `selection_reason`，Card 可以区分 Polars 表达式、列式聚合、连接分组、矩阵后端和自定义 runner。
- **待讨论**：四个插槽的统一输入、输出类型、空适用集语义和失败状态。
- **待讨论**：哪些检查可以产生 treatment，哪些只能阻断或记录。

### 3.2 正确性与方法论

- **决定**：正确性 eval 回答“数据或产物是否正确”。
- **决定**：方法论回答“正确的产物是否支持研究目的”。
- **决定**：两者不能混成一个默认 eval。
- **决定**：target 相关统计证据、null world 和预测力进入 Sig；经济价值、仓位、成本和收益判定进入 Algo。

### 3.3 Eval 生命周期

- **决定**：Eval 结果至少区分 `passed`、`failed`、`error` 和显式 `not_applicable`；不能用没有列或空值表示没有执行。
- **待讨论**：声明、执行、失败、处理、重跑和冻结的状态机。
- **待讨论**：`output_kind` 如何覆盖 `none`、`summary`、`metric`、`findings` 和完整证据。
- **待讨论**：case eval 的输入是否只能来自当前阶段产物。
- **待讨论**：eval 变更后如何使下游产物失效。
- **待讨论**：eval 结果如何进入卡片、账本和 manifest。
- **完成**：Data、Feat Card 增加 `execution_status`、`research_disposition` 和 reason codes；同时写出 `eval/assessment.json`。它只做证据分流，不把单个指标自动解释为 alpha 通过。
- **完成**：Feat Card 增加逐列 structural triage，区分 `candidate_for_review`、`incomplete`、`weak_data`、`redundant` 和 `invalid`，并列出 coverage、usable rows、VIF 和原因码。
- **决定**：有了显式 target 和样本外切分后，在 Sig 中比较候选 feature 或 signal 的 Pareto 证据（IC/ICIR、稳定性、增量信息、换手和复杂度）；不使用单一总分自动排序。

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
- **决定**：Feat 主表只保存 key 和 feature columns；label、target、horizon 和 signal 不写入 `feat/main.parquet`。
- **决定**：Feat 的职责是生成可复用的 point-in-time feature，并完成与研究目标无关的结构筛选；它不负责决定某个 feature 对某个 target 是否有预测力。
- **决定**：Feat 的 Engine baseline 只判断特征产物是否正确、可复现、可因果计算和可被下游读取；不因为某个特征的 IC 高低改变 correctness verdict。
- **完成**：最小 Feat builder 接口为 `build_features(data_table)`（兼容 `build(data_table)`）返回一张 Polars 宽表，并通过 `FEATURE_KEYS` 声明 key；后续再扩展显式 evaluation context。
- **完成**：FeatLoop 只读取当前 run 已批准的 `data/main.parquet`，输出 `feat/main.parquet`、`exploration.json`、四槽 `report.jsonl`、`card.md` 和 `manifest.json`。
- **完成**：Feat 产物进入 `review` 后停止，必须明确 approve 才解锁 Sig；Agent 可以在 review 状态修改 builder/eval 并重跑。
- **完成**：当前 SK Hynix case 生成原始价差、资金费率差、24 小时滚动 z-score 和 6 小时价差变化；已删除没有假设依据的 `funding_diff_lag8h` 示例。
- **完成**：当前 Feat 四槽已实际执行；case Agent 选择 coverage、dispersion、feature-feature correlation、VIF、PCA、autocorrelation、feature-feature MI 和 drift，没有选择 target 相关方法。
- **完成**：第一版 PCA 使用声明的列、完整行和可选标准化，输出 explained variance、累计解释率和 effective rank；当前仍只作为结构诊断，不能直接当作 alpha 结论。
- **完成**：VIF、PCA 和 dispersion 接收显式多列输入，结果留在 sidecar metric/findings 中。
- **待讨论**：特征列的名字、来源、lookback、可用时间和版本元数据。
- **待讨论**：Feat 必选 eval 列。
- **待讨论**：Feat 四个插槽的默认检查和 case eval。
- **完成**：第一版通用库已加入 coverage、missing value floor、dispersion、pairwise correlation、VIF、PCA、autocorrelation、mutual information、split association、rank autocorrelation 和 distribution shift；Feat 只选择其中与结构、稳定性和正确性相关的方法。
- **完成**：baseline 和大部分 quality/structure/stability 方法先走 Polars 列式路径；PCA、VIF、MI 在 Polars 完成矩阵筛选后使用 scikit-learn/NumPy。
- **决定**：硬错误、不可计算列和完全重复列可以从候选池排除，但必须在 selection/manifest 中记录原因；高相关、VIF 高或信息重叠等软冗余先分簇或标记，不由 Engine 自动删除。
- **待实现**：`eval/defaults/feat.py` 收敛为 target-free 提议；IC、ICIR、累计 IC、bucket spread 和 feature-target MI 的默认选择迁移到 Sig。通用 library 实现继续按方法论共享。
- **待讨论**：通过后的特征表、结构筛选记录、特征池和方法论证据如何冻结。

### 5.1 Feat Engine baseline 候选

- **完成**：最小输出结构检查要求至少一个 key 和一个 feature column；无法识别 key 时停止。
- **完成**：输出 key 与 Data key 必须一对一对齐；重复 key、缺行、多行和新增 key 会停止。
- **完成**：Feat builder 执行前后比较 DataFrame，禁止修改输入 Data 主表。
- **待讨论**：feature name、dtype、禁止意外覆盖 Data columns，以及 key 排序变化是否应阻断。
- **待讨论**：数值列的 null、NaN、正负无穷、字符串 sentinel、全常数和近常数检查。
- **待讨论**：feature coverage、按时间和 instrument 的缺失分布、最小有效样本量。
- **待讨论**：重复运行一致性和输入顺序不变性。
- **待讨论**：point-in-time replay：在历史 cut 上重算 feature，验证结果不能读取 cut 之后的数据。
- **待讨论**：feature provenance：输入列、lookback、available time、单位、变换和 builder hash。

### 5.2 Feat Engine eval library 候选

以下方法用于 feature 本身的正确性、可用性、结构和稳定性。除 baseline 正确性外，默认输出 metric/findings，不直接批准或否定 alpha。

- **完成**：第一版提供 required columns、finite values、coverage、dispersion、pairwise correlation、VIF、PCA、autocorrelation、feature-feature MI 和 distribution shift。
- **待讨论**：完整的 coverage、missingness、有效样本数和按时间/instrument 的 coverage heatmap。
- **待讨论**：分布、分位数、均值、标准差、偏度、尾部、异常值和跨窗口 drift。
- **待讨论**：横截面 dispersion、tie rate、rankability、time-series autocorrelation 和 decay。
- **待讨论**：feature-feature Pearson/Spearman correlation、相关簇、冗余和有效维度。
- **待讨论**：PCA explained variance、累计 explained variance、effective rank、loadings concentration 和窗口稳定性。
- **待讨论**：rank autocorrelation、rank turnover、feature turnover 和潜在执行负担；经济成本留给 Algo。
- **待讨论**：每个 library method 声明适用的数据形状和最小 group size；不满足时输出显式 `not_applicable`，不能用空值伪装成结果。

### 5.3 Feat metrics 的边界

- **决定**：PCA、VIF、feature-feature correlation 和 feature-feature MI 评价候选特征集合的结构，属于 Feat。
- **决定**：IC、rank IC、ICIR、累计 IC、target MI、bucket spread、horizon decay 和 target-aware ablation 需要显式 target，属于 Sig。
- **决定**：Feat 不修改或覆盖 Data、target、signal；所有结构筛选保留声明、理由和证据。
- **决定**：PnL、交易成本、容量、仓位、滑点和执行质量属于 Algo。

## 6. Sig 阶段

- **决定**：Sig 读取已批准的 `feat/main.parquet`，并把 feature 转成预测、排序、方向或强度；最终 signal 必须有明确语义，不能只是一组未解释的模型输出。
- **决定**：Sig 可以是规则、feature 加权组合、矩阵方法、训练模型或它们的 ensemble。是否训练模型由 case 决定，不新增顶层 Model stage。
- **决定**：Sig 同时负责两类 target-aware 评价：先评价原始 feature 与 target 的关系，用于本 signal recipe 的筛选；再评价最终 signal 与 target 的关系。
- **决定**：targetless 或纯规则 signal 也是合法路径；IC 体系是可选方法论，不是 Sig 的强制通过条件。
- **决定**：Sig 不修改 `feat/main.parquet`。最终 feature 选择、排除、分簇和组合记录保存在 Sig sidecar/manifest 中。
- **决定**：同一 feature pool 可以产生多个 signal recipe；每个 recipe 独立声明 target/horizon 或 targetless hypothesis、方向、归一化、权重和可选 estimator。

### 6.1 Sig Agent loop

以下是 Agent 在一个 Sig 阶段内的研究循环，不拆成额外顶层 stage：

1. 声明 target/horizon，或声明 targetless hypothesis。
2. 依据 Feat 的结构证据和 Sig 的 target-aware 证据选择候选 feature。
3. 用规则、加权、矩阵方法、模型或 ensemble 构造 signal。
4. 对 signal 做方向统一、归一化和必要的校准。
5. 任何 fitted transform、PCA、权重或模型都使用时间隔离的 train/validation/test 或 walk-forward；预测时只使用当时已经 fit 的状态。
6. 运行 Sig 四槽，生成 Sig Card，冻结 recipe、selection、artifact 和 signal table，等待明确批准。

### 6.2 Sig 产物

- **待实现**：`SignalSet`，显式声明 keys、features、targets/horizons 或 targetless hypothesis、signal columns、recipe 和 fit policy。
- **待实现**：SigLoop 和 CLI，只读取已批准的 Feat freeze，执行外部 signal builder 与四槽计划，生成 review 状态并锁住 Algo。
- **待实现**：`sig/main.parquet`，只包含 key 和下游需要的 signal columns。
- **待实现**：`sig/selection.json`，记录每个候选 feature 的保留、排除、分簇、方向和理由。
- **待实现**：`sig/recipe.json`，记录规则、组合权重、归一化、estimator、超参数和训练窗口。
- **待实现**：`sig/artifacts/`，保存可选的系数、PCA、模型或校准器，并记录 hash 和训练范围。
- **待实现**：`sig/eval/report.jsonl`、`sig/eval/assessment.json`、`sig/card.md` 和 `sig/manifest.json`。
- **决定**：当前 `FeatureSet.targets`/`EvalContext` 可以作为实现 Sig 的底层连接能力，但公开阶段协议最终由 `SignalSet` 承担。

### 6.3 Sig 四个插槽

- **决定**：`baseline` 检查 key 对齐、signal schema、coverage、null/NaN/非有限值、PIT、fit/predict 时间隔离、确定性和 artifact 可复现性。
- **决定**：`library` 由外部 Agent 按本 case 选择 IC、rank IC、ICIR、累计 IC、target MI、bucket/monotonicity、horizon decay、分时段/分 regime 稳定性、OOS、null/permutation、turnover 和 concentration 等方法；Engine 不自动全选。
- **决定**：`case` 声明并执行本任务的 signal recipe，包括输入 feature、target/horizon 或 hypothesis、方向、权重、归一化和可选 fit 配置。
- **决定**：`case_eval` 验证 Slot 3：有效 fixture 应通过，错误权重或维度应失败，打乱 label 应破坏 target evidence，修改未来数据不能改变历史 signal，反例不能被规则误判。

### 6.4 Sig 方法论与判定

- **待讨论**：IC 的横截面/时间序列语义、分组列、最小样本数、年化频率和序列相关处理。
- **待讨论**：横截面 Pearson IC、rank IC、分组 IC、IC 均值、标准差、ICIR 和正 IC 比例。
- **待讨论**：累计 IC 的终值、斜率、最大回撤和 block/bootstrap 区间；累计 IC 不单独作为通过条件。
- **待讨论**：quantile/bucket spread、单调性、top-bottom spread、horizon decay 和分 regime 稳定性。
- **待讨论**：permutation/null world、block bootstrap、有效独立样本数、multiple-testing/FDR、selection bias 和 nested walk-forward。
- **决定**：Card 根据四槽 evidence 说明候选是 invalid、incomplete、redundant、unstable、target-weak、overfit-risk 或 candidate_for_review；Engine 不用单一分数自动选“最佳 signal”。
- **决定**：Sig 的批准表示 signal 定义和证据足以交给 Algo 研究，不表示策略已经盈利或可交易。

## 7. Algo、Seal 和 Forward

- **决定**：Algo 只读取已批准的 signal 产物，把 signal value 转成 entry/exit、rebalance、position sizing、资产/资本权重和订单。
- **决定**：止损、止盈、杠杆、gross/net exposure、风险预算、组合约束、成本、滑点、容量、执行和 PnL 属于 Algo。
- **决定**：Sig 的组合权重用于形成预测分数；Algo 的组合权重用于分配资本。两者分别记录，不能共用一个含义不清的 `weight`。
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
- **决定**：Feat 通过结构筛选并批准后，再在 Sig 中声明 target/horizon、做目标相关筛选并构造 signal。
- **决定**：Sig 批准后，Algo 才能研究仓位、风险、执行、成本和收益。
- **决定**：先完成四个插槽和 eval 协议，再添加统计方法论。
- **决定**：在表结构和阶段边界稳定前，不做大规模性能优化。
- **待讨论**：每一阶段的已知答案样本和最小端到端 case。

## 10. 后置问题

- 事件独立性、beta 中和和多重检验。
- 大规模内存和分批执行。
- 具体研究 case 的收益门槛。
- `research-ci-lab` 的接口和迁移。
