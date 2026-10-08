# 接入契约

## 提案

提案是严格 JSON 对象，不允许悄悄添加 `skip_validation` 等未知字段。

```json
{
  "schema_version": 1,
  "parent_id": null,
  "source": "workspace",
  "purpose": "experiment",
  "hypothesis": "当前收敛不足来自过小学习率，而不是模型容量不足。",
  "expected_observation": "相同步数下训练损失下降更快，固定验证集改善；否则不能支持这个解释。",
  "config": {
    "optimizer": "sgd",
    "learning_rate": 0.06,
    "steps": 40,
    "seed": 17,
    "failure_mode": "none"
  },
  "decisions": [
    {"key":"optimizer","origin":"inherited_default","reason":"保持父实验的优化器。","evidence_ids":[]},
    {"key":"learning_rate","origin":"deliberate","reason":"检验收敛不足假设。","evidence_ids":[]},
    {"key":"steps","origin":"inherited_default","reason":"固定训练量。","evidence_ids":[]},
    {"key":"seed","origin":"inherited_default","reason":"固定比较条件。","evidence_ids":[]}
  ],
  "edits": [],
  "timeout_seconds": 30
}
```

`source=parent` 从父实验的完整源码快照继承；需要有效 parent_id。
`source=workspace` 则明确从当前工作区复制，即使记录了 parent_id 也不会假装源码相同：实际文件差异会写入 diff.json。

`edits` 中每项是 `{"path":"train.py","content":"完整新文件内容"}`；只接受可移植相对路径与策略允许的文件。第一版不提供删除文件指令，新增文件可通过允许的路径实现。

`decisions` 至少覆盖 policy.decision_keys。`validated` 必须引用已完成的正式实验；这只校验来源存在，不自动证明因果关系。

## 执行环境变量

| 变量 | 含义 |
|---|---|
| KH_RUN_ID | 当前实验 ID |
| KH_CONFIG | 完整注册配置的 JSON 路径 |
| KH_OUTPUT_DIR | 训练产物唯一输出目录 |
| KH_EVAL_OUTPUT | 评估器输出 JSON 路径 |
| KH_DATASETS_JSON | 数据名称到原始路径的 JSON 映射 |
| KH_PROTOCOL_HASH | 固定评估/数据/seed 协议指纹 |
| KH_FOLD_IDS | 预期 fold ID 列表 |
| KH_PURPOSE | smoke_test 或 experiment |

训练脚本不通过数据库写事实；Recorder 只写本实验产物。控制器负责将检查后的结果记入账本。

## 训练结果

`Recorder` 生成：

- `resolved_config.json`：实际解析后的完整配置。本版必须与注册配置相等；实际不相等则标为 incomplete，而非偷偷接受。
- `curves.csv`：step、epoch、train_loss、validation_metric、learning_rate、时间和额外字段。
- `training_summary.json`：schema_version、finished、completed_steps、curve_rows、stop_reason、details、时间。
- 异常时另写 `training_error.json`。异常退出不会自动产生成功完成记录。

`budget_complete` 必须匹配注册步数；正式实验必须达到最低步数。早停默认关闭；打开后仍须达到最低正式步数且写明终止说明。程序只能检查契约，无法从一句解释证明早停标准确实科学；需要把真正早停规则写进被审查的训练/评估协议。

## 评估结果

```json
{
  "schema_version": 1,
  "metric": "mse",
  "value": 0.012,
  "folds": [{"id":"0","value":0.010},{"id":"1","value":0.014}],
  "seed": 17,
  "protocol_hash": "使用环境中提供的 KH_PROTOCOL_HASH"
}
```

总体 value 由受保护评估器计算；程序不擅自用 fold 均值替代，因为不同 metric/样本数未必允许简单平均。每个 fold 的 ID 必须唯一且与策略完全一致；所有指标必须为有限数。

示例中的两个 fold 是固定留出集的两个分组，不冒充“两次独立训练的交叉验证”。真实 K-fold 或多 seed 协议应由你自己的训练/evaluate 程序实现，并将额外文件列为受保护文件。

## 指标比较的含义

只比较 completed 的正式实验，并要求 protocol_hash 一致。本版把 seed 也纳入协议；跨 seed 的统计汇总需要独立实现，不能通过换 seed 来晋升。

晋升条件是**严格超过** min_improvement，平分不能覆盖 Champion。没有内置统计显著性或多重比较校正；多次试验仍可能对固定验证集过拟合。程序记录 `statistical_significance: not_assessed`，不会把阈值达标包装成显著结果。

一个 SGD 配置失败不等于 SGD 不适合整个任务。优化器切换往往需要分别调整学习率等适配参数；账本保存事实，研究结论仍要限制在证据能支持的范围。
