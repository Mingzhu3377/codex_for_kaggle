# 模块积累与复用 · v0.3

第三版保存实际实现、改动和条件性经验。研究者继续积累模块，不必把常规研究流程写成不断扩张的 Skills。明确选中模块后，Codex 阅读冻结源码，决定如何接入和验证；库不会自动匹配、下载或安装方法。

## 共享库与赛事账本

推荐将 `module-library/`、正在整理的 `module-sources/`、赛事源码 `competition-a/` 和实验账本 `competition-a-state/` 放在彼此分离的目录。其他赛事可连接同一模块库。

库保存不可修改的版本、父子关系和案例。指标、训练状态及 Champion 仍以各赛事的账本为准。使用索引保存账本路径、run ID 和绑定哈希，查询原账本，不维护另一份分数排行榜。绑定说明源码被接入；是否调用以及是否构成性能原因，仍需检查实际代码与实验。

## 登记与改进实现

```bash
python -m kaggle_harness --library ../module-library module init
python -m kaggle_harness module template --output ../module-card.json
python -m kaggle_harness --library ../module-library module add --card ../module-card.json --folder ../module-sources/my-module
python -m kaggle_harness --library ../module-library module list
python -m kaggle_harness --library ../module-library module show M-实际ID
python -m kaggle_harness --library ../module-library module verify M-实际ID
```

卡片包含方法族、版本名称、实际计算机制、输入输出、约束/不变量、插入位置、初始化、适配说明、来源定位与版本、许可、贡献者、局限和作者。未知许可或贡献者明确写未知。卡片总量限 12,000 字符，长材料保留在源码目录。卡片不能声明 `validated: true`；程序生成的登记检查只有语法与文件完整性，运行、论文忠实度和任务收益标为尚未验证。

登记按实际字节冻结并记录 SHA-256，不 import 或执行 Python、不安装依赖。复制前排除标准缓存、Git、凭证文件等，每版本源码限 8 MB。包含配套实现与许可证，数据/权重放在赛事管理范围。常见凭证检测用于防误存，不能识别所有秘密。冻结目录中的额外文件也会导致完整性校验失败；测试应在独立副本进行。

改进时编辑独立源码目录和新卡片，将旧 `M-...` 放到 `parents`，再次 `module add`。原版本保留。实际源码和卡片的差异可查：

```bash
python -m kaggle_harness --library ../module-library module diff M-父版本 M-新变体
```

## 明确接入实验

```bash
python -m kaggle_harness --store ../competition-a-state --library ../module-library module attach
python -m kaggle_harness --store ../competition-a-state proposal --output ../proposal.json
```

在完整提案中添加可选字段 `modules`：

```json
{
  "modules": [{
    "module_id": "M-0123456789abcdef",
    "mode": "copy",
    "files": [{"source": "attention.py", "target": "models/attention.py"}],
    "adaptation": "接在编码器第二阶段；维持输出形状；初始化与预算另有明确记录。"
  }]
}
```

这只是新增字段片段，完整提案仍需原有假设、配置、决策与预算。`files` 明确列出需要的依赖文件。目标须符合赛事 `editable_globs`，不能替换受保护文件或与其他映射冲突。最多 16 个模块，每个最多 64 个映射。

登记先复制模块再应用 `edits`：`runs/E-.../modules/M-...` 保存完整原始参考，`runs/E-.../source` 保存真正执行的适配实现。绑定记录原始、修改前和最终文件哈希；报告与完整性校验覆盖这些关系。

`source=parent` 的子提案省略 `modules` 时继承父绑定和已适配源码。`mode=inherit` 保留父实现，不再复制原始参考；换映射须明确用 `copy`。`modules: []` 清除本轮活跃绑定，父源码中的文件仍保留，要停止调用需修改训练代码。来源绑定不能自动证明文件已执行。

同一 store 固定连接一个库路径。参考库暂时不可用时，已冻结实验仍可校验，子实验仍能继承父实现；新的 `copy` 需要库可用。共享使用索引不可用会记事件，不抹掉有效快照。

## 让模型阅读源码

```bash
python -m kaggle_harness --store ../competition-a-state ask --module M-实际ID --objective "阅读实现，检查形状和初始化，提出有基线的适配实验" --dry-run
python -m kaggle_harness --store ../competition-a-state context --module M-实际ID
```

`ask/cycle/context` 可重复传 `--module`。只有明确选择的源码、卡片、最近至多 8 条案例，以及父实验活跃模块进入上下文。单模块案例超过 16,000 字符时保留较新的完整案例并标注省略数量，原案例仍在库中。总卡片/案例限 64,000 字符、选中源码限 16 MB，超过会要求缩小选择。模型不能新增未选中的模块。父模块未明确再选时只能沿用其适配实现，不能悄悄重置为原始参考。默认不扫描整个库。

## 记录条件性经验

编辑 `examples/module_case.json` 中的 outcome、条件、解释、失败层和作者，再执行：

```bash
python -m kaggle_harness --store ../competition-a-state --library ../module-library module case-add M-实际ID --run E-候选实验 --baseline E-同协议基线 --record ../case.json
python -m kaggle_harness --library ../module-library module cases M-实际ID
python -m kaggle_harness --library ../module-library module uses M-实际ID
python -m kaggle_harness --store ../competition-a-state module bindings E-实际ID
```

案例要求终态实验、有效冻结绑定和完整性校验。可选基线经过同协议比较。smoke 或失败运行只能写 `undecided`；`positive/negative/mixed/neutral` 是人工解释，不能自动证明原因，也不晋升 Champion。

案例冻结任务、用途、状态、配置、seed、协议、结果、绑定和比较。写清输入形状、插入位置、初始化、训练量、划分与观察。例如“这个配置下改变预训练特征尺度”比“CBAM 无效”更有用。查阅时比对原账本元数据；原账本不可用会显示 unavailable，保留冻结来源。此查询不重算原实验所有文件哈希，完整文件校验使用原 store 的 `verify/check`。

## 任务专用 preflight

策略配置 `preflight_command: ["{python}", "preflight.py"]`，检查脚本加入 `protected_files`。脚本在独立干净副本检查实际接入源码，将报告写到 `KH_PREFLIGHT_OUTPUT`：

```json
{
  "schema_version": 1,
  "scope": "此任务的形状、有限值和梯度检查；具体探针见源码。",
  "checks": [{"name": "forward_backward", "passed": true, "details": "记录实际测试形状、设备与精度。"}]
}
```

失败、缺失/无效报告、超时或修改检查/训练副本源码都会阻止训练。启用 preflight 时，训练后还核对源码未变化，否则标为 incomplete；临时产物应写入 KH_OUTPUT_DIR。通过后报告、源码快照身份和日志进入完成实验的封存产物。检查、训练与评估共用原有预留时间；检查命令进入新协议哈希。旧策略缺少该字段仍可使用，原协议哈希保持不变。

环境还提供 `KH_SOURCE_ROOT` 与 `KH_MODULE_BINDINGS_FILE`。形状、残差恒等、特征顺序、归一化轴、数值和梯度探针可按任务加入。程序不替你自动生成通用模块检查；通过的探针只支持声明范围，不等于忠实复现论文或保证涨点。可运行例子是 `examples/module_competition/preflight.py`。

## 迁移与本地文档

```bash
python -m kaggle_harness --library ../module-library module export M-变体ID --to ../module-bundle
python -m kaggle_harness --library ../another-library module init
python -m kaggle_harness --library ../another-library module import --folder ../module-bundle
python -m kaggle_harness --library ../another-library module check
```

导出包括父版本、源码、卡片和条件性案例，不执行模块；原实验数据/权重不随 bundle 复制。导入校验路径、哈希、祖先顺序、案例来源和重复身份，内容一致时可重复导入。案例包含原配置和路径，分享前检查材料范围与许可。新赛事建立自己的评估协议，旧任务正例不能替代新的收益证据。

仓库还提供显式的本地文档导入：

```bash
python scripts/import_yue_modules.py --document /path/to/modules.md --review-index /path/to/human-review.json --library ../private-library --work-dir ../new-extraction --section 3
```

`--all` 可明确选择全部章节。人工审阅索引的文档 SHA、标题、行号须完全一致。工具只提取编号章节的 Python 代码，不误读代码注释为章节，保留源字节与审阅备注，不执行、安装或发布。私有材料放 `.local/` 或仓库外，项目 MIT 许可仅覆盖自有代码。索引字段示例见 `tests/test_yue_import.py`；工具不会替代人工审阅。
