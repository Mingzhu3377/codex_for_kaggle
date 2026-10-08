# v0.2 研究与 Kaggle 工作流

设计参考 [Timothy-kira/kaggle-agent](https://github.com/Timothy-kira/kaggle-agent) 的研究树、分支冷却、
资料采集、账号和监控逻辑。本实现继续使用 Harness 的 SQLite 实验账本和冻结快照。
实验分数只有一个来源；研究解释是追加记录，远端 Notebook 是独立的操作节点。
不自动匹配、下载或调用公共 Skill。跨领域 trick 保留在现有文档中。

## 研究记录

`--store` 仍放在主命令之前。先读取树，拿到 `revision`，再追加研究或解释：

```text
python -m kaggle_harness --store ../competition-state tree
python -m kaggle_harness --store ../competition-state research-add --record examples/research_node.json --revision 1
python -m kaggle_harness --store ../competition-state annotate E-实际ID --record examples/run_annotation.json --revision 实际版本
```

revision 以实际 tree 输出为准。研究节点可以链接已有实验、研究节点或远端任务，只链接已存在节点，
禁止重复父节点，所以正常接口无法创建环。修订通过追加新节点或解释完成，原记录保留。

解释必须写方法 family、operator、采用/回退判断、理由、来源和作者。revert 必须写失败层；
可选 data、representation、optimization、objective、execution、evaluation、transfer、resource、other。
只有完成的正式实验能被解释为 keep，但 keep 不更新 Champion；晋升仍使用原来的程序门槛。
监控 tick 不改变研究 revision，避免日志观察使研究写入不断失效。

## 分支选择与回放

```text
python -m kaggle_harness --store ../competition-state branch-next
python -m kaggle_harness --store ../competition-state branch-next --record-visit
python -m kaggle_harness --store ../competition-state replay --budget-seconds 1800 --steps 12
python -m kaggle_harness --store ../competition-state ask --parent E-实际ID --objective "检查这个分支" --dry-run
python -m kaggle_harness --store ../competition-state cycle --selection balanced --steps 3 --execute --objective "在固定协议下改善表现"
```

默认循环继续从 Champion 研究，balanced 才启用新选择器。只考虑同协议、完整正式实验，
验证快照/产物完整性后，按质量、父版改进、已标注方法族稀有程度和访问冷却排序。
失败、测试、异协议、revert 或产物损坏的版本不会成为自动研究父版。
reference 可以指定比较协议；未设时用 Champion，否则用最早的完整正式实验。
所选父版同时用于配置、复制给模型的源码和结构化上下文。

回放比较 chronological、greedy、balanced，在固定历史树上逐步揭示已登记的孩子。
父节点排序不使用尚未揭示孩子的指标。预算以事前 timeout 预留量计，实际耗时另列。
解释采用当前冻结标签，未模拟标签历史修订时机，没有创造未登记分支。
这只能诊断已有历史的探索顺序，不能证明未来竞赛收益或因果优势。

## 监控与完整性

```text
python -m kaggle_harness --store ../competition-state monitor E-实际ID
python -m kaggle_harness --store ../competition-state monitor E-实际ID --watch --max-polls 20 --interval 2
python -m kaggle_harness --store ../competition-state check
python -m kaggle_harness --store ../competition-state report --output ../report-v2.html
python -m kaggle_harness --store ../competition-state audit-report --data ../report-v2.html.data.json
```

训练/评估自动挂日志观察，每约 2 秒读取有界尾部，记录进度、控制器状态和错误线索。
无日志变化只提示，不直接判断卡死；终态优先于旧错误文本。超时和清理由执行器负责。
watch 有轮数上限，状态不变时保持静默，直到出现提示或结束。

check 检查 SQLite、引用、研究图、正式结果与评估文件一致性、冻结文件和 Notebook 上传快照。
报告旁生成 data.json，审核数字、状态、覆盖和产物；它不自动证明叙述中的科学判断。
对账针对当前账本，增加实验后的旧报告需要重生成。旧 store 自动增加新表，
不重写原实验身份、策略、快照或分数。

## Kaggle CLI

```text
python -m kaggle_harness kaggle doctor
python -m kaggle_harness kaggle quota
python -m kaggle_harness kaggle competitions --search "arc prize 2026" --limit 3
python -m kaggle_harness kaggle pages arc-prize-2026-arc-agi-3
python -m kaggle_harness kaggle files arc-prize-2026-arc-agi-3
python -m kaggle_harness kaggle topics arc-prize-2026-arc-agi-3
python -m kaggle_harness kaggle kernels --competition arc-prize-2026-arc-agi-3 --limit 5
python -m kaggle_harness kaggle collect arc-prize-2026-arc-agi-3 --limit 5 --output ../arc-sources-001
python -m kaggle_harness kaggle pull owner/notebook --output ../notebook-source-001
```

复用已安装、登录的 CLI。doctor 检查实际帮助，不安装软件、不触发登录。
使用参数数组和明确超时，没有任意 shell/CLI 透传、print-access-token 或 revoke 接口。
Kaggle 2.2.x 的 JSON 可能混有分页/警告行，data 与 notices 分开保存。
部分接口忽略 page-size，客户端再次限制索引结果，标记 received_items/truncated。

collect 并行查询页面正文、文件列表、讨论索引和 Notebook 索引，保留响应及 SHA-256。
失败会标记部分覆盖，不把空白冒充成功。提供 store 时，还登记资料采集节点。
pull 只下载源码与 metadata、保存清单，不执行 Notebook。输出目录必须是新目录。
比赛数据分析、模型训练与方案判断仍需单独设计。

## 多账号

```text
python -m kaggle_harness --store ../competition-state kaggle account-add research --config-dir C:/Users/你的用户名/.kaggle-research --default
python -m kaggle_harness --store ../competition-state kaggle accounts
python -m kaggle_harness --store ../competition-state kaggle quota --account research
```

Kaggle 自己保存凭据，Harness 只登记目录与别名，目录应在比赛代码和包源码之外。
指定别名时移除进程继承的 KAGGLE_API_TOKEN/USERNAME/KEY，再设置该目录。
未设别名时保留 Kaggle 原来的环境/默认配置。默认别名仅作用于本 store。
输出对已知凭据及常见 token 脱敏，凭据不复制到实验、报告或源码。

## 私有 Notebook

```text
python -m kaggle_harness kaggle launch --folder examples/private_kernel --timeout-seconds 120
python -m kaggle_harness --store ../competition-state kaggle launch --folder ../my-private-notebook --timeout-seconds 120 --execute
python -m kaggle_harness --store ../competition-state kaggle jobs
python -m kaggle_harness --store ../competition-state kaggle poll K-实际ID
python -m kaggle_harness --store ../competition-state kaggle status owner/notebook/7
python -m kaggle_harness --store ../competition-state kaggle logs owner/notebook/7
```

launch 默认预览。实际启动需 execute、独立 store、现存源码和显式 is_private=true。
先登记身份、冻结上传源，再传递 CLI 支持的 timeout，上限 12 小时。
活动/不确定 ref 禁止重复启动。已知凭据或 KGAT token 在上传前拒绝。
成功返回的版本固定到 owner/notebook/version，轮询不会跟随最新版本。
缺版本号或传输超时标为 unknown，保留记录，禁止自动重投。

用户核实未知任务的实际版本后，可以显式恢复关联：

```text
python -m kaggle_harness --store ../competition-state kaggle reconcile K-实际ID --version 7 --reason "已在 Kaggle 核对本次启动版本"
```

恢复记录人工依据、读取精确版本；不认证远端源码逐字相同，不覆盖已有版本绑定。
远端 complete 仅表示平台任务结束，不计正式分数、不更新 Champion。
纳入正式比较仍需固定评估与产物检查。本版本不含比赛提交或 GPU 实训验证。
