# 架构与不可混淆的职责

```text
                objective + compact current evidence
                              |
                    Codex CLI (optional)
                    read-only structured proposal
                              |
                 Python contract/path/budget checks
                              |
              register -> snapshot -> atomic run claim
                              |
                optional task-specific preflight
                              |
               training subprocess + persistent recorder
                              |
            separate clean evaluator + completeness checks
                              |
                SQLite facts + artifact hashes + events
                              |
           controlled comparison / explicit champion promotion
                              |
                 next finite controller iteration
```

Skills、AGENTS.md、Hooks 都不是本版的执行前提。代码可以不调用 Codex，直接执行人工提案。模型可以替换，账本、数据和评估协议仍然保留。

## 生命周期

```text
preparing -> ready -> running -> completed
     |                     |--> failed
     |                     |--> incomplete
     |                     |--> timed_out
     |                     |--> cancelled
     |                     |--> interrupted (recovery)
     +--> failed/interrupted
```

登记发生在快照之前。快照失败也保留登记记录。预算不足或已被认领的 ready 实验不会偷偷开始训练。终态不能重新运行：重试需要新 ID，并指出来源。

## 三种数据，不是一个无限聊天记录

**事实档案**：配置、源码、数据清单、环境版本、日志、曲线、评估值、错误、产物和事件。成功与失败都保存。

**压缩解释**：研究笔记记录现象、机制性判断、当前 hypothesis；带作者与来源 ID，不修改原始事实。

**模型视图**：Champion、预算、状态计数、相关压缩笔记，以及明确选中的 evidence ID 和模块源码/卡片/条件性案例。默认不扫描模块库或把完整失败档案堆叠进 prompt。

`context --evidence` 可以明确取出失败实验详情。此选取机制是上下文管理，不是操作系统层面的禁止读取。

## 快照与恢复

复制实际工作区字节而不是只写 Git commit，包含未提交文件。保留 Git 状态与 diff 统计作为补充来源，不把可能包含凭证的原始 Git patch 另行写进档案。父实验继承通过 source=parent 使用被验证过的源码。每次新实验拥有独立 source、train_work、eval_work。

Champion 只是 SQLite 中指向完整历史实验的指针；原产物保留不动。导出到新目录恢复代码/配置/产物，不覆盖现有工作区。改进失败不会导致“当前代码”成为新的最佳版本。

## 事务边界

SQLite `BEGIN IMMEDIATE` 保护运行认领、并发/预算检查和 Champion 更新。事件表、解释笔记和实验身份具有拒绝常规修改/删除的触发器。

数据库事务与文件系统不构成一个统一的分布式事务。本版通过先登记、原子 JSON 写入、可校验的快照和 recover 标记减少不一致，不声称断电/磁盘损坏时万无一失。实际部署应备份账本与产物。

## 共享模块库

`ModuleLibrary` 在赛事外保存源码版本、父子关系和条件性案例；`module_usage` 将明确选中的版本复制进实验快照，再应用适配 edits。原始参考与最终执行源码分别冻结、共同校验。父分支保留已经适配过的实现，不将原始参考覆盖回来。

使用索引只指向各赛事账本；案例是带冻结来源的人工解释，不是第二份实验事实或 Champion。离线参考库不妨碍已冻结实验的验证与继承。登记检查语法而不执行源码，可选 preflight 在赛事固定脚本中检查实际接入实现，在训练前阻断失败。详见 [模块文档](MODULE_LIBRARY.md)。

## 其他扩展位置

Kaggle News：先在独立采集层清洗来源、版本、分数、验证条件和成本，产出简短证据笔记或显式上下文，再进入提案阶段。不直接把未审查 notebook 当执行命令。

模块实现：已有模块库不代替具体模型研究。登记 JTS、代理模型或宝可梦 Transformer 时需要真实源码、许可和接口，当前未编造这些实现。

调参器：可以用优化器替代 Codex 的参数建议，只要产生相同提案即可。程序化搜索和 LLM reasoning 共用同一账本，不需要绑定特定 Agent 框架。

分布式执行：将 _stage 替换成任务提交/轮询接口，同时重新设计租约、资源预算和 worker 身份；不能把本地 PID 恢复逻辑直接搬到 Slurm。
