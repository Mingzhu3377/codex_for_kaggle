# Kaggle Research Harness · v0.1.0

**让 Codex 提出研究决策，让程序保存实验事实。**

这是一个可运行的本地 Python 实验控制器，不依赖 Skills 是否触发。第一版围绕四件事建立约束：**每次运行都有身份和记录；两轮测试不能冒充完整实验；最佳版本不被下一次改动覆盖；完整失败档案不默认塞进模型上下文。**

运行依赖：Python 3.10+，核心和测试仅用标准库。Codex CLI 是可选接入：不安装它也能运行实验、故障测试、报告和示例。本次实际验证环境与结果见 `validation/`；真实 Codex 调用及原生 Windows/GPU 路径不计入已验证范围。

## 1. 先跑起来

解压后，进入本文件所在目录：

```bash
python -m kaggle_harness --help
python -m unittest discover -s tests -v
python -m kaggle_harness demo --output ../my-harness-demo
```

`../my-harness-demo` 必须是**尚不存在**的目录，重复运行请换名字。程序不会为了演示而删除你的历史结果。整个示例在本机 CPU 上进行，不调用模型、不联网、不提交 Kaggle。

这个示例不是伪造分数：它实际执行 **清洗 CSV → 用训练集统计量标准化 → SGD 训练 → 固定验证集评估 → 保存产物**，然后注入崩溃、两轮提前结束和超时。数据是随项目附带的合成回归数据，不是 Kaggle 比赛数据。

执行后会得到：

```text
my-harness-demo/
├── workspace/          # 可修改的比赛代码
├── state/              # 与代码分离的实验账本和完整档案
├── context.json        # 给模型的压缩视图，不含所有失败配置/日志
├── demo_results.json   # 真实执行结果
└── report.html         # 本地可打开的实验报告
```

可选安装命令行入口；不安装也能一直用 `python -m kaggle_harness`：

```bash
python -m pip install -e .
kh --help
```

## 2. 接入自己的比赛仓库

**不要把控制器、账本和可修改的比赛代码混在一个目录。** 推荐：

```text
work/
├── kaggle-research-harness/   # 这份项目
├── competition/              # train.py / evaluate.py / config.json / 自己的模型代码
└── competition-state/        # init 时创建，不事先创建
```

先生成并编辑策略文件：

```bash
python -m kaggle_harness policy-template --output ../competition-policy.json
```

重点修改 `competition_id`、评估指标及方向、训练/评估命令、数据路径、受保护的评估文件、最低正式训练步数和时间预算。完整示例是 `examples/toy_policy.json`。命令必须是参数数组，而不是让模型随意拼接的 shell 字符串。

```bash
python -m kaggle_harness --store ../competition-state init --workspace ../competition --policy ../competition-policy.json
```

`policy.reference.json` 是供查阅的副本。**直接修改它不会更改生效策略**；生效策略存于账本初始化元数据中。本版把数据版本和评估协议固定在一个 store 内。需要更换验证划分/评估器/数据时，请建立新的 store，不要强行横比不同协议的分数。

大型数据在 `policy.datasets` 中注册；这些文件不复制进源码快照，而是逐次计算 SHA-256。保留原始数据版本是你的职责；本版不代替 DVC/对象存储。全量校验大数据会有 I/O 成本。

## 3. 训练脚本只需要接一层 Recorder

```python
import json
import os
from pathlib import Path
from kaggle_harness.recorder import Recorder

config = json.loads(Path(os.environ["KH_CONFIG"]).read_text())
output = Path(os.environ["KH_OUTPUT_DIR"])
datasets = json.loads(os.environ["KH_DATASETS_JSON"])

# 先完成参数解析，再记录实际配置；不要把未使用的原始参数冒充实际配置。
resolved_config = dict(config)

with Recorder.from_env(resolved_config) as recorder:
    for step in range(1, resolved_config["steps"] + 1):
        # 在这里调用自己的数据清洗、模型、优化器、训练代码。
        # loss = ...
        # actual_lr = optimizer.param_groups[0]["lr"]
        recorder.log(step=step, train_loss=loss, learning_rate=actual_lr)

    # 将 checkpoint/model/predictions 写入 output 中。
    # 在 policy.required_artifacts 中声明必须存在的文件。
    recorder.finish(stop_reason="budget_complete")
```

上面是**接入片段**，`loss`、`actual_lr`、模型保存需要接到你的训练代码；可直接运行的完整版本在 `examples/toy_competition/train.py`。

`step` 是你提前约定的训练计量单位：可以表示 optimizer step，也可以表示 epoch。必须在项目中保持同一含义。`min_formal_steps` 与配置中的预算使用相同单位。示例把一个全量梯度更新计为一步。

Recorder 每次写曲线会 flush + fsync，尽量保留中断前已经记录的数据。大规模训练无需每个 minibatch 都记录，但必须明确日志间隔；最后一个记录步必须与完成记录一致。它无法恢复模型代码根本没有上报的信息。

评估器从 `KH_OUTPUT_DIR` 读取产物，在独立的干净代码副本里运行，向 `KH_EVAL_OUTPUT` 写结构化结果。完整契约见 `docs/PROTOCOL.md` 和 `examples/toy_competition/evaluate.py`。

## 4. 日常实验操作

以下命令都从 harness 项目目录执行，`--store` 放在子命令之前。

生成提案，但不执行：

```bash
python -m kaggle_harness --store ../competition-state proposal --output ../proposal-001.json
```

编辑其中的 hypothesis、预期现象、参数来源、预算，然后：

```bash
python -m kaggle_harness --store ../competition-state run --proposal ../proposal-001.json
python -m kaggle_harness --store ../competition-state list
```

也可以分开登记和执行：

```bash
python -m kaggle_harness --store ../competition-state submit --proposal ../proposal-001.json
python -m kaggle_harness --store ../competition-state run --id E-实际返回的ID
```

`submit` 会先写账本，再冻结代码和配置。`run` 只执行 `ready` 状态的记录；同一个 ID 不能重复运行。重试必须是新实验，并保留 `parent_id`。

查看、比较、晋升和恢复：

```bash
python -m kaggle_harness --store ../competition-state show E-实际ID
python -m kaggle_harness --store ../competition-state logs E-实际ID --stderr --tail 40
python -m kaggle_harness --store ../competition-state compare E-父实验ID E-候选ID
python -m kaggle_harness --store ../competition-state promote E-候选ID
python -m kaggle_harness --store ../competition-state promote E-候选ID --yes --reason "同协议改进超过预先设定阈值"
python -m kaggle_harness --store ../competition-state champion
python -m kaggle_harness --store ../competition-state verify E-实际ID
python -m kaggle_harness --store ../competition-state export E-最佳ID --to ../restored-best
```

不带 `--yes` 的晋升只做预览。当前 Champion 存在 SQLite 单一指针中，没有第二份容易失步的 champion.yaml。晋升前会检查状态、实验用途、数据/验证/seed 协议、文件完整性和改进阈值。

**`export` 是恢复已保存的代码、配置和产物到新目录，不是保证在任何硬件上逐位复现训练，也不是自动续训。** 外部数据只附带清单，不复制数据本身。

## 5. Codex 接入

此适配器使用官方非交互 CLI 的 `codex exec`、JSON 事件流、输出 Schema 和最终响应文件；不是靠 Skill 强制模型执行流程。官方文档来源见 `docs/SOURCES.md`。

先在自己的机器安装并登录 Codex CLI，再检查能力：

```bash
python -m kaggle_harness doctor
```

本版会检测实际 CLI 是否支持需要的参数，不锁死模型名称。不支持时会明确报错，不会悄悄退回无沙箱运行。适配器使用 `--sandbox read-only`，关闭交互审批并忽略用户 config；模型只提出方案和代码内容，由控制器按路径白名单应用。自定义 provider/profile 依赖用户配置的安装方式，需要你相应扩展适配器，不能直接假定兼容。

先查看准备提供给模型的上下文和提示，不消费模型调用：

```bash
python -m kaggle_harness --store ../competition-state ask --objective "判断下一步应该研究学习率、优化器还是训练预算" --dry-run
```

真实请求一项提案，仍然**不启动训练**：

```bash
python -m kaggle_harness --store ../competition-state ask --objective "基于当前证据提出一个受控实验，不要无意义调参"
```

返回结果包含 `proposal_path`。检查提案后，把该路径传给 `run --proposal`。每次调用的 prompt、原始事件、stderr、原始响应和校验结果都保存在 `agent_sessions/`，错误响应也不丢弃。

明确授权有限轮数后，可以运行程序控制的循环：

```bash
python -m kaggle_harness --store ../competition-state cycle --objective "在固定验证协议下改善表现，每次记录明确假设" --steps 3 --execute
```

加 `--promote` 才启用通过程序门槛的自动晋升。不加 `--execute` 只产生第一项提案。实验失败时循环停止，不自动连环重试；时间预算不足也不会开始新训练。**Codex 调用预算与 GPU/实验预算是两回事：本版限制调用轮数和单次调用超时，尚未实现 token 或货币总预算。**

模型可以通过 `edits` 返回受允许源码文件的完整内容，但不能用该接口改评估器/账本。代码修改首先作用于新实验快照，不自动回写你原始比赛工作区。后续从 Champion 建分支时使用 `source=parent`，因此会继承被验证过的代码，而不是一个可能已经改坏的工作区。

## 6. 经验管理：事实、解释和上下文分开

完整成功/失败/中断轨迹都在 `state/runs/`。默认上下文只包含 Champion 配置与指标、预算、状态数量、至多 12 条压缩研究笔记。不会主动注入失败实验的完整配置或曲线。

```bash
python -m kaggle_harness --store ../competition-state note --run E-实际ID --text "现象：相同预算下收敛较慢；尚不能归因为优化器本身，需要调整其适配学习率再比较。" --tag optimizer
python -m kaggle_harness --store ../competition-state context --tag optimizer
python -m kaggle_harness --store ../competition-state context --evidence E-确实相关的失败ID
```

`--evidence` 是显式调取，不是默认全库 RAG。研究笔记是解释，不是自动确认的因果结论。完整日志、事实记录和人工/模型解释分别保存。`validated` 参数来源必须引用完成的正式实验，但引用是否足以支持科学主张仍需要研究判断。

## 7. 中断与预算

```bash
python -m kaggle_harness --store ../competition-state recover --stale-seconds 30
```

恢复操作检查主机、心跳和进程存活状态，不会因任务跑得久而抢占仍在运行的控制器。Linux 额外检查进程启动 token，避免盲目杀掉复用 PID 的进程。强制结束控制器时，已登记的子进程会被清理、实验标为 `interrupted`，不会被自动冒充为成功。对无法验证身份的残留进程，返回人工清理提示。

这里恢复的是**实验状态**，不是神经网络训练状态。续训需要你将 checkpoint 显式接入新的子实验。控制器意外消失时会保守计费，至少计入整个预留时间，防止预算凭空返还。

SQLite 的事务负责运行认领、并发限制、预算预留和 Champion 更新。训练/evaluation 子进程有墙钟超时；快照、数据校验、磁盘写入和清理仍可能增加 I/O 开销，不是操作系统实时资源配额。GPU 时间字段是“分配 GPU 数 × 运行墙钟时间”的估计，不是实测利用率。

## 8. 当前交付边界

已经实现：实验登记与状态机、代码/参数/数据来源记录、全流程运行、曲线 Recorder、固定评估协议、失败保留、最佳版本晋升、恢复导出、按需证据、Codex 适配器和有界循环。

本版**没有**自动抓 Kaggle Code/Discussion、自动提交比赛、多机/Slurm 调度、真实 JTS 实现、代理 Transformer 实现或统计显著性判定。它们是后续可接入的研究层，不伪装成已经完成的功能。

**这是单机、可信研究代码场景的可靠性控制器，不是恶意代码安全沙箱。** 目录分离、哈希、SQLite 触发器和参数校验能防很多误操作，但不能阻止同一系统用户的任意代码直接访问文件或绕过程序。需要抗恶意代码、硬性冷档案读隔离、验证数据防泄漏或网络隔离时，必须再用不同系统身份/容器/虚拟机落实权限；详细边界见 `docs/SECURITY_AND_LIMITS.md`。

## 项目结构

```text
kaggle_harness/
  cli.py          # 命令行入口
  engine.py       # 登记、快照、状态机、执行、比较、晋升、恢复
  store.py        # SQLite 账本、事件、笔记、事务
  contracts.py    # 严格提案与策略校验
  recorder.py     # 训练端曲线与配置记录
  codex.py        # CLI 适配器、只读提案、有界循环
  util.py         # 原子写入、哈希、路径与进程工具
  report.py       # 本地 HTML 报告
  demo.py         # 可运行的端到端验证
examples/         # 完整 CPU 训练/评估和策略示例
tests/           # 标准库 unittest 测试（包含实际中断、并发与故障）
docs/             # 协议、架构、安全边界、官方接口来源
validation/       # 本次交付的真实验证记录
```
