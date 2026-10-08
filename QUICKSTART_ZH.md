# 从这里开始

这份交付是 **Kaggle Research Harness v0.1.0**：源码、可运行示例、自动化测试和中文说明都在压缩包中。不是一份仅靠 Codex 遵守的提示词。

## 第一步：不连接 Codex，先运行本地示例

解压，进入 `kaggle-research-harness` 目录，然后执行：

```bash
python -m kaggle_harness demo --output ../first-harness-run
```

需要 Python 3.10+，这一步只用标准库和 CPU，不联网，不需要 API Key。输出目录必须尚不存在。

打开 `../first-harness-run/report.html` 查看实验结果。程序会跑一个真实的小型回归训练，并测试崩溃、提前停止和超时。示例分数不是 Kaggle 比赛成绩。

## 第二步：验证自己机器上的行为

```bash
python -m unittest discover -s tests -v
```

交付时的实际验证结果在 `validation/test-results.json` 与 `validation/test-output.txt`。Linux 进程恢复测试会在不适用的平台跳过，不表示那些平台已验证。

## 第三步：检查 Codex 接口

本机装好并登录 Codex CLI 后：

```bash
python -m kaggle_harness doctor
python -m kaggle_harness --store ../first-harness-run/state ask --objective "根据当前证据提出下一项有价值的受控实验" --dry-run
```

第二条只是生成提示和上下文，不调用模型。去掉 `--dry-run` 才真实请求 Codex，但仍不训练；查看返回的 proposal_path，再交给 `run --proposal`。

```bash
python -m kaggle_harness --store ../first-harness-run/state ask --objective "根据当前证据提出下一项有价值的受控实验"
```

当前交付验证了适配器的离线替身进程流程，未在交付环境进行真实 Codex 认证调用。`doctor` 会检查你本机所装版本是否支持所需参数。

## 换成自己的比赛

读 `README.md` 第 2–5 节：让训练脚本使用 Recorder，再配置受保护的 evaluate.py、数据路径、指标和预算。完整可运行的脚本可参考 `examples/toy_competition/`，不是要求你从零猜接入方式。

完整资料：`docs/PROTOCOL.md` 说明接口；`docs/ARCHITECTURE.md` 说明控制流程；`docs/SECURITY_AND_LIMITS.md` 说明本地可靠性控制与真正权限隔离的区别。

这版先完成实验执行核心，尚未自动采集 Kaggle News、自动提交比赛或实现你自己的 JTS/代理 Transformer 模块。
