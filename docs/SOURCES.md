# Codex 接口依据

查询日期：2026-10-05。只依据 OpenAI 官方文档接入 CLI；真实支持情况还通过本机 `codex exec --help` 检查。

1. OpenAI, Codex non-interactive mode.
   https://developers.openai.com/codex/noninteractive
   本次页面重定向至 https://learn.chatgpt.com/docs/non-interactive-mode
   用于核对 `codex exec`、`--json`、`--output-schema`、`--output-last-message`、只读沙箱、`--ephemeral` 和 `--ignore-user-config`。

2. OpenAI, Codex CLI reference / Developer commands.
   https://developers.openai.com/codex/cli/reference
   本次页面重定向至 https://learn.chatgpt.com/docs/developer-commands?surface=cli
   用于核对 `--cd`、`--skip-git-repo-check`、`-c/--config`、标准输入提示 `-` 和可选模型参数。

接口测试分两类：离线替身进程验证参数构造、事件保存和输出校验；真实 Codex 调用需要用户机器已有安装、权限与认证。本次交付未把前者冒充为后者。
