## Ralph-loop.skill（Unix管线+md文档）

用于生成“AI循环编码脚本”的skill,为用户生成一段可直接复制粘贴的bash脚本，让编码CLI以循环方式自动完成计划和构建任务

![截屏2026-04-14 00.24.47](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 00.24.47.png)

Prompt.md + AGENTS.md持久化上下文，让AI在每次迭代中读取并推进工作

YAML前置元数据

```yaml
---
name: ralph-loop
description: Generate copy-paste bash scripts for Ralph Wiggum/AI agent loops
(Codex, Claude Code, OpenCode, Goose). Use when asked for a "Ralph loop",
"Ralph Wiggum loop", or an AI loop to plan/build code via PROMPT.md + AGENTS.md,
SPECS, and IMPLEMENTATION_PLAN.md, including PLANNING vs BUILDING modes,
backpressure, sandboxing, and completion conditions.
---
```

### 译文

为 Ralph Wiggum/AI 代理循环（Codex、Claude Code、OpenCode、Goose）生成可直接复制粘贴的 bash 脚本。当被要求创建 "Ralph loop"、"Ralph Wiggum loop"，或通过 `PROMPT.md + AGENTS.md`、`SPECS` 和 `IMPLEMENTATION_PLAN.md` 进行 AI 循环规划 / 构建代码时触发，包括 PLANNING 与 BUILDING 模式、反压机制、沙箱和完成条件。

------

### 触发关键词

`Ralph loop`、`Ralph Wiggum loop`、`AI loop`，以及涉及 `PROMPT.md` / `AGENTS.md` / `IMPLEMENTATION_PLAN.md` 的循环构建请求。

### 主要执行流程

```markdown
## Overview
Generate a ready-to-run bash script that runs an AI coding CLI in a loop.
Align with the Ralph playbook flow:

1) Define requirements → JTBD → topics of concern → specs/*.md
2) PLANNING loop → create/update IMPLEMENTATION_PLAN.md (no implementation)
3) BUILDING loop → implement tasks, run tests (backpressure), update plan, commit

The loop persists context via PROMPT.md + AGENTS.md (loaded every iteration)
plus the on-disk plan/specs.
```



### 译文（结构化整理） JTBD: Jobs To Be Done

生成一个可直接运行的 bash 脚本，让 AI 编码 CLI 以循环方式运行，遵循 Ralph 规范流程：

1. **需求定义阶段**：定义需求 → JTBD（用户目标）→ 关注主题 → 生成 `specs/*.md` 规格文件
2. **规划循环阶段**：PLANNING 循环 → 创建 / 更新 `IMPLEMENTATION_PLAN.md`（仅做规划，不做代码实现）
3. **构建循环阶段**：BUILDING 循环 → 实现任务、运行测试（反压机制）、更新计划、提交代码

循环通过 `PROMPT.md + AGENTS.md`（每次迭代都会加载）以及磁盘上的计划 / 规格文件，实现上下文的持久化。



### Step1 --收集输入

```markdown
### 1) Collect inputs (ask if missing)
- Goal / JTBD (what outcome is needed)
- CLI (codex, claude-code, opencode, goose, other)
- Mode: PLANNING, BUILDING, or BOTH
- Completion condition
  - Promise phrase (string to detect), or
  - Test/command to run each iteration, or
  - Plan sentinel (e.g., a line STATUS: COMPLETE in IMPLEMENTATION_PLAN.md)
- Max iterations
- Sandbox choice (none | docker | other) + security posture
- Backpressure commands (tests/lints/build) to embed in AGENTS.md
- Auto-approve flags (ask explicitly)
  - Codex: --full-auto
  - Claude Code: --dangerously-skip-permissions
```

#### 译文（结构化整理）

生成脚本前，**若以下信息缺失则主动询问用户**：

- **目标 / JTBD**：需要达成什么结果

- **CLI 工具**：codex、claude-code、opencode、goose 或其他

- **模式**：PLANNING（规划）、BUILDING（构建），或两者同时

- **完成条件（三选一）**：

  - Promise 短语（检测到该字符串即停止）
  - 每次迭代执行的测试命令
  - 计划哨兵（如 `IMPLEMENTATION_PLAN.md` 中出现 `STATUS: COMPLETE`）

  

- **最大迭代次数**

- **沙箱选择**：none /docker/ 其他，以及安全策略

- **反压命令**：测试 /lint/ 构建命令，嵌入到 `AGENTS.md`

- **自动批准标志（必须显式询问）**：

  - Codex 用 `--full-auto`
  - Claude Code 用 `--dangerously-skip-permissions`



![截屏2026-04-14 00.54.36](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 00.54.36.png)

### Step 2 --需求拆解为规格文件

![截屏2026-04-14 01.12.45](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 01.12.45.png)

Step 3 Prompt.md 和 Agents.md

![截屏2026-04-14 01.26.53](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 01.26.53.png)

![截屏2026-04-14 01.27.32](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 01.27.32.png)

![截屏2026-04-14 01.29.52](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 01.29.52.png)

![截屏2026-04-14 01.30.13](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-14 01.30.13.png)



























