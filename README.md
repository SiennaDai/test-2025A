# 2025 CUMCM A：烟幕干扰弹的投放策略

当前阶段：问题 1–5 均已完成求解与验证，中文论文终稿已补全并通过数值一致性复核。

## 文件入口

- `problem.md`：经过页面核对的结构化题面；
- `model-spec.md`：当前模型规格；
- `analysis-log.md`：逐步分析、决定和待确认事项；
- `validation-q1.md`：问题 1 的人工可读结果与验证报告；
- `validation-q2.md`：问题 2 的策略、求解状态与验证报告；
- `validation-q3.md`：问题 3 的联合覆盖策略、数值证据与局限；
- `validation-q4.md`：问题 4 的三机接力策略与验证；
- `validation-q5.md`：问题 5 的分层多导弹策略、验证与目标口径限制；
- `paper-draft.md`：中文数学建模论文 Markdown 终稿；
- `latex/paper.tex`：符合 2025 版格式规范的匿名电子版 LaTeX 论文；
- `latex/AI工具使用详情.tex`：按 2025 试行规定准备的 AI 使用披露材料；
- `data/raw/`：官方 PDF 和后续附件，只读；
- `data/processed/problem-extracted.md`：PDF 自动提取记录；
- `configs/`：正式参数与场景配置；
- `src/`：本题专用实现；
- `outputs/`：图、表、日志和运行清单。

Windows 目录：

```text
\\wsl.localhost\Ubuntu\home\<Linux用户名>\projects\modeling\competitions\2025A
```

运行入口：

```bash
uv run python competitions/2025A/run.py
```

重新提取题面：

```bash
uv run python scripts/extract_pdf.py \
  competitions/2025A/data/raw/CUMCM-2025-problem+A-Chinese.pdf \
  --output competitions/2025A/data/processed/problem-extracted.md \
  --render-dir competitions/2025A/data/processed/problem-pages \
  --overwrite
```

所有生成结果写入 `outputs/`，不得覆盖 `data/raw/`。正式建模按 `AGENTS.md` 的检查点逐步推进。
