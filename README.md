# CUMCM 2025 A：烟幕干扰弹的投放策略

这是 2025 年全国大学生数学建模竞赛 A 题的完整复盘项目，包含题面提取、五个问题的模型与代码、优化记录、数值验证、结果工作簿以及可直接编译的中文论文。

项目采用严格的整圆柱遮蔽判据：只有来袭导弹到真目标的全部视线都穿过至少一朵有效烟幕，才计为有效遮蔽。多烟幕按 `∀ ray, ∃ cloud` 联合覆盖，遮蔽时长按连续时间区间并集计算，重叠部分不重复计时。

## 先看什么

如果只想阅读结论，建议按以下顺序：

1. [paper.pdf](latex/paper.pdf)：完整排版论文，含摘要、正文、参考文献和全部源代码附录；
2. [paper-draft.md](paper-draft.md)：便于搜索和修改的 Markdown 论文；
3. [validation-q1.md](validation-q1.md) 至 [validation-q5.md](validation-q5.md)：每问的策略、数值证据和结论边界；
4. [problem.md](problem.md)：经过原 PDF 页面核对的结构化题面；
5. [model-spec.md](model-spec.md)：运动学、几何判据、变量、约束和目标函数的完整规格。

## 当前结果

| 问题 | 当前结果 | 说明 |
|---|---:|---|
| 问题一 | `1.391643 s` | 固定策略；中心视线近似为 `1.435082 s` |
| 问题二 | `4.542880 s` | FY1 单机单弹的最佳已知严格遮蔽时长 |
| 问题三 | `5.513651 s` | FY1 三弹联合策略；当前候选实质由前两弹完成 |
| 问题四 | `10.138857 s` | FY1、FY2、FY3 各一弹形成三段接力 |
| 问题五 | `13.904024 s` | 三枚导弹遮蔽时长之和；`D=(10.250038, 1.912852, 1.741134) s` |

问题二至问题五都是固定评估预算下经过严格复算的最佳已知可行策略，不是已证明的全局最优解。问题五以总遮蔽导弹—秒数为目标，因此结果明显偏向 M1；若更重视三枚导弹的最低保障，需要改用 `J_min` 优先目标重新搜索。

## 目录说明

```text
2025A/
├── problem.md                 # 结构化题面
├── model-spec.md              # 可审计模型规格
├── analysis-log.md            # 建模决策与修正记录
├── paper-draft.md             # Markdown 论文终稿
├── validation-q1.md ... q5.md # 分问题验证报告
├── configs/                   # 问题一至问题五正式配置
├── data/
│   ├── raw/                   # 官方题目 PDF，只读
│   └── processed/             # 提取文本与页面图像
├── src/q1.py ... q5.py        # 题目专用实现
├── tests/                     # 联合覆盖、事件边界和回归测试
├── outputs/
│   ├── figures/               # 图形
│   ├── tables/                # CSV 与结果工作簿
│   ├── logs/                  # 优化、收敛和独立验证记录
│   └── manifest.json          # 随机种子、版本、输入哈希与产物清单
└── latex/                     # LaTeX、PDF、AI 披露和格式依据
```

## 运行环境

本目录是主代码库 `modeling` 的比赛项目，不复制公共 `mmkit` 包。推荐把它放在以下位置运行：

```text
modeling/
└── competitions/
    └── 2025A/
```

主代码库要求 Python 3.11 或更高版本，并使用 `uv` 管理环境：

```bash
cd /path/to/modeling
uv sync --frozen --dev
uv run python competitions/2025A/run.py
```

当前 `run.py` 执行问题五的正式检查点，并重写 `outputs/manifest.json`。该计算包含严格几何复算和验证，运行时间可能达到数分钟。问题一至问题四的正式结果已经保存在 `outputs/`，对应实现位于 `src/q1.py` 至 `src/q4.py`。

运行本题测试：

```bash
uv run pytest -q competitions/2025A/tests
uv run ruff check competitions/2025A
```

如果单独克隆本仓库，需要先安装主代码库提供的 `mmkit` 及其依赖；最简单的方式是把本仓库克隆到主代码库的 `competitions/2025A` 位置。

## 论文生成

LaTeX 入口与编译说明见 [latex/README.md](latex/README.md)。在主代码库根目录执行：

```bash
cd competitions/2025A/latex
../../../.venv/bin/python render.py
~/.local/bin/tectonic -X compile paper.tex --keep-logs
~/.local/bin/tectonic -X compile 'AI工具使用详情.tex' --keep-logs
```

- [paper.tex](latex/paper.tex) 默认展开完整源代码附录；
- [paper.pdf](latex/paper.pdf) 共 92 页，其中摘要为第 1 页、正文从第 2 页开始、参考文献和附录从第 16 页开始；
- [AI工具使用详情.pdf](latex/AI工具使用详情.pdf) 记录工具版本、使用环节、关键交互和人工修改；
- [FORMAT-SOURCES.md](latex/FORMAT-SOURCES.md) 记录联网核对的 2025 年格式依据及其与 2026 版的差异。

## 验证与可复现性

正式结果不是由时间网格点数近似得到，而是先定位状态变化，再使用连续求根确定区间边界。验证材料覆盖：

- 速度、投放时刻、引信、投弹间隔、起爆高度等可行性检查；
- 时间网格和圆柱表面网格收敛；
- 遮蔽区间起点前、区间内、终点后的边界探针；
- 独立视线线段—烟幕球相交复算；
- 多云标签置换、双云互补构造和前序问题回归；
- 随机种子、求解预算、停止原因和候选排名记录。

机器可读证据集中在 `outputs/logs/`，人工可读摘要集中在 `validation-q*.md`。

## 已知限制

- 烟幕按无风、固定半径、匀速下沉的理想球体处理；
- 导弹始终直指假目标，无制导反馈；无人机忽略转向和加速时间；
- 问题二至问题五没有全局最优证明；
- 问题三当前最佳已知策略没有有效利用第三枚弹；
- `result1.xlsx`、`result2.xlsx`、`result3.xlsx` 使用带 `provisional-schema` 标记的临时字段结构。获得官方模板后必须完成字段映射，但不需要重新求解。

## 使用约定

- `data/raw/` 视为只读，不覆盖原始题面；
- 所有计算产物写入 `outputs/`；
- 继续建模前先阅读 [AGENTS.md](AGENTS.md)、`problem.md`、`model-spec.md` 和 `analysis-log.md`；
- 不把一次性题目规则写入公共 `mmkit`；只有经过抽象、具有复用价值并能独立测试的实现才考虑提升到公共库。
