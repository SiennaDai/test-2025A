# LaTeX 论文与 AI 使用详情

本目录依据 2025 年全国大学生数学建模竞赛论文格式规范排版：A4 纸、四边页边距 2.5 cm、摘要作为电子版第一页并从 1 编页码、不生成目录、正文后附支撑材料列表和完整源程序。

## 生成与编译

在 `competitions/2025A/latex` 下运行：

```bash
../../../.venv/bin/python render.py
~/.local/bin/tectonic -X compile paper.tex --keep-logs
~/.local/bin/tectonic -X compile 'AI工具使用详情.tex' --keep-logs
```

`render.py` 仅把 `../paper-draft.md` 转换为 `abstract.tex` 和 `body.tex`，不改写原 Markdown。正式论文入口为 `paper.tex`。

本机已把 Tectonic 0.15.0 长期安装到 `~/.local/bin/tectonic`。也可使用 XeLaTeX 编译；如采用 XeLaTeX，需连续运行两次以刷新交叉引用。

正式提交稿默认展开 `run.py` 与 `src/q1.py`—`src/q5.py` 的完整源代码。只检查正文页数或快速预览时，可暂时把 `paper.tex` 中的 `\fullsourceappendixtrue` 改为 `\fullsourceappendixfalse`；提交前必须恢复。

## 提交前检查

- `paper.pdf` 第一页应为摘要，不得包含承诺书和编号专用页；
- 摘要（含标题、关键词）不得超过一页，正文从第二个 PDF 页面开始；
- 正文尽量控制在 20 页内，附录页数不限，整个 PDF 不超过 20 MB；
- `paper.pdf` 和支撑材料中不得出现参赛者、学校或赛区身份信息；
- 支撑材料另行打包为 ZIP/RAR，且不超过 20 MB；
- 将 `AI工具使用详情.pdf` 放入支撑材料，文件名保持不变；
- 获得官方 `result1.xlsx`—`result3.xlsx` 模板后，替换当前 `provisional-schema` 工作簿；
- 逐式、逐表核对 PDF 与 `outputs/` 正式结果一致。

联网核对的规则来源及 2025/2026 版本差异见 `FORMAT-SOURCES.md`。
