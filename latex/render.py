"""Render the reviewed Markdown paper into deterministic LaTeX fragments.

The converter intentionally supports only the Markdown constructs used by
``paper-draft.md``.  It has no third-party dependencies and keeps equations
verbatim so that the mathematical source remains auditable.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "paper-draft.md"


def escape_plain(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


INLINE_TOKEN = re.compile(r"(\$[^$]+\$|`[^`]+`|\*\*[^*]+\*\*)")


def inline(text: str) -> str:
    parts: list[str] = []
    for part in INLINE_TOKEN.split(text):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            parts.append(part.replace(",", r",\allowbreak{}") if len(part) > 20 else part)
        elif part.startswith("`") and part.endswith("`"):
            value = part[1:-1].replace("\n", " ")
            parts.append(r"\nolinkurl{" + value + "}")
        elif part.startswith("**") and part.endswith("**"):
            parts.append(r"\textbf{" + escape_plain(part[2:-2]) + "}")
        else:
            parts.append(escape_plain(part))
    return "".join(parts)


def table_block(rows: list[str]) -> list[str]:
    parsed = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    parsed = [row for index, row in enumerate(parsed) if index != 1]
    column_count = len(parsed[0])
    landscape = column_count >= 7
    size = r"\scriptsize" if column_count >= 6 else r"\small"
    resize = column_count >= 4
    column_kind = "c" if resize else "X"
    columns = "|" + "|".join(column_kind for _ in range(column_count)) + "|"
    result: list[str] = []
    if landscape:
        result.append(r"\begin{landscape}")
    result.extend(
        [
            r"\begin{center}",
            size,
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.18}",
            *([r"\resizebox{\linewidth}{!}{%"] if resize else []),
            (
                rf"\begin{{tabular}}{{{columns}}}"
                if resize
                else rf"\begin{{tabularx}}{{\linewidth}}{{{columns}}}"
            ),
            r"\hline",
        ]
    )
    for index, row in enumerate(parsed):
        cells = [inline(cell) for cell in row]
        if index == 0:
            cells = [r"\textbf{" + cell + "}" for cell in cells]
            result.append(r"\rowcolor{gray!15} " + " & ".join(cells) + r" \\")
        else:
            result.append(" & ".join(cells) + r" \\")
        result.append(r"\hline")
    result.append(r"\end{tabular}" if resize else r"\end{tabularx}")
    if resize:
        result.append(r"}")
    result.append(r"\end{center}")
    if landscape:
        result.append(r"\end{landscape}")
    return result


def strip_number(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\s*", "", title).strip()


def convert(lines: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    in_math = False
    while index < len(lines):
        line = lines[index].rstrip()
        if line == "$$":
            output.append(r"\[" if not in_math else r"\]")
            in_math = not in_math
            index += 1
            continue
        if in_math:
            output.append(line)
            index += 1
            continue
        if line.startswith("## "):
            output.append(r"\section{" + inline(strip_number(line[3:])) + "}")
            index += 1
            continue
        if line.startswith("### "):
            output.append(r"\subsection{" + inline(strip_number(line[4:])) + "}")
            index += 1
            continue
        if (
            line.startswith("|")
            and index + 1 < len(lines)
            and re.match(r"^\|[\s:|-]+\|$", lines[index + 1])
        ):
            table_rows = [line]
            index += 1
            while index < len(lines) and lines[index].startswith("|"):
                table_rows.append(lines[index])
                index += 1
            output.extend(table_block(table_rows))
            continue
        numbered = re.match(r"^\d+\.\s+(.*)$", line)
        if numbered:
            output.append(r"\begin{enumerate}")
            while index < len(lines):
                match = re.match(r"^\d+\.\s+(.*)$", lines[index].rstrip())
                if not match:
                    break
                output.append(r"\item " + inline(match.group(1)))
                index += 1
            output.append(r"\end{enumerate}")
            continue
        bullet = re.match(r"^-\s+(.*)$", line)
        if bullet:
            output.append(r"\begin{itemize}")
            while index < len(lines):
                match = re.match(r"^-\s+(.*)$", lines[index].rstrip())
                if not match:
                    break
                output.append(r"\item " + inline(match.group(1)))
                index += 1
            output.append(r"\end{itemize}")
            continue
        if line:
            output.append(inline(line))
        else:
            output.append("")
        index += 1
    return output


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("paper-draft.md must start with an H1 title")

    title = lines[0][2:].strip()
    abstract_start = lines.index("## 摘要") + 1
    body_start = next(index for index, line in enumerate(lines) if line.startswith("## 1 "))
    appendix_start = lines.index("## 附录与排版说明")

    abstract_lines = lines[abstract_start:body_start]
    while abstract_lines and not abstract_lines[0].strip():
        abstract_lines.pop(0)
    while abstract_lines and not abstract_lines[-1].strip():
        abstract_lines.pop()

    keyword_line = abstract_lines.pop()
    if not keyword_line.startswith("**关键词**："):
        raise ValueError("abstract must end with a Markdown keyword line")
    keywords = keyword_line.removeprefix("**关键词**：")

    abstract_tex = [r"\PaperTitle{" + inline(title) + "}", ""]
    abstract_tex.extend(convert(abstract_lines))
    abstract_tex.extend(["", r"\noindent\textbf{关键词：}" + inline(keywords)])
    (HERE / "abstract.tex").write_text("\n".join(abstract_tex) + "\n", encoding="utf-8")

    body_tex = convert(lines[body_start:appendix_start])
    while body_tex and not body_tex[-1]:
        body_tex.pop()
    if body_tex and body_tex[0].startswith(r"\section{问题重述}"):
        body_tex.insert(
            1,
            r"本题参数与任务定义依据 2025 年全国大学生数学建模竞赛 A 题\cite{cumcm2025a}。",
        )
    (HERE / "body.tex").write_text("\n".join(body_tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
