"""Small LaTeX helpers shared by the table generators."""
import re

_BEGIN = re.compile(r"\\begin\{(table\*?)\}")


def _balanced(text, start):
    """Index just past the brace group that opens at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def caption_below(tex):
    """Move the caption and label of every table environment below the tabular."""
    out, pos = [], 0
    for m in _BEGIN.finditer(tex):
        if m.start() < pos:
            continue
        env = m.group(1)
        end_tag = "\\end{%s}" % env
        stop = tex.index(end_tag, m.end())
        body = tex[m.end():stop]
        moved = []
        for cmd in ("\\caption", "\\label"):
            k = body.find(cmd + "{")
            if k < 0:
                continue
            j = _balanced(body, k + len(cmd))
            a = body.rfind("\n", 0, k) + 1
            b = j + 1 if j < len(body) and body[j] == "\n" else j
            moved.append(body[k:j])
            body = body[:a] + body[b:] if body[a:k].strip() == "" else body[:k] + body[j:]
        if moved:
            body = body.rstrip() + "\n" + "".join("  " + x + "\n" for x in moved)
        out.append(tex[pos:m.end()] + body)
        pos = stop
    out.append(tex[pos:])
    return "".join(out)
