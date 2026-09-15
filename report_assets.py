"""Turn results/*.json into the LaTeX tables and trees included by report/report.tex."""
import json
from pathlib import Path

from nltk import Tree

from data import ALLOWED, SENTENCES, gold_spans
from grammar import G1, G2

R = Path("results")
OUT = Path("report/gen")
OUT.mkdir(parents=True, exist_ok=True)
POS = json.loads((R / "pos.json").read_text(encoding="utf-8"))
CYK = json.loads((R / "cyk.json").read_text(encoding="utf-8"))
MODELS = ["en_core_web_sm", "en_core_sci_sm"]


def tex(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write(name, content):
    (OUT / name).write_text(content, encoding="utf-8")


def tree(bracket):
    """Bracketed tree -> forest picture, shrunk only if wider than the line."""
    def rec(node):
        if isinstance(node, str):
            return f"[{{{tex(node)}}}, tier=word]"
        return f"[{{{node.label()}}} {' '.join(rec(c) for c in node)}]"

    return (r"\begin{adjustbox}{max width=\linewidth}\begin{forest} for tree={s sep=1.2mm, l sep=2pt, l=0, "
            r"inner sep=1pt, font=\footnotesize}, where n children=0{font=\footnotesize\itshape}{} "
            + rec(Tree.fromstring(bracket)) + r" \end{forest}\end{adjustbox}")


# Part 1: manual tags and expected trees
dagger = r"$^\dagger$"
lines = []
for s in SENTENCES:
    toks = " ".join(rf"{tex(f)}\tg{{{g}{dagger if g not in ALLOWED else ''}}}" for f, g, _ in s["tokens"])
    lines.append(rf"\item[{s['id']}] {toks}")
write("gold.tex", "\\begin{description}[leftmargin=2em, style=sameline, itemsep=1pt]\n" + "\n".join(lines)
      + "\n\\end{description}\n")
alternatives = dict.fromkeys((f, a) for s in SENTENCES for f, _, a in s["tokens"] if a)  # S3/S4 share words
write("alternatives.tex", ", ".join(rf"\emph{{{tex(f)}}} ({', '.join(a.split())})" for f, a in alternatives))
for s in SENTENCES:
    write(f"target_{s['id']}.tex", tree(s["target"][0]))

# Part 2a: PoS tagging tables
rows = []
for m in MODELS:
    a = POS["accuracy"][m]
    name, version = POS["models"][m].split(" (")[0].rsplit("-", 1)
    c, n = a["general"]["correct"] + a["specialty"]["correct"], a["general"]["n"] + a["specialty"]["n"]
    cell = lambda d: rf"{d['correct']}/{d['n']} ({100 * d['correct'] / d['n']:.1f}\%)"
    rows.append(rf"\texttt{{{tex(name)}}} {version} & {cell(a['general'])} & {cell(a['specialty'])} & "
                rf"{c}/{n} ({100 * c / n:.1f}\%) \\")
write("pos_acc.tex", "\n".join(rows) + "\n")

rows = []
for s, p in zip(SENTENCES, POS["sentences"]):
    spans = gold_spans(s)
    for i, (f, g, _) in enumerate(s["tokens"]):
        if all(p["aligned"][m][i] == g for m in MODELS):
            continue
        cells = []
        for m in MODELS:
            start, end = spans[i]
            tokens = [t for t in p["pred"][m] if t["start"] < end and t["end"] > start]
            if len(tokens) == 1:
                cells.append(tokens[0]["pos"])
            else:  # the model split our token
                cells.append(" ".join(rf"{tex(t['text'])}\tg{{{t['pos']}}}" for t in tokens))
        rows.append(rf"{s['id']} & \textit{{{tex(f)}}} & {g} & {cells[0]} & {cells[1]} \\")
write("pos_diff.tex", "\n".join(rows) + "\n")

# Part 2b: grammars, CYK counts, Time flies trees
rule = lambda r: rf"{r[0]} $\to$ {' '.join(r[1])}"
write("grammar_g1.tex", ", ".join(rule(r) for r in G1 if len(r[1]) == 2))
write("grammar_g2.tex", ", ".join(rule(r) for r in G2 if r not in G1))

rows = []
for s in SENTENCES:
    g = "G1" if s["domain"] == "general" else "G2"
    cells = []
    for inp in ["manual", "spacy", "scispacy", "multiple"]:
        r = next(x for x in CYK if x["id"] == s["id"] and x["input"] == inp and x["grammar"] == g)
        cells.append(rf"{r['n_trees']}\,\checkmark" if r["expected_found"] else str(r["n_trees"]))
    rows.append(rf"{s['id']} & {g} & {' & '.join(cells)} \\")
write("cyk_counts.tex", "\n".join(rows) + "\n")

s2 = next(x for x in CYK if x["id"] == "S2" and x["input"] == "multiple" and x["grammar"] == "G1+imp")
write("time_flies.tex", "\n".join(
    rf"\begin{{minipage}}[b]{{0.49\linewidth}}\centering {tree(t)}\\ \small ({'abcd'[i]})\end{{minipage}}"
    + ("\\hfill" if i % 2 == 0 else "\\\\[6pt]") for i, t in enumerate(s2["trees"])))
print("assets written to", OUT)
