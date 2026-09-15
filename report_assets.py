"""Turn results/*.json into LaTeX fragments (tables + forest trees) included by report/report.tex."""
import json
from pathlib import Path

from nltk import Tree

from data import ALLOWED, SENTENCES, gold_spans
from grammar import G_FULL, G_GENERAL, G_GENERAL_IMP, LEXICON, parse

R = Path("results")
OUT = Path("report/gen")
OUT.mkdir(parents=True, exist_ok=True)
POS = json.loads((R / "pos.json").read_text(encoding="utf-8"))
CYK = json.loads((R / "cyk.json").read_text(encoding="utf-8"))
BEN = {b["id"]: b for b in json.loads((R / "benepar.json").read_text(encoding="utf-8"))}
MODELS = ["en_core_web_sm", "en_core_sci_sm"]


def tex(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write(name, content):
    (OUT / name).write_text(content, encoding="utf-8")


# ---------- trees ----------
def forest(bracket):
    """Bracketed tree -> forest code (words aligned on one tier)."""
    t = Tree.fromstring(bracket)
    if t.label() == "TOP":
        t = t[0]

    def rec(node):
        if isinstance(node, str):
            return f"[{{{tex(node)}}}, tier=word]"
        return f"[{{{tex(node.label())}}} {' '.join(rec(c) for c in node)}]"

    return rec(t)


def tree_env(bracket):
    """forest tree, shrunk only if wider than the current line."""
    return (r"\begin{adjustbox}{max width=\linewidth}\begin{forest} for tree={s sep=1.2mm, l sep=2pt, l=0, "
            r"inner sep=1pt, font=\footnotesize}, where n children=0{font=\footnotesize\itshape}{} "
            + forest(bracket) + r" \end{forest}\end{adjustbox}")


# ---------- Part 1: gold annotation ----------
DAGGER = r"$^\dagger$"
lines = []
for s in SENTENCES:
    toks = " ".join(rf"{tex(f)}\tg{{{g}{DAGGER if g not in ALLOWED else ''}}}" for f, g, _ in s["tokens"])
    lines.append(rf"\item[\textbf{{{s['id']}}}] \emph{{({s['domain']})}} {toks}")
write("gold.tex", "\\begin{description}[leftmargin=2.2em, style=sameline, itemsep=2pt]\n" + "\n".join(lines) + "\n\\end{description}\n")

rows = []
for s in SENTENCES:
    alts = [rf"\textit{{{tex(f)}}}: {g} / {', '.join(a.split())}" for f, g, a in s["tokens"] if a]
    rows.append(rf"{s['id']} & {'; '.join(alts)} \\")
write("ambiguous_words.tex", "\n".join(rows) + "\n")

for s in SENTENCES:
    write(f"target_{s['id']}.tex", tree_env(s["target"][0]))

# ---------- Part 2a: PoS tagging ----------
rows = []
for s, p in zip(SENTENCES, POS["sentences"]):
    spans = gold_spans(s)
    for i, (f, g, _) in enumerate(s["tokens"]):
        if all(p["aligned"][m][i] == g for m in MODELS):
            continue
        cells = []
        for m in MODELS:
            a, b = spans[i]
            over = [t for t in p["pred"][m] if t["start"] < b and t["end"] > a]
            txt = " ".join(rf"{tex(t['text'])}\tg{{{t['pos']}/{tex(t['tag'])}}}" if len(over) > 1
                           else rf"{t['pos']}\tg{{{tex(t['tag'])}}}" for t in over)
            ok = p["aligned"][m][i] == g
            cells.append(txt if ok else rf"\textcolor{{red}}{{{txt}}}")
        rows.append(rf"{s['id']} & \textit{{{tex(f)}}} & {g} & {cells[0]} & {cells[1]} \\")
write("pos_diff.tex", "\n".join(rows) + "\n")

rows = []
for m in MODELS:
    acc = POS["accuracy"][m]
    tot_c = acc["general"]["correct"] + acc["specialty"]["correct"]
    tot_n = acc["general"]["n"] + acc["specialty"]["n"]
    cell = lambda d: rf"{d['correct']}/{d['n']} ({100 * d['correct'] / d['n']:.1f}\%)"
    name, version = POS["models"][m].split(" (")[0].rsplit("-", 1)
    rows.append(rf"\texttt{{{tex(name)}}} {version} & {cell(acc['general'])} & {cell(acc['specialty'])} & "
                rf"{tot_c}/{tot_n} ({100 * tot_c / tot_n:.1f}\%) & {acc['specialty']['tok_mismatch']} \\")
write("pos_acc.tex", "\n".join(rows) + "\n")

# ---------- Part 2b: grammar + CYK ----------
def rule(r):
    return rf"{r[0]} $\to$ {' '.join(r[1])}"


added = [r for r in G_FULL if r not in G_GENERAL]
write("grammar_lexicon.tex", ", ".join(rule(r) for r in LEXICON if r[0] != r[1][0]))
write("grammar_general.tex", ", ".join(rule(r) for r in G_GENERAL if len(r[1]) == 2))
write("grammar_imp.tex", ", ".join(rule(r) for r in G_GENERAL_IMP if r not in G_GENERAL))
write("grammar_added.tex", ", ".join(rule(r) for r in added))

res = CYK["results"]
GR = ["G_general", "G_general+imp", "G_full", "G_treebank"]
INP = [("gold", "manual"), ("spacy", "spaCy"), ("scispacy", "sciSpaCy"), ("ambiguous", "ambiguous")]
rows = []
for s in SENTENCES:
    for k, (inp, label) in enumerate(INP):
        cells = []
        for g in GR:
            r = next(x for x in res if x["id"] == s["id"] and x["input"] == inp and x["grammar"] == g
                     and not x["strip_final_punct"])
            n = r["n_S"]
            if r["target_index"] is not None:
                cells.append(rf"\textbf{{{n}}}\,\ok")
            elif not r["enumerated"]:
                cells.append(rf"{n}\,\notenum")  # counted by dynamic programming, trees not built
            else:
                cells.append(str(n) if n else r"\zero")
        head = rf"\multirow{{4}}{{*}}{{{s['id']}}}" if k == 0 else ""
        rows.append(rf"{head} & {label} & {' & '.join(cells)} \\")
    rows.append(r"\midrule" if s["id"] != "S6" else "")
write("cyk_counts.tex", "\n".join(rows) + "\n")

punct = []
for sid in ["S5", "S6"]:
    for g in ["G_full", "G_treebank"]:
        keep, strip = (next(x for x in res if x["id"] == sid and x["input"] == "gold" and x["grammar"] == g
                            and x["strip_final_punct"] == st)["n_S"] for st in (False, True))
        punct.append(rf"{sid} & {g.replace('_', chr(92) + '_')} & {keep} & {strip} \\")
write("punct.tex", "\n".join(punct) + "\n")

# Time flies like an arrow: the 4 readings of G_general+imp
s2 = next(x for x in res if x["id"] == "S2" and x["input"] == "ambiguous" and x["grammar"] == "G_general+imp"
          and not x["strip_final_punct"])
_, trees = parse(G_GENERAL_IMP, s2["tags"][:-1], s2["words"][:-1])
write("time_flies.tex", "\n".join(rf"\begin{{minipage}}[b]{{0.49\linewidth}}\centering {tree_env(t)}\\[-2pt]"
                                  rf"\small ({chr(97 + i)})\end{{minipage}}" + ("\\hfill" if i % 2 == 0 else "\\\\[6pt]")
                                  for i, t in enumerate(trees)))
write("s2_notes.tex", str(CYK["notes"]["S2_ambiguous_G_full+imp_n_S"]))
write("s2_tags.tex", ", ".join(rf"\textit{{{w}}}: \{{{', '.join(t)}\}}" for w, t in zip(s2["words"], s2["tags"])
                               if t != ["PUNCT"]))

# ---------- Part 2c: benepar ----------
for sid in ["S4", "S5"]:
    write(f"benepar_{sid}.tex", tree_env(BEN[sid]["gold_tokens"]))
print("assets written to", OUT)
