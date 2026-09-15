"""Part 2b - CYK constituency parsing with different PoS inputs and grammars."""
import json
from collections import Counter
from pathlib import Path

from nltk import Tree

from data import SENTENCES
from grammar import G_FULL, G_GENERAL, G_GENERAL_IMP, parse, with_imperative

OUT = Path("results")
pos = {s["id"]: s for s in json.loads((OUT / "pos.json").read_text(encoding="utf-8"))["sentences"]}

# rules read off the gold target trees (= G_FULL without the one rule no target uses)
G_TREEBANK = [r for r in G_FULL if r != ("NP", ("NP", "S"))]
GRAMMARS = {"G_general": G_GENERAL, "G_general+imp": G_GENERAL_IMP, "G_full": G_FULL, "G_treebank": G_TREEBANK}
INPUTS = ["gold", "spacy", "scispacy", "ambiguous"]


def inputs(sent):
    """PoS inputs fed to CYK: (words, list of candidate tags per word)."""
    words = [f for f, _, _ in sent["tokens"]]
    gold = [[g] for _, g, _ in sent["tokens"]]
    p = pos[sent["id"]]
    ambiguous = []
    for i, (_, g, alt) in enumerate(sent["tokens"]):
        cands = [g] + alt.split()
        cands += [a[i] for a in p["aligned"].values() if a[i] is not None]
        ambiguous.append(list(dict.fromkeys(cands)))
    res = {"gold": (words, gold), "ambiguous": (words, ambiguous)}
    for m, name in [("en_core_web_sm", "spacy"), ("en_core_sci_sm", "scispacy")]:
        toks = p["pred"][m]
        res[name] = ([t["text"] for t in toks], [[t["pos"]] for t in toks])
    return res


def spans(tree):
    """Labelled spans of phrase nodes (lexical nodes '(X word)' excluded)."""
    out = set()

    def rec(node, i):
        if isinstance(node, str):
            return i + 1
        j = i
        for child in node:
            j = rec(child, j)
        if not (len(node) == 1 and isinstance(node[0], str)):
            out.add((node.label(), i, j))
        return j

    rec(tree, 0)
    return out


def disputed_constituents(trees, words):
    """Constituents present in some but not all parses = the decisions the grammar cannot make."""
    parsed = [spans(Tree.fromstring(t)) for t in trees]
    freq = Counter(sp for s in parsed for sp in s)
    return [dict(label=l, span=[i, j], text=" ".join(words[i:j]), share=f"{k}/{len(parsed)}")
            for (l, i, j), k in sorted(freq.items(), key=lambda kv: (kv[0][1], -kv[0][2])) if k < len(parsed)]


def main():
    results = []
    for sent in SENTENCES:
        target = sent["target"][0]
        for inp_name, (words, tags) in inputs(sent).items():
            for strip in (False, True):
                w, t = (words[:-1], tags[:-1]) if strip and tags[-1] == ["PUNCT"] else (words, tags)
                for g_name, g in GRAMMARS.items():
                    counts, trees = parse(g, t, w)
                    r = dict(id=sent["id"], input=inp_name, strip_final_punct=strip, grammar=g_name,
                             tags=t, words=w, roots=counts, n_S=counts.get("S", 0), enumerated=trees is not None,
                             target_index=None, trees=None, disputed=None)
                    if trees is not None:
                        if not strip and target in trees:
                            r["target_index"] = trees.index(target)
                        r["trees"] = trees if len(trees) <= 12 else None
                        if inp_name == "gold" and g_name == "G_treebank" and strip and len(trees) > 2:
                            r["disputed"] = disputed_constituents(trees, w)
                    results.append(r)

    # interaction effect: the imperative rules added to the full grammar (Time flies like an arrow)
    s2 = next(s for s in SENTENCES if s["id"] == "S2")
    w, t = inputs(s2)["ambiguous"]
    notes = {"S2_ambiguous_G_full+imp_n_S": parse(with_imperative(G_FULL), t, w)[0].get("S", 0)}

    (OUT / "cyk.json").write_text(json.dumps(dict(results=results, notes=notes), indent=1, ensure_ascii=False),
                                  encoding="utf-8")

    print("number of S-rooted parses (* = target tree found, ? = too many to enumerate)")
    print(f"{'sent':4s} {'input':10s} {'punct':6s} " + " ".join(f"{g:>14s}" for g in GRAMMARS))
    for sent in SENTENCES:
        for inp_name in INPUTS:
            for strip in (False, True):
                cells = []
                for g_name in GRAMMARS:
                    r = next(x for x in results if x["id"] == sent["id"] and x["input"] == inp_name
                             and x["strip_final_punct"] == strip and x["grammar"] == g_name)
                    mark = "*" if r["target_index"] is not None else ("?" if not r["enumerated"] and not strip else "")
                    cells.append(f"{r['n_S']}{mark}".rjust(14))
                print(f"{sent['id']:4s} {inp_name:10s} {'strip' if strip else 'keep':6s} " + " ".join(cells))
    print(notes)


if __name__ == "__main__":
    main()
