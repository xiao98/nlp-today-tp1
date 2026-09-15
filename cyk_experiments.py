"""Part 2b - CYK parsing of the six sentences with different PoS inputs."""
import json
from pathlib import Path

from data import SENTENCES
from grammar import G1, G1_IMP, G2, parse

OUT = Path("results")
pos = {s["id"]: s for s in json.loads((OUT / "pos.json").read_text(encoding="utf-8"))["sentences"]}


def inputs(sent):
    """PoS inputs fed to CYK: (words, list of possible tags per word)."""
    words = [f for f, _, _ in sent["tokens"]]
    p = pos[sent["id"]]
    multiple = []
    for i, (_, gold, alt) in enumerate(sent["tokens"]):
        tags = [gold] + alt.split() + [a[i] for a in p["aligned"].values() if a[i] is not None]
        multiple.append(list(dict.fromkeys(tags)))
    res = {"manual": (words, [[g] for _, g, _ in sent["tokens"]])}
    for model, name in [("en_core_web_sm", "spacy"), ("en_core_sci_sm", "scispacy")]:
        res[name] = ([t["text"] for t in p["pred"][model]], [[t["pos"]] for t in p["pred"][model]])
    res["multiple"] = (words, multiple)
    return res


def main():
    results = []
    for sent in SENTENCES:
        grammars = {"G1": G1, "G2": G2}
        if sent["id"] == "S2":
            grammars["G1+imp"] = G1_IMP
        for inp, (words, tags) in inputs(sent).items():
            for g_name, g in grammars.items():
                trees = parse(g, tags, words)
                found = sent["target"][0] in trees
                results.append(dict(id=sent["id"], input=inp, grammar=g_name, tags=tags, n_trees=len(trees),
                                    expected_found=found, trees=trees if len(trees) <= 4 else None))
                print(f"{sent['id']} {inp:9s} {g_name:7s} {len(trees):5d} trees  expected tree found: {found}")
    (OUT / "cyk.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
