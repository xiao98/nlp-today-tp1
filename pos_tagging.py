"""Part 2a - automatic PoS tagging (spaCy vs sciSpaCy) evaluated against the manual gold."""
import json
from pathlib import Path

import spacy

from data import SENTENCES, gold_spans

MODELS = ["en_core_web_sm", "en_core_sci_sm"]
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def tag(nlp, text):
    return [dict(text=t.text, start=t.idx, end=t.idx + len(t), pos=t.pos_, tag=t.tag_) for t in nlp(text)]


def align(sent, pred):
    """For each gold token: the predicted UPOS if the model produced exactly the same token, else None."""
    by_span = {(p["start"], p["end"]): p for p in pred}
    return [by_span[s]["pos"] if s in by_span else None for s in gold_spans(sent)]


def main():
    nlps = {m: spacy.load(m) for m in MODELS}
    results = {"models": {m: f"{m}-{nlps[m].meta['version']} (spaCy {spacy.__version__})" for m in MODELS},
               "sentences": []}
    totals = {m: {"general": [0, 0, 0], "specialty": [0, 0, 0]} for m in MODELS}  # correct, tok-mismatch, n

    for sent in SENTENCES:
        row = dict(id=sent["id"], domain=sent["domain"], gold=[(f, g) for f, g, _ in sent["tokens"]], pred={}, aligned={})
        for m, nlp in nlps.items():
            pred = tag(nlp, sent["text"])
            row["pred"][m] = pred
            row["aligned"][m] = align(sent, pred)
            bucket = totals[m]["general" if sent["domain"] == "general" else "specialty"]
            for (_, g), p in zip(row["gold"], row["aligned"][m]):
                bucket[0] += p == g
                bucket[1] += p is None
                bucket[2] += 1
        results["sentences"].append(row)

        print(f"\n== {sent['id']} ({sent['domain']})")
        print(f"{'token':22s} {'gold':6s} " + " ".join(f"{m:15s}" for m in MODELS))
        for i, (form, g) in enumerate(row["gold"]):
            cells = []
            for m in MODELS:
                p = row["aligned"][m][i]
                cells.append(f"{'<tok>' if p is None else p:6s}{'' if p == g else ' X':9s}")
            print(f"{form:22s} {g:6s} " + " ".join(cells))
        for m in MODELS:
            print(f"  {m} tokens: {[p['text'] + '/' + p['pos'] + '/' + p['tag'] for p in row['pred'][m]]}")

    results["accuracy"] = {m: {d: dict(correct=c, tok_mismatch=t, n=n, acc=round(c / n, 3)) for d, (c, t, n) in v.items()}
                           for m, v in totals.items()}
    print("\n", json.dumps(results["accuracy"], indent=1))
    (OUT / "pos.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
