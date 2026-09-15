"""Part 2c - neural constituency parser (Berkeley Neural Parser, benepar_en3) for comparison."""
import json
from pathlib import Path

import benepar
import spacy

from data import SENTENCES

# sentencepiece cannot open files under a non-ASCII Windows path -> model copied to an ASCII folder
MODEL = "C:/Users/Public/nlp_models/benepar_en3"
OUT = Path("results")


def main():
    parser = benepar.Parser(MODEL)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("benepar", config={"model": MODEL})
    results = []
    for sent in SENTENCES:
        words = [f for f, _, _ in sent["tokens"]]
        gold_tok = parser.parse(benepar.InputSentence(words=words))       # our tokenisation
        doc = nlp(sent["text"])                                           # spaCy pipeline as-is
        spacy_trees = [s._.parse_string for s in doc.sents]
        results.append(dict(id=sent["id"], gold_tokens=" ".join(str(gold_tok).split()),
                            spacy_pipeline=spacy_trees))
        print(f"== {sent['id']}\n  gold tokens : {results[-1]['gold_tokens']}")
        for t in spacy_trees:
            print(f"  spaCy+benepar: {t}")
    (OUT / "benepar.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
