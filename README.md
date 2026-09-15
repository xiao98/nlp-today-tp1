# NLP Today – Session 1: PoS tagging & constituency parsing

Université Paris-Saclay, *NLP Today* (T. Gerald). Lab report for session 1 ([instructions](https://thomas-gerald.fr/TMC/index.html)).

**Report:** [`report/report.pdf`](report/report.pdf)

| File | Content |
|---|---|
| `data.py` | the 6 sentences, manual PoS annotation, target constituency trees |
| `pos_tagging.py` | spaCy `en_core_web_sm` vs sciSpaCy `en_core_sci_sm`, scored against the manual tags |
| `cyk.py` | CYK implementation provided by the course (unmodified) |
| `grammar.py` | grammars G1 (general sentences) and G2 (+ scientific sentences) in Chomsky normal form |
| `cyk_experiments.py` | CYK with manual, spaCy, sciSpaCy and multiple-PoS inputs |
| `benepar_parse.py` | neural constituency parser (`benepar_en3`) for comparison |
| `report_assets.py` | turns `results/*.json` into the LaTeX tables and trees in `report/gen/` |

## Reproduce (Python 3.11)

```bash
uv venv --python 3.11 .venv
uv pip install "spacy==3.7.5" "scispacy==0.6.2" "numpy<2.0" nltk \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install benepar "transformers==4.46.3" "tokenizers<0.21" "huggingface-hub<1.0"
python -c "import benepar; benepar.download('benepar_en3')"

python pos_tagging.py && python cyk_experiments.py && python benepar_parse.py && python report_assets.py
cd report && pdflatex report.tex && pdflatex report.tex
```

Pitfalls:
- benepar 0.2.0 does not work with transformers 5, hence the 4.46.3 pin.
- `benepar_parse.py` loads the model from `MODEL`. On Windows, sentencepiece cannot read a path containing non-ASCII characters, so copy the model folder to an ASCII path and point `MODEL` to it.
