"""Dataset + manual (gold) analysis.

Each token: (form, gold UPOS, other PoS the form can take out of context).
Gold follows UD guidelines, restricted to the imposed list
ADJ ADP AUX DET NOUN NUM PROPN PUNCT SCONJ VERB.
PRON is used for 3 tokens (they, which, this) because no tag of the list fits them.

Target trees are the parses we want the CFG to produce, written in the
binarisation used by grammar.G_FULL (lexical NP/VP nodes = unit rules folded
into the lexicon, as required by Chomsky normal form).
"""

ALLOWED = ["ADJ", "ADP", "AUX", "DET", "NOUN", "NUM", "PROPN", "PUNCT", "SCONJ", "VERB"]

SENTENCES = [
    dict(id="S1", domain="general", text="The cat sat on the couch.", tokens=[
        ("The", "DET", ""), ("cat", "NOUN", ""), ("sat", "VERB", ""), ("on", "ADP", ""),
        ("the", "DET", ""), ("couch", "NOUN", "VERB"), (".", "PUNCT", "")],
        target=["(S (S (NP (DET The) (NOUN cat)) (VP (VP sat) (PP (ADP on) (NP (DET the) (NOUN couch))))) (PUNCT .))"]),

    dict(id="S2", domain="general", text="Time flies like an arrow.", tokens=[
        ("Time", "NOUN", "VERB"), ("flies", "VERB", "NOUN"), ("like", "ADP", "VERB"),
        ("an", "DET", ""), ("arrow", "NOUN", ""), (".", "PUNCT", "")],
        target=["(S (S (NP Time) (VP (VP flies) (PP (ADP like) (NP (DET an) (NOUN arrow))))) (PUNCT .))"]),

    dict(id="S3", domain="general", text="The spy saw the cop with the telescope.", tokens=[
        ("The", "DET", ""), ("spy", "NOUN", "VERB"), ("saw", "VERB", "NOUN"), ("the", "DET", ""),
        ("cop", "NOUN", "VERB"), ("with", "ADP", ""), ("the", "DET", ""), ("telescope", "NOUN", "VERB"),
        (".", "PUNCT", "")],
        # instrument reading: the PP modifies the verb phrase
        target=["(S (S (NP (DET The) (NOUN spy)) (VP (VP (VERB saw) (NP (DET the) (NOUN cop))) "
                "(PP (ADP with) (NP (DET the) (NOUN telescope))))) (PUNCT .))"]),

    dict(id="S4", domain="general", text="The spy saw the cop with the revolver.", tokens=[
        ("The", "DET", ""), ("spy", "NOUN", "VERB"), ("saw", "VERB", "NOUN"), ("the", "DET", ""),
        ("cop", "NOUN", "VERB"), ("with", "ADP", ""), ("the", "DET", ""), ("revolver", "NOUN", ""),
        (".", "PUNCT", "")],
        # attribute reading: the PP modifies "the cop"
        target=["(S (S (NP (DET The) (NOUN spy)) (VP (VERB saw) (NP (NP (DET the) (NOUN cop)) "
                "(PP (ADP with) (NP (DET the) (NOUN revolver)))))) (PUNCT .))"]),

    dict(id="S5", domain="biology",
         text="Arabidopsis thaliana seedlings exhibit longer hypocotyls when they are grown under "
              "high ambient temperature, which is defined as thermomorphogenesis.", tokens=[
        ("Arabidopsis", "PROPN", "NOUN"), ("thaliana", "PROPN", "NOUN"), ("seedlings", "NOUN", ""),
        ("exhibit", "VERB", "NOUN"), ("longer", "ADJ", ""), ("hypocotyls", "NOUN", ""),
        ("when", "SCONJ", ""), ("they", "PRON", ""), ("are", "AUX", ""), ("grown", "VERB", "ADJ"),
        ("under", "ADP", ""), ("high", "ADJ", "NOUN"), ("ambient", "ADJ", ""), ("temperature", "NOUN", ""),
        (",", "PUNCT", ""), ("which", "PRON", ""), ("is", "AUX", ""), ("defined", "VERB", "ADJ"),
        ("as", "ADP", "SCONJ"), ("thermomorphogenesis", "NOUN", ""), (".", "PUNCT", "")],
        # "which" refers to the whole phenomenon -> relative clause attached to the main clause
        target=["(S (S (S (NP (PROPN Arabidopsis) (NP (PROPN thaliana) (NP seedlings))) "
                "(VP (VP (VERB exhibit) (NP (ADJ longer) (NP hypocotyls))) "
                "(S (SCONJ when) (S (NP they) (VP (VP (AUX are) (VERB grown)) "
                "(PP (ADP under) (NP (ADJ high) (NP (ADJ ambient) (NP temperature))))))))) "
                "(S (PUNCT ,) (S (NP which) (VP (VP (AUX is) (VERB defined)) "
                "(PP (ADP as) (NP thermomorphogenesis)))))) (PUNCT .))"]),

    dict(id="S6", domain="astronomy",
         text="A spectrogram of PSN J10354824+3900279 obtained on Dec. 19.33 UT suggests that "
              "this is a type-Ia at redshift z 0.044.", tokens=[
        ("A", "DET", ""), ("spectrogram", "NOUN", ""), ("of", "ADP", ""), ("PSN", "PROPN", ""),
        ("J10354824+3900279", "PROPN", ""), ("obtained", "VERB", "ADJ"), ("on", "ADP", ""),
        ("Dec.", "PROPN", ""), ("19.33", "NUM", ""), ("UT", "PROPN", "NOUN"), ("suggests", "VERB", ""),
        ("that", "SCONJ", "DET PRON"), ("this", "PRON", "DET"), ("is", "AUX", ""), ("a", "DET", ""),
        ("type-Ia", "NOUN", "ADJ"), ("at", "ADP", ""), ("redshift", "NOUN", ""), ("z", "NOUN", ""),
        ("0.044", "NUM", ""), (".", "PUNCT", "")],
        # the spectrogram (not PSN) was obtained; the redshift locates the supernova (NP attachment)
        target=["(S (S (NP (NP (NP (DET A) (NOUN spectrogram)) (PP (ADP of) (NP (PROPN PSN) (NP J10354824+3900279)))) "
                "(VP (VP obtained) (PP (ADP on) (NP (PROPN Dec.) (NP (NUM 19.33) (NP UT)))))) "
                "(VP (VP suggests) (S (SCONJ that) (S (NP this) (VP (AUX is) (NP (NP (DET a) (NOUN type-Ia)) "
                "(PP (ADP at) (NP (NOUN redshift) (NP (NOUN z) (NP 0.044)))))))))) (PUNCT .))"]),
]


def gold_spans(sent):
    """Character offsets of gold tokens inside the raw text."""
    spans, pos = [], 0
    for form, _, _ in sent["tokens"]:
        start = sent["text"].index(form, pos)
        spans.append((start, start + len(form)))
        pos = start + len(form)
    return spans
