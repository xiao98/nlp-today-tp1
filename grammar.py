"""Grammars in Chomsky normal form for the provided cyk.CYK."""
from cyk import CYK

TAGS = ["ADJ", "ADP", "AUX", "DET", "NOUN", "NUM", "PROPN", "PRON", "PUNCT", "SCONJ", "VERB"]

# each tag is a symbol; one-word NPs and VPs are lexical rules (cyk.py has no unary rule X -> Y)
LEXICON = [(t, (t,)) for t in TAGS] + [
    ("NP", ("NOUN",)), ("NP", ("PROPN",)), ("NP", ("PRON",)), ("NP", ("NUM",)),
    ("VP", ("VERB",)),
]

# G1: general-domain sentences
G1 = LEXICON + [
    ("S", ("NP", "VP")),
    ("S", ("S", "PUNCT")),     # final punctuation
    ("NP", ("DET", "NOUN")),   # the cat
    ("NP", ("NOUN", "NP")),    # noun compound (time flies)
    ("NP", ("NP", "PP")),      # the cop with the revolver
    ("PP", ("ADP", "NP")),
    ("VP", ("VERB", "NP")),    # saw the cop
    ("VP", ("VP", "PP")),      # sat on the couch
]

# G1 + imperative sentences (S -> VP written in CNF)
G1_IMP = G1 + [("S", ("VERB", "NP")), ("S", ("VP", "PP"))]

# G2: G1 + rules needed by the scientific sentences
G2 = G1 + [
    ("NP", ("ADJ", "NP")),     # high ambient temperature
    ("NP", ("PROPN", "NP")),   # Arabidopsis thaliana seedlings
    ("NP", ("NUM", "NP")),     # 19.33 UT
    ("NP", ("NP", "VP")),      # a spectrogram obtained on ...
    ("VP", ("AUX", "VERB")),   # are grown
    ("VP", ("AUX", "NP")),     # is a type-Ia
    ("VP", ("VP", "S")),       # suggests that ... / exhibit ... when ...
    ("S", ("SCONJ", "S")),     # when they are grown ... (SBAR is not allowed)
    ("S", ("PUNCT", "S")),     # , which is defined as ...
    ("S", ("S", "S")),         # clause attached to a clause
]


def bracket(node):
    """Node -> '(S (NP (DET The) (NOUN cat)) ...)'."""
    if node.word is not None:
        return f"({node.type} {node.word})"
    return f"({node.type} {bracket(node.left)} {bracket(node.right)})"


def parse(grammar, tags, words):
    """S-rooted trees returned by the provided CYK, as bracketed strings."""
    return [bracket(t) for t in CYK(grammar)(tags, words) if t.type == "S"]
