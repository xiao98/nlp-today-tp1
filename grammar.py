"""Grammars in Chomsky normal form for cyk.CYK + small helpers around the provided implementation."""
from cyk import CYK

TAGS = ["ADJ", "ADP", "AUX", "DET", "NOUN", "NUM", "PROPN", "PRON", "PUNCT", "SCONJ", "VERB"]

# Grammar given in the docstring of cyk.py (no punctuation rule).
G_DOC = [
    ("S", ("NP", "VP")),
    ("NP", ("DET", "NOUN")), ("PP", ("PREP", "NP")), ("VP", ("VERB", "PP")),
    ("DET", ("DET",)), ("VERB", ("VERB",)), ("NOUN", ("NOUN",)), ("PREP", ("PREP",)),
]

LEXICON = [(t, (t,)) for t in TAGS] + [
    # unit rules NP -> NOUN|PROPN|PRON|NUM and VP -> VERB, folded into the lexicon (CNF)
    ("NP", ("NOUN",)), ("NP", ("PROPN",)), ("NP", ("PRON",)), ("NP", ("NUM",)),
    ("VP", ("VERB",)),
]

G_GENERAL = LEXICON + [
    ("S", ("NP", "VP")),
    ("S", ("S", "PUNCT")),     # final punctuation
    ("NP", ("DET", "NOUN")),   # determiner + noun
    ("NP", ("NOUN", "NP")),    # noun compound (right-branching)
    ("NP", ("NP", "PP")),      # PP attached to a noun phrase
    ("PP", ("ADP", "NP")),
    ("VP", ("VERB", "NP")),    # transitive verb
    ("VP", ("VP", "PP")),      # PP attached to a verb phrase
]

G_FULL = G_GENERAL + [
    ("NP", ("ADJ", "NP")),     # adjective modifier
    ("NP", ("PROPN", "NP")),   # proper-noun modifier (Arabidopsis thaliana seedlings, PSN J...)
    ("NP", ("NUM", "NP")),     # numeral modifier (19.33 UT)
    ("NP", ("NP", "VP")),      # reduced relative (a spectrogram obtained on ...)
    ("NP", ("NP", "S")),       # relative clause attached to a noun phrase
    ("VP", ("AUX", "VERB")),   # passive / periphrastic verb (are grown)
    ("VP", ("AUX", "NP")),     # copula (is a type-Ia)
    ("VP", ("VP", "S")),       # clausal complement or adverbial clause
    ("S", ("SCONJ", "S")),     # subordinate clause (stands for SBAR, not allowed)
    ("S", ("PUNCT", "S")),     # clause introduced by a comma
    ("S", ("S", "S")),         # clause attached to a clause (sentential relative)
]

# Imperative reading ("Time flies like an arrow" = measure the flies...): unit rule S -> VP,
# which CNF turns into one S copy of every binary VP rule.
def with_imperative(grammar):
    return grammar + [("S", rhs) for lhs, rhs in grammar if lhs == "VP" and len(rhs) == 2]


G_GENERAL_IMP = with_imperative(G_GENERAL)


def bracket(node):
    """Node -> '(S (NP (DET The) (NOUN cat)) ...)' ; lexical nodes print as '(TYPE word)'."""
    if node.word is not None:
        return f"({node.type} {node.word})"
    return f"({node.type} {bracket(node.left)} {bracket(node.right)})"


def count_parses(grammar, tags):
    """Same recurrence as cyk.CYK but counting trees instead of building them (guards against blow-up)."""
    n = len(tags)
    cell = {}
    for i, options in enumerate(tags):
        c = {}
        for t in options:
            for p, rhs in grammar:
                if len(rhs) == 1 and t in rhs:
                    c[p] = c.get(p, 0) + 1
        cell[(i, i + 1)] = c
    binary = [(p, rhs) for p, rhs in grammar if len(rhs) == 2]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            c = {}
            for k in range(i + 1, i + length):
                left, right = cell[(i, k)], cell[(k, i + length)]
                for p, (l, r) in binary:
                    if l in left and r in right:
                        c[p] = c.get(p, 0) + left[l] * right[r]
            cell[(i, i + length)] = c
    return cell[(0, n)], sum(sum(c.values()) for c in cell.values())


def parse(grammar, tags, words, limit=200_000):
    """Run the provided CYK. Returns (count of full-span trees per root label, S-rooted bracketed trees).
    Trees are only built when the whole chart holds fewer than `limit` nodes."""
    counts, chart_size = count_parses(grammar, tags)
    if chart_size > limit:
        return counts, None
    roots = CYK(grammar)(tags, words)
    return counts, [bracket(t) for t in roots if t.type == "S"]
