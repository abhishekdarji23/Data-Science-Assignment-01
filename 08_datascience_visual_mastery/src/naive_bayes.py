"""
naive_bayes.py
--------------
Topic 1: Naive Bayes — deep intuition + live simulation.

Trains a REAL sklearn MultinomialNB on the curated spam/ham dataset, and
exposes classify(text) which returns not just the predicted label but
also each word's individual contribution to the decision -- this is the
"deep intuition" piece: Naive Bayes multiplies per-word likelihood
ratios together (it's "naive" because it assumes word occurrences are
independent given the class), so showing which words pushed the decision
which way IS an explanation of how the algorithm actually works, not a
separate explainability bolt-on.
"""
from pathlib import Path
import math

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spam_ham_synthetic.csv"

_vectorizer = None
_model = None
_classes = None


def _load():
    global _vectorizer, _model, _classes
    if _model is None:
        df = pd.read_csv(DATA_PATH)
        _vectorizer = CountVectorizer()
        X = _vectorizer.fit_transform(df["message"])
        _model = MultinomialNB()
        _model.fit(X, df["label"])
        _classes = list(_model.classes_)
    return _vectorizer, _model, _classes


def get_training_examples() -> dict:
    df = pd.read_csv(DATA_PATH)
    return {
        "spam": df[df["label"] == "spam"]["message"].tolist(),
        "ham": df[df["label"] == "ham"]["message"].tolist(),
    }


def classify(text: str) -> dict:
    vectorizer, model, classes = _load()
    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")

    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    predicted = classes[int(proba.argmax())]

    # Per-word contribution: log P(word | spam) - log P(word | ham).
    # Positive => pushes toward spam, negative => pushes toward ham.
    # This is exactly the quantity Naive Bayes sums (in log-space) across
    # all words in the message to reach its decision -- not an approximation.
    feature_names = vectorizer.get_feature_names_out()
    log_prob_spam = model.feature_log_prob_[spam_idx]
    log_prob_ham = model.feature_log_prob_[ham_idx]
    word_to_idx = {w: i for i, w in enumerate(feature_names)}

    tokens = vectorizer.build_analyzer()(text)
    seen = set()
    word_contributions = []
    for token in tokens:
        if token in seen or token not in word_to_idx:
            continue
        seen.add(token)
        idx = word_to_idx[token]
        contribution = round(float(log_prob_spam[idx] - log_prob_ham[idx]), 3)
        word_contributions.append({"word": token, "spam_lean_score": contribution})

    word_contributions.sort(key=lambda w: abs(w["spam_lean_score"]), reverse=True)

    return {
        "predicted_label": predicted,
        "prob_spam": round(float(proba[spam_idx]), 4),
        "prob_ham": round(float(proba[ham_idx]), 4),
        "word_contributions": word_contributions,
        "unknown_words": [t for t in tokens if t not in word_to_idx and t not in seen],
    }
