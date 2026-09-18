"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This project is a teaching tool, not a single production model, so it
uses two small, hand-curated datasets rather than one large Kaggle
dataset -- appropriate for a "deep intuition + live simulation" goal
where a learner benefits from being able to see and reason about every
row, not just aggregate statistics over thousands.

1. spam_ham_synthetic.csv -- a small labeled text dataset for the
   Naive Bayes live classifier demo (topic 1). Short, readable messages
   so a learner can see exactly which words push the classifier toward
   "spam" or "ham."

2. eval_test_set_synthetic.csv -- a small set of (true_label,
   predicted_probability) pairs for the model-evaluation demo (topic 2):
   confusion matrix, type I/II errors, ROC-AUC, precision/recall
   tradeoff. Probabilities are NOT from a real trained model -- they are
   directly constructed so that positives skew toward high probability
   and negatives skew toward low probability, WITH deliberate overlap in
   the middle, so that moving the decision threshold visibly trades
   precision against recall (the whole point of the demo).

Both are synthetic/curated because this build environment has no
internet access to Kaggle, and because hand-curated data suits a
teaching tool better than an anonymized real dataset would anyway.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
OUT_DIR = Path(__file__).parent

# --- 1. Naive Bayes spam/ham dataset -----------------------------------
SPAM_MESSAGES = [
    "win a free prize now click here",
    "claim your free cash prize today",
    "urgent your account has been suspended click now",
    "you have won a lottery claim your money now",
    "limited time offer buy now save big",
    "free entry to win a brand new car",
    "congratulations you are selected for a free gift",
    "act now to claim your discount before it expires",
    "get rich quick with this one simple trick",
    "your loan has been approved click to claim cash",
    "hot singles in your area click now",
    "free vacation click here to claim your trip",
    "exclusive deal buy now and save fifty percent",
    "you are our lucky winner claim your reward",
    "double your money in just one week guaranteed",
]
HAM_MESSAGES = [
    "hey are we still meeting for lunch tomorrow",
    "can you send me the report before the meeting",
    "happy birthday hope you have a great day",
    "reminder the dentist appointment is at three",
    "thanks for helping me move this weekend",
    "let's catch up over coffee next week",
    "the project deadline has been moved to friday",
    "did you watch the game last night",
    "please review the attached document and reply",
    "the flight was delayed but we landed safely",
    "can you pick up milk on your way home",
    "great job on the presentation today",
    "let me know if you need help with the assignment",
    "the weather looks nice for the hike this weekend",
    "our meeting is rescheduled to monday morning",
]


def generate_naive_bayes_data() -> pd.DataFrame:
    rows = [{"message": m, "label": "spam"} for m in SPAM_MESSAGES]
    rows += [{"message": m, "label": "ham"} for m in HAM_MESSAGES]
    return pd.DataFrame(rows)


# --- 2. Model evaluation demo dataset -----------------------------------
def generate_eval_data(n_positive: int = 60, n_negative: int = 90, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Positives: probabilities skew high, but with a long lower tail (hard cases)
    pos_probs = rng.beta(a=5, b=2, size=n_positive)
    # Negatives: probabilities skew low, but with an upper tail (hard cases) --
    # this overlap with positives is deliberate: it's what makes threshold choice matter.
    neg_probs = rng.beta(a=2, b=5, size=n_negative)

    df = pd.DataFrame({
        "predicted_probability": np.concatenate([pos_probs, neg_probs]).round(4),
        "true_label": [1] * n_positive + [0] * n_negative,
    })
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "sample_id", [f"s{i:03d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    nb_df = generate_naive_bayes_data()
    nb_df.to_csv(OUT_DIR / "spam_ham_synthetic.csv", index=False)
    print(f"Wrote {len(nb_df)} labeled messages to spam_ham_synthetic.csv "
          f"({(nb_df['label'] == 'spam').sum()} spam, {(nb_df['label'] == 'ham').sum()} ham)")

    eval_df = generate_eval_data()
    eval_df.to_csv(OUT_DIR / "eval_test_set_synthetic.csv", index=False)
    print(f"Wrote {len(eval_df)} labeled probability samples to eval_test_set_synthetic.csv "
          f"({(eval_df['true_label'] == 1).sum()} positive, {(eval_df['true_label'] == 0).sum()} negative)")
