"""
test_api.py
-----------
Minimal smoke tests across all 4 topics. Run with: `python -m pytest tests/`
from the project root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200


def test_topics_list_has_4():
    r = client.get("/api/topics")
    assert r.status_code == 200
    assert len(r.json()) == 4


# --- Topic 1: Naive Bayes ---

def test_naive_bayes_classifies_obvious_spam():
    r = client.post("/api/naive-bayes/classify", json={"text": "win a free prize click now"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_label"] == "spam"
    assert body["prob_spam"] > 0.5


def test_naive_bayes_classifies_obvious_ham():
    r = client.post("/api/naive-bayes/classify", json={"text": "can we meet for lunch tomorrow"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_label"] == "ham"


# --- Topic 2: Model Evaluation ---

def test_confusion_matrix_precision_recall_tradeoff():
    """Raising the threshold should never decrease precision or increase
    recall on the same dataset -- a fundamental property of the tradeoff."""
    low = client.post("/api/eval/confusion-matrix", json={"threshold": 0.2, "cost_fp": 1, "cost_fn": 1}).json()
    high = client.post("/api/eval/confusion-matrix", json={"threshold": 0.8, "cost_fp": 1, "cost_fn": 1}).json()
    assert high["precision"] >= low["precision"]
    assert high["recall"] <= low["recall"]


def test_roc_curve_auc_reasonable():
    r = client.get("/api/eval/roc-curve")
    assert r.status_code == 200
    body = r.json()
    assert 0.5 < body["auc"] <= 1.0  # dataset is constructed to be separable, so should beat chance


# --- Topic 3: Calculus & Gradient Descent ---

def test_gradient_descent_converges_with_reasonable_learning_rate():
    r = client.post("/api/calculus/gradient-descent", json={
        "function_id": "quadratic_bowl", "start_x": -5, "learning_rate": 0.1, "n_steps": 40,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["converged"] is True
    assert abs(body["final_x"] - 3.0) < 0.1  # true minimum is at x=3


def test_gradient_descent_diverges_with_large_learning_rate():
    r = client.post("/api/calculus/gradient-descent", json={
        "function_id": "quadratic_bowl", "start_x": -5, "learning_rate": 1.5, "n_steps": 30,
    })
    assert r.status_code == 200
    assert r.json()["diverged"] is True


def test_gradient_descent_unknown_function_rejected():
    r = client.post("/api/calculus/gradient-descent", json={
        "function_id": "not_a_real_function", "start_x": 0, "learning_rate": 0.1,
    })
    assert r.status_code == 400


# --- Topic 4: Chain Rule & Backprop ---

def test_backprop_gradients_match_numerical_check():
    payload = {"x1": 1.0, "x2": 0.5, "w1": 0.6, "w2": -0.3, "w3": 0.9, "target": 1.0}
    r = client.post("/api/backprop/gradient-check", json=payload)
    assert r.status_code == 200
    assert r.json()["max_abs_difference"] < 0.001


# --- Quizzes (all 4 topics) ---

def test_every_topic_has_a_gradeable_quiz():
    topics = client.get("/api/topics").json()
    for t in topics:
        quiz = client.get(f"/api/quiz/{t['id']}").json()
        assert len(quiz) >= 3
        # options should be present but correct_index should NOT leak to the client
        for q in quiz:
            assert "options" in q
            assert "correct_index" not in q

        # grade with all-zero answers, should still return a valid score structure
        answers = {q["id"]: 0 for q in quiz}
        grade = client.post(f"/api/quiz/{t['id']}/grade", json={"answers": answers}).json()
        assert grade["total"] == len(quiz)
        assert 0 <= grade["score"] <= grade["total"]


def test_every_topic_has_interview_questions():
    topics = client.get("/api/topics").json()
    for t in topics:
        r = client.get(f"/api/interview-questions/{t['id']}")
        assert r.status_code == 200
        assert len(r.json()) >= 2


def test_unknown_topic_quiz_404():
    r = client.get("/api/quiz/not_a_real_topic")
    assert r.status_code == 404
