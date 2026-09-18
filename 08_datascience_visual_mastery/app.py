"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend serving 4 topics, each with a live simulation, a
server-graded quiz (correct answers never sent to the browser before
grading), and interview-prep material. Serves the plain HTML/JS/CSS
frontend from ./static (no build step).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    uvicorn app:app --reload --port 8008
Then open http://127.0.0.1:8008
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from naive_bayes import classify, get_training_examples  # noqa: E402
from eval_metrics import confusion_at_threshold, roc_curve_points, dataset_summary  # noqa: E402
from calculus import gradient_descent, list_functions  # noqa: E402
from backprop import forward_backward, numerical_gradient_check  # noqa: E402
from content import TOPICS, get_quiz, grade_quiz, get_interview_questions  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Data Science Visual Foundations",
    description="CRISP-DM-adjacent teaching tool: Naive Bayes, model evaluation, gradient descent, and backprop, with live simulations and graded quizzes.",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/topics")
def topics():
    return TOPICS


# --- Topic 1: Naive Bayes -------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


@app.get("/api/naive-bayes/examples")
def naive_bayes_examples():
    return get_training_examples()


@app.post("/api/naive-bayes/classify")
def naive_bayes_classify(req: ClassifyRequest):
    return classify(req.text)


# --- Topic 2: Model Evaluation ---------------------------------------------

class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    cost_fp: float = Field(1.0, ge=0.0, le=100.0)
    cost_fn: float = Field(1.0, ge=0.0, le=100.0)


@app.get("/api/eval/dataset-summary")
def eval_dataset_summary():
    return dataset_summary()


@app.post("/api/eval/confusion-matrix")
def eval_confusion_matrix(req: ThresholdRequest):
    return confusion_at_threshold(req.threshold, req.cost_fp, req.cost_fn)


@app.get("/api/eval/roc-curve")
def eval_roc_curve():
    return roc_curve_points()


# --- Topic 3: Calculus & Gradient Descent ----------------------------------

class GradientDescentRequest(BaseModel):
    function_id: str
    start_x: float = Field(..., ge=-50, le=50)
    learning_rate: float = Field(..., gt=0, le=3.0)
    n_steps: int = Field(40, ge=1, le=200)


@app.get("/api/calculus/functions")
def calculus_functions():
    return list_functions()


@app.post("/api/calculus/gradient-descent")
def calculus_gradient_descent(req: GradientDescentRequest):
    try:
        return gradient_descent(req.function_id, req.start_x, req.learning_rate, req.n_steps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Topic 4: Chain Rule & Backprop ----------------------------------------

class BackpropRequest(BaseModel):
    x1: float = Field(..., ge=-10, le=10)
    x2: float = Field(..., ge=-10, le=10)
    w1: float = Field(..., ge=-10, le=10)
    w2: float = Field(..., ge=-10, le=10)
    w3: float = Field(..., ge=-10, le=10)
    target: float = Field(..., ge=0, le=1)


@app.post("/api/backprop/forward-backward")
def backprop_forward_backward(req: BackpropRequest):
    return forward_backward(req.x1, req.x2, req.w1, req.w2, req.w3, req.target)


@app.post("/api/backprop/gradient-check")
def backprop_gradient_check(req: BackpropRequest):
    return numerical_gradient_check(req.x1, req.x2, req.w1, req.w2, req.w3, req.target)


# --- Quizzes & interview prep (shared across all 4 topics) -----------------

class QuizGradeRequest(BaseModel):
    answers: dict[str, int]


@app.get("/api/quiz/{topic_id}")
def quiz(topic_id: str):
    questions = get_quiz(topic_id)
    if not questions:
        raise HTTPException(status_code=404, detail=f"Unknown topic_id '{topic_id}'")
    return questions


@app.post("/api/quiz/{topic_id}/grade")
def quiz_grade(topic_id: str, req: QuizGradeRequest):
    result = grade_quiz(topic_id, req.answers)
    if result["total"] == 0:
        raise HTTPException(status_code=404, detail=f"Unknown topic_id '{topic_id}'")
    return result


@app.get("/api/interview-questions/{topic_id}")
def interview_questions(topic_id: str):
    questions = get_interview_questions(topic_id)
    if not questions:
        raise HTTPException(status_code=404, detail=f"Unknown topic_id '{topic_id}'")
    return questions


# --- static frontend (plain HTML/CSS/JS, no build step) --------------------
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
