"""
calculus.py
-----------
Topic 3: differential calculus, derivatives, and gradient descent.

Uses REAL, hand-derived analytic derivatives (not numerical/automatic
differentiation) for a small set of named functions, so the "derivative"
shown at each step is exactly what a calculus student would compute by
hand -- and gradient_descent() is the direct, literal algorithm
(x_new = x_old - learning_rate * f'(x_old)) run step by step, which is
the live simulation: watching x walk downhill toward a minimum, or
diverge if the learning rate is too large (a deliberately included
failure mode -- gradient descent isn't magic, and seeing it fail for a
too-large learning rate is part of the intuition).
"""
import math

# Each function: (f, f_prime, description, true minimum x for reference)
FUNCTIONS = {
    "quadratic_bowl": {
        "label": "f(x) = (x - 3)^2 + 2",
        "f": lambda x: (x - 3) ** 2 + 2,
        "f_prime": lambda x: 2 * (x - 3),
        "true_minimum_x": 3.0,
    },
    "quartic_with_flat_region": {
        "label": "f(x) = 0.1x^4 - x^2 + 3 (has a flat plateau near x=0)",
        "f": lambda x: 0.1 * x ** 4 - x ** 2 + 3,
        "f_prime": lambda x: 0.4 * x ** 3 - 2 * x,
        "true_minimum_x": math.sqrt(5),  # one of two symmetric global minima
    },
    "steep_asymmetric": {
        "label": "f(x) = e^(0.5x) + (x + 2)^2",
        "f": lambda x: math.exp(0.5 * x) + (x + 2) ** 2,
        "f_prime": lambda x: 0.5 * math.exp(0.5 * x) + 2 * (x + 2),
        "true_minimum_x": None,  # no closed form; found numerically for reference only
    },
}


def list_functions() -> list[dict]:
    return [{"id": k, "label": v["label"]} for k, v in FUNCTIONS.items()]


def gradient_descent(function_id: str, start_x: float, learning_rate: float, n_steps: int = 40) -> dict:
    if function_id not in FUNCTIONS:
        raise ValueError(f"Unknown function_id '{function_id}'. Options: {list(FUNCTIONS.keys())}")
    spec = FUNCTIONS[function_id]
    f, f_prime = spec["f"], spec["f_prime"]

    trajectory = []
    x = start_x
    diverged = False
    for step in range(n_steps + 1):
        try:
            fx = f(x)
            grad = f_prime(x)
        except OverflowError:
            diverged = True
            break
        trajectory.append({"step": step, "x": round(x, 5), "f_x": round(fx, 5), "gradient": round(grad, 5)})
        if abs(x) > 1e6 or abs(fx) > 1e12:
            diverged = True
            break
        x = x - learning_rate * grad  # the actual gradient descent update rule

    converged = (not diverged) and len(trajectory) >= 2 and \
        abs(trajectory[-1]["gradient"]) < 0.01

    return {
        "function_id": function_id,
        "function_label": spec["label"],
        "learning_rate": learning_rate,
        "trajectory": trajectory,
        "diverged": diverged,
        "converged": converged,
        "final_x": trajectory[-1]["x"] if trajectory else None,
    }
