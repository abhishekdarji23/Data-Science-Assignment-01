"""
backprop.py
-----------
Topic 4: the chain rule and how it connects to backpropagation.

Implements the forward and backward pass of the smallest network that
still has a genuine hidden layer -- 2 inputs -> 1 hidden neuron (sigmoid)
-> 1 output neuron (sigmoid) -- entirely by hand (no autograd), so every
gradient returned is a literal, inspectable application of the chain
rule: dL/dw1 = dL/dyhat * dyhat/dz2 * dz2/da1 * da1/dz1 * dz1/dw1, computed
term by term. This IS backpropagation; a deep network just chains more
of these same local derivatives together.
"""
import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_prime(sig_x: float) -> float:
    """Derivative of sigmoid, expressed in terms of sigmoid(x) itself
    (the standard, numerically convenient form: sigma'(x) = sigma(x)(1-sigma(x)))."""
    return sig_x * (1 - sig_x)


def forward_backward(x1: float, x2: float, w1: float, w2: float, w3: float, target: float) -> dict:
    """
    Architecture:
        z1 = w1*x1 + w2*x2      (hidden pre-activation)
        a1 = sigmoid(z1)        (hidden activation)
        z2 = w3*a1              (output pre-activation)
        yhat = sigmoid(z2)      (output / prediction)
        L = 0.5*(yhat - target)^2   (squared error loss)

    Backward pass (chain rule, term by term):
        dL/dyhat = (yhat - target)
        dyhat/dz2 = sigmoid_prime(yhat)
        dz2/da1 = w3
        da1/dz1 = sigmoid_prime(a1)
        dz1/dw1 = x1
        dz1/dw2 = x2
        dz2/dw3 = a1

        dL/dw3 = dL/dyhat * dyhat/dz2 * dz2/dw3
        dL/dw1 = dL/dyhat * dyhat/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
        dL/dw2 = dL/dyhat * dyhat/dz2 * dz2/da1 * da1/dz1 * dz1/dw2
    """
    # --- forward pass ---
    z1 = w1 * x1 + w2 * x2
    a1 = sigmoid(z1)
    z2 = w3 * a1
    yhat = sigmoid(z2)
    loss = 0.5 * (yhat - target) ** 2

    # --- backward pass: the chain rule, one local derivative at a time ---
    dL_dyhat = (yhat - target)
    dyhat_dz2 = sigmoid_prime(yhat)
    dz2_da1 = w3
    da1_dz1 = sigmoid_prime(a1)
    dz1_dw1 = x1
    dz1_dw2 = x2
    dz2_dw3 = a1

    dL_dz2 = dL_dyhat * dyhat_dz2
    dL_dw3 = dL_dz2 * dz2_dw3
    dL_da1 = dL_dz2 * dz2_da1
    dL_dz1 = dL_da1 * da1_dz1
    dL_dw1 = dL_dz1 * dz1_dw1
    dL_dw2 = dL_dz1 * dz1_dw2

    return {
        "forward": {
            "z1 (= w1*x1 + w2*x2)": round(z1, 5),
            "a1 (= sigmoid(z1))": round(a1, 5),
            "z2 (= w3*a1)": round(z2, 5),
            "yhat (= sigmoid(z2))": round(yhat, 5),
            "loss (= 0.5*(yhat-target)^2)": round(loss, 5),
        },
        "backward_chain": {
            "dL/dyhat": round(dL_dyhat, 5),
            "dyhat/dz2": round(dyhat_dz2, 5),
            "dz2/da1 (= w3)": round(dz2_da1, 5),
            "da1/dz1": round(da1_dz1, 5),
            "dz1/dw1 (= x1)": round(dz1_dw1, 5),
            "dz1/dw2 (= x2)": round(dz1_dw2, 5),
            "dz2/dw3 (= a1)": round(dz2_dw3, 5),
        },
        "gradients": {
            "dL/dw1": round(dL_dw1, 5),
            "dL/dw2": round(dL_dw2, 5),
            "dL/dw3": round(dL_dw3, 5),
        },
    }


def numerical_gradient_check(x1, x2, w1, w2, w3, target, epsilon: float = 1e-5) -> dict:
    """Verifies the analytic (chain-rule) gradients above against numerical
    (finite-difference) gradients -- the standard way anyone implementing
    backprop by hand checks their work. Included as a live 'trust but
    verify' demo: the two should agree to ~5-6 decimal places."""
    def loss_of(w1_, w2_, w3_):
        z1 = w1_ * x1 + w2_ * x2
        a1 = sigmoid(z1)
        z2 = w3_ * a1
        yhat = sigmoid(z2)
        return 0.5 * (yhat - target) ** 2

    num_dw1 = (loss_of(w1 + epsilon, w2, w3) - loss_of(w1 - epsilon, w2, w3)) / (2 * epsilon)
    num_dw2 = (loss_of(w1, w2 + epsilon, w3) - loss_of(w1, w2 - epsilon, w3)) / (2 * epsilon)
    num_dw3 = (loss_of(w1, w2, w3 + epsilon) - loss_of(w1, w2, w3 - epsilon)) / (2 * epsilon)

    analytic = forward_backward(x1, x2, w1, w2, w3, target)["gradients"]

    return {
        "analytic": analytic,
        "numerical": {
            "dL/dw1": round(num_dw1, 5),
            "dL/dw2": round(num_dw2, 5),
            "dL/dw3": round(num_dw3, 5),
        },
        "max_abs_difference": round(max(
            abs(analytic["dL/dw1"] - num_dw1),
            abs(analytic["dL/dw2"] - num_dw2),
            abs(analytic["dL/dw3"] - num_dw3),
        ), 6),
    }
