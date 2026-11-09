"""A small feed-forward network with inspectable, full-batch gradient descent."""

import numpy as np

ACTIVATIONS = ("Tanh", "ReLU", "Leaky ReLU", "Sigmoid", "Linear")
CLASSIFICATION_LOSSES = ("Binary cross-entropy", "Mean squared error", "Mean absolute error")
REGRESSION_LOSSES = ("Mean squared error", "Mean absolute error", "Huber")
MAX_ITERATIONS = 4000


def activate(name, z):
    if name == "Sigmoid":
        return np.exp(-np.logaddexp(0, -z))
    if name == "Tanh":
        return np.tanh(z)
    if name == "ReLU":
        return np.maximum(0, z)
    if name == "Leaky ReLU":
        return np.where(z >= 0, z, 0.05 * z)
    return z


def activation_slope(name, z, a):
    if name == "Sigmoid":
        return a * (1 - a)
    if name == "Tanh":
        return 1 - a**2
    if name == "ReLU":
        return z > 0
    if name == "Leaky ReLU":
        return np.where(z >= 0, 1.0, 0.05)
    return np.ones_like(z)


def cost_and_gradient(name, z, prediction, target):
    """Return mean cost and its per-row derivative before any output activation."""
    error = prediction - target
    if name == "Binary cross-entropy":
        return np.mean(np.logaddexp(0, (1 - 2 * target) * z)), error, True
    if name == "Mean squared error":
        return np.mean(error**2), 2 * error, False
    if name == "Mean absolute error":
        return np.mean(np.abs(error)), np.sign(error), False
    absolute = np.abs(error)
    return np.mean(np.where(absolute <= 1, error**2 / 2, absolute - 0.5)), np.clip(error, -1, 1), False


class Network:
    def __init__(self, sizes, hidden_activations, output_activation, loss, seed, case):
        rng = np.random.default_rng(seed)
        self.mask_rng = np.random.default_rng(seed + 1)
        self.activations = (*hidden_activations, output_activation)
        self.loss = loss
        self.weights = [
            rng.normal(0, np.sqrt(2 / (fan_in + fan_out)), (fan_in, fan_out))
            for fan_in, fan_out in zip(sizes, sizes[1:])
        ]
        self.biases = [np.zeros(fan_out) for fan_out in sizes[1:]]
        self.x, self.y = case.train_x, case.train_y
        self.eval_x, self.eval_y = case.eval_x, case.eval_y
        self.history = []
        self.record([np.ones_like(w, dtype=bool) for w in self.weights], 0)

    @property
    def iteration(self):
        return len(self.history) - 1

    def forward(self, x, weights=None, biases=None):
        weights = self.weights if weights is None else weights
        biases = self.biases if biases is None else biases
        activations, sums = [x], []
        for w, b, name in zip(weights, biases, self.activations):
            z = activations[-1] @ w + b
            sums.append(z)
            activations.append(activate(name, z))
        return activations, sums

    def evaluate(self, x, y):
        activations, sums = self.forward(x)
        return float(cost_and_gradient(self.loss, sums[-1], activations[-1], y)[0])

    def record(self, masks, drop_rate):
        self.history.append({
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "masks": masks,
            "drop_rate": drop_rate,
            "training_cost": self.evaluate(self.x, self.y),
            "evaluation_cost": self.evaluate(self.eval_x, self.eval_y),
        })

    def step(self, learning_rate, drop_rate):
        keep = 1 - drop_rate
        masks = [self.mask_rng.random(w.shape) < keep for w in self.weights]
        scales = [mask / keep for mask in masks]
        effective = [w * scale for w, scale in zip(self.weights, scales)]
        activations, sums = self.forward(self.x, effective)
        _, delta, is_logit_gradient = cost_and_gradient(self.loss, sums[-1], activations[-1], self.y)
        delta = delta / len(self.x)
        next_weights, next_biases = self.weights.copy(), self.biases.copy()
        for layer in reversed(range(len(self.weights))):
            if layer != len(self.weights) - 1 or not is_logit_gradient:
                delta = delta * activation_slope(self.activations[layer], sums[layer], activations[layer + 1])
            gradient = (activations[layer].T @ delta) * scales[layer]
            next_weights[layer] = self.weights[layer] - learning_rate * gradient
            next_biases[layer] = self.biases[layer] - learning_rate * delta.sum(axis=0)
            delta = delta @ effective[layer].T
        previous_weights, previous_biases = self.weights, self.biases
        self.weights, self.biases = next_weights, next_biases
        with np.errstate(over="ignore", invalid="ignore"):
            self.record(masks, drop_rate)
        if not all(np.isfinite(self.history[-1][name]) for name in ("training_cost", "evaluation_cost")):
            self.history.pop()
            self.weights, self.biases = previous_weights, previous_biases
            return False
        return True


def train(model, count, rate, drop_rate, tolerance=None, progress=None):
    """Return the stopping reason. Patience is measured on unmasked training cost."""
    reference = model.history[-1]["training_cost"]
    stale = 0
    budget = min(count, MAX_ITERATIONS - model.iteration)
    for index in range(1, budget + 1):
        if not model.step(rate, drop_rate):
            return "unstable", index - 1
        if progress and (index % 25 == 0 or index == budget):
            progress(index / budget)
        if tolerance is not None:
            cost = model.history[-1]["training_cost"]
            if cost < reference - tolerance:
                reference, stale = cost, 0
            else:
                stale += 1
            if stale >= 100:
                return "plateau", index
    return "cap" if model.iteration == MAX_ITERATIONS else "budget", budget
