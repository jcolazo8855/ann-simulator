"""Repeatable synthetic classroom examples with fixed evaluation rows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Feature:
    name: str
    low: float
    high: float
    step: float


@dataclass(frozen=True)
class Lesson:
    icon: str
    description: str
    prompt: str
    inputs: int
    nodes: int
    rate: float


LESSONS = {
    "Customer churn": Lesson(
        "↗", "Predict whether a subscriber will leave from their account and support activity.",
        "Try a subscriber with frequent support tickets and few recent logins. How does the estimated churn probability change?",
        4, 6, 0.15,
    ),
    "Sales prediction": Lesson(
        "▥", "Estimate weekly sales from a store’s traffic, marketing, and promotions.",
        "Increase ad spend for two stores with different levels of foot traffic. Does the predicted lift look the same?",
        4, 6, 0.06,
    ),
    "XOR challenge": Lesson(
        "⊕", "Predict 1 when exactly one switch is on. Each switch alone gives you no signal.",
        "Train with Linear in every hidden layer. Then reset with Tanh and compare the four predictions.",
        2, 4, 0.65,
    ),
}

CHURN = (
    Feature("Support tickets", 0, 10, 1), Feature("Monthly logins", 0, 50, 1),
    Feature("Tenure (months)", 1, 48, 1), Feature("Monthly bill ($)", 20, 120, 1),
    Feature("Late payments", 0, 6, 1), Feature("Days since login", 0, 60, 1),
    Feature("Plan add-ons", 0, 5, 1), Feature("Discount (%)", 0, 30, 1),
    Feature("Satisfaction (1–10)", 1, 10, 1), Feature("Devices", 1, 6, 1),
)
SALES = (
    Feature("Foot traffic", 100, 1400, 10), Feature("Ad spend ($)", 0, 5000, 50),
    Feature("Discount (%)", 0, 30, 1), Feature("Local event", 0, 1, 1),
    Feature("Store area (sq ft)", 800, 5500, 50), Feature("Staff hours", 60, 240, 5),
    Feature("Online visits", 200, 3000, 10), Feature("Competitors nearby", 0, 8, 1),
    Feature("Rainfall (in)", 0, 4, 0.1), Feature("Prior visits", 100, 1400, 10),
)
XOR = (Feature("Switch A", 0, 1, 1), Feature("Switch B", 0, 1, 1)) + tuple(
    Feature(f"Extra switch {i}", 0, 1, 1) for i in range(3, 11)
)


@dataclass
class Case:
    name: str
    features: tuple
    raw_eval: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    eval_x: np.ndarray
    eval_y: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    target_center: float
    target_scale: float

    @property
    def regression(self):
        return self.name == "Sales prediction"

    @property
    def evaluation_name(self):
        return "Truth table" if self.name == "XOR challenge" else "Holdout"

    def encode(self, raw):
        return (raw - self.center) / self.scale

    def decode_target(self, values):
        return values * self.target_scale + self.target_center


def make_case(name, inputs):
    rng = np.random.default_rng(417)
    if name == "XOR challenge":
        features = XOR
        truth = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        switches = np.tile(truth, (64, 1))
        train_raw = np.column_stack((switches, rng.integers(0, 2, (len(switches), 8))))
        eval_raw = np.column_stack((truth, np.zeros((4, 8))))
        train_y = np.logical_xor(switches[:, 0], switches[:, 1]).astype(float)
        eval_y = np.array([0, 1, 1, 0], dtype=float)
    else:
        features = CHURN if name == "Customer churn" else SALES
        columns = []
        for f in features:
            ticks = round((f.high - f.low) / f.step)
            columns.append(f.low + f.step * rng.integers(0, ticks + 1, 320))
        raw = np.column_stack(columns)
        scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0)
        a, b, c, d = scaled[:, :4].T
        if name == "Customer churn":
            score = 1.65 * a - 1.55 * b - 0.75 * c + 0.38 * d - 0.62 * a * b
            target = (rng.random(len(raw)) < 1 / (1 + np.exp(-score))).astype(float)
        else:
            target = 22_000 + 4200 * a + 2400 * b + 1300 * c + 950 * d + 1800 * a * b
            target += rng.normal(0, 850, len(raw))
        train_raw, eval_raw = raw[:240], raw[240:]
        train_y, eval_y = target[:240], target[240:]
    train_raw, eval_raw = train_raw[:, :inputs], eval_raw[:, :inputs]
    center, scale = train_raw.mean(axis=0), train_raw.std(axis=0)
    target_center, target_scale = (float(train_y.mean()), float(train_y.std())) if name == "Sales prediction" else (0., 1.)
    return Case(
        name, features[:inputs], eval_raw, (train_raw - center) / scale,
        ((train_y - target_center) / target_scale)[:, None], (eval_raw - center) / scale,
        ((eval_y - target_center) / target_scale)[:, None], center, scale, target_center, target_scale,
    )
