"""Explicit ground-truth identity smoke for loader/evaluator wiring only."""


def predict(sample):
    return sample["target"]
