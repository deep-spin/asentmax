import evaluate

from .base import MetricPerSample


class TextExactMatch(MetricPerSample):
    def __init__(self):
        super().__init__()
        self.metric = evaluate.load("exact_match", keep_in_memory=True) # keep_in_memory=True to disable caching
        self.key_to_return = "exact_match"

    def __call__(self, preds, targets, score_per_sample=False):
        if score_per_sample:
            return self.compute_per_sample(preds, targets)
        else:
            score = self.metric.compute(
                predictions=preds,
                references=targets
            )

            return score["exact_match"]
