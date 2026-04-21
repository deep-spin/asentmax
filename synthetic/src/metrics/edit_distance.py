import numpy as np
import editdistance

from .base import MetricPerSample


class TokenBasedEditDistance(MetricPerSample):
    def __init__(self, split=True):
        super().__init__()
        self.split = split

    def __call__(self, preds, targets, score_per_sample=False):
        all_scores = []
        for i, (pr, tr) in enumerate(zip(preds, targets)):
            if self.split:
                pr, tr = pr.split(), tr.split()
            all_scores += [editdistance.eval(pr, tr)]

        score = np.mean(all_scores).item()

        if score_per_sample:
            return score, all_scores
        else:
            return score
