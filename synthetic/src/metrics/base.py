import numpy as np
from torch import nn


class MetricPerSample(nn.Module):
    def compute_per_sample(self, preds, targets):
        all_scores = []
        for i, (pr, tr) in enumerate(zip(preds, targets)):
            all_scores += [self.metric.compute(
                predictions=[pr],
                references=[tr]
            )[self.key_to_return]]

        score = np.mean(all_scores).item()
        return score, all_scores
