import numpy as np
from torchmetrics.text import SacreBLEUScore


class WrapBleu(SacreBLEUScore):
    def __call__(self, preds, targets, score_per_sample=False):
        if score_per_sample:
            all_scores = []
            for i, (pr ,tr) in enumerate(zip(preds, targets)):
                score = self._batch_calc(super().__call__, [pr], [tr]).item()
                all_scores.append(score)
            score = np.mean(all_scores).item()
            return score, all_scores
        else:
            return self._batch_calc(super().__call__, preds, targets)

    def _batch_calc(self, uni_func, preds, targets):
        return uni_func(preds, [[t] for t in targets])
