from torch.optim.lr_scheduler import LinearLR, SequentialLR, PolynomialLR


class PolynomialWithWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_iters, power=0.5, last_epoch=-1):
        warmup = LinearLR(
            optimizer,
            start_factor=0.00001,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        decay = PolynomialLR(
            optimizer,
            total_iters=total_iters,
            power=power,
            last_epoch=last_epoch
        )
        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps]
        )

    def step(self):
        self.scheduler.step()