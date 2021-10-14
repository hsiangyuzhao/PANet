import os
import random
import math
import torch
import numpy as np
import pandas as pd
import torchio as tio
from torch.optim.lr_scheduler import _LRScheduler


def set_random_seed(seed=512, benchmark=False):
    """
    Set random seed for reproducibility.
    Args:
        seed: random seed.
        benchmark: Boolean value, whether to enable CUDNN benchmark to accelerate training.
            will slightly affect reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    if benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MetricMeter(object):
    """
    Metric saver，receives metrics when validation and print results
    """
    def __init__(self, metrics, class_names):
        """
        Args:
            metrics: metrics to record.
            class_names: the name of classes.
        """
        self.metrics = metrics
        self.class_names = class_names
        self.initialization()
        self.name = []

    def initialization(self):
        for metric in self.metrics:
            for class_name in self.class_names:
                exec('self.' + class_name + '_' + metric + '=[]')

    @staticmethod
    def update(metric_dict):
        for metric_key, metric_value in metric_dict.items():
            try:
                exec('self.' + metric_key + '.append(metric_value)')
            except:
                continue

    def report(self, print_stats=True):
        report_str = ''
        for metric in self.metrics:
            for class_name in self.class_names:
                metric_mean = np.nanmean(eval('self.' + class_name + '_' + metric), axis=0)
                metric_std = np.nanstd(eval('self.' + class_name + '_' + metric), axis=0)
                if print_stats:
                    stats = class_name + '_' + metric + ': {} ± {};'.format(np.around(metric_mean, decimals=4),
                                                                            np.around(metric_std, decimals=4))
                    print(stats, end=' ')
                    report_str += stats
        print('\n')
        return report_str

    def save(self, savedir='./metrics', filename='metric.csv'):
        os.makedirs(savedir, exist_ok=True)
        series = [pd.Series(self.name)]
        columns = ['name']
        for metric in self.metrics:
            for class_name in self.class_names:
                exec('series.append(pd.Series(self.' + class_name + '_' + metric + '))')
                columns.append(class_name + '_' + metric)
        df = pd.concat(series, axis=1)
        df.columns = columns
        df.to_csv(os.path.join(savedir, filename), index=False)


class CosineAnnealingWithWarmUp(_LRScheduler):
    """
            optimizer (Optimizer): Wrapped optimizer.
            first_cycle_steps (int): First cycle step size.
            cycle_mult(float): Cycle steps magnification. Default: -1.
            max_lr(float): First cycle's max learning rate. Default: 0.1.
            min_lr(float): Min learning rate. Default: 0.001.
            warmup_steps(int): Linear warmup step size. Default: 0.
            gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
            last_epoch (int): The index of last epoch. Default: -1.
        """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_steps: int,
                 max_lr_steps: int,
                 max_lr: float = 1e-2,
                 min_lr: float = 1e-8,
                 warmup_steps: int = 0,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < cycle_steps
        assert warmup_steps + max_lr_steps < cycle_steps

        self.cycle_steps = cycle_steps  # first cycle step size
        self.max_lr_steps = max_lr_steps
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size

        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWithWarmUp, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        elif self.step_in_cycle >= self.warmup_steps and self.step_in_cycle < self.warmup_steps + self.max_lr_steps:
            return [self.max_lr for _ in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * (
                    self.step_in_cycle - self.warmup_steps - self.max_lr_steps) / (
                    self.cycle_steps - self.warmup_steps - self.max_lr_steps))) / 2 for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
        else:
            self.step_in_cycle = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
