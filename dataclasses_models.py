"""State containers for the training/testing pipeline.

These are plain data bundles so the functions in main_file.py can take a few
objects instead of ~30 loose scalars. The only behavior lives in
GeneralizationTracker.update, which needs get_avgErr_SPlitEpochs from utils.
"""
from dataclasses import dataclass, field

from utils.stop_early import get_avgErr_SPlitEpochs


@dataclass
class ModelBundle:
    model: object
    optimizer: object
    scheduler: object


@dataclass
class Loaders:
    train: object
    test: object


@dataclass
class Paths:
    # Field order matches the return order of inialize_log_file(args).
    timestr: str
    losses_file: str
    gl_file: str
    log: object
    root: str
    checkpoints: str
    logs: str
    test_results: str
    valid_results: str


@dataclass
class MetricsHistory:
    losses_all_stages: dict = field(default_factory=dict)
    losses_sum_all_stages: dict = field(default_factory=dict)
    epes_all_stages: dict = field(default_factory=dict)
    epes_sum_all_stages: dict = field(default_factory=dict)
    test_losses: dict = field(default_factory=dict)
    test_epes: dict = field(default_factory=dict)
    test_sum_stages_losses: dict = field(default_factory=dict)
    test_outliers_1: dict = field(default_factory=dict)
    test_outliers_2: dict = field(default_factory=dict)
    test_outliers_3: dict = field(default_factory=dict)


@dataclass
class GeneralizationTracker:
    split_epochs: int
    start_averaging: int = 1
    end_averaging: int = 0
    sum_err_tr: float = 0.0
    sum_err_val: float = 0.0
    temp_min_tr: float = float('inf')
    temp_min_val: float = float('inf')
    avg_tr: list = field(default_factory=list)
    min_tr: list = field(default_factory=list)
    avg_val: list = field(default_factory=list)
    min_val: list = field(default_factory=list)
    gl_tr: list = field(default_factory=list)
    gl_val: list = field(default_factory=list)

    def __post_init__(self):
        if self.end_averaging == 0:
            self.end_averaging = self.split_epochs

    def update(self, train_sum_loss, val_sum_loss, epoch):
        """Accumulate per-epoch sums; on a split boundary compute the split-epoch
        averages/minima/generalization losses. Returns True on a split boundary."""
        self.sum_err_tr += train_sum_loss
        self.sum_err_val += val_sum_loss
        if train_sum_loss < self.temp_min_tr:
            self.temp_min_tr = train_sum_loss
        if val_sum_loss < self.temp_min_val:
            self.temp_min_val = val_sum_loss

        if epoch % self.split_epochs == 0:
            (self.avg_tr, self.min_tr, self.gl_tr,
             self.temp_min_tr, self.sum_err_tr) = get_avgErr_SPlitEpochs(
                self.avg_tr, self.min_tr, self.gl_tr,
                self.sum_err_tr, self.split_epochs, self.temp_min_tr)
            (self.avg_val, self.min_val, self.gl_val,
             self.temp_min_val, self.sum_err_val) = get_avgErr_SPlitEpochs(
                self.avg_val, self.min_val, self.gl_val,
                self.sum_err_val, self.split_epochs, self.temp_min_val)
            self.start_averaging = self.end_averaging + 1
            self.end_averaging *= 2
            return True
        return False


@dataclass
class EarlyStopping:
    overfit_checker: object
    steadystate_checker: object
    best_checkpoints: object
    n_stop_epochs: int
    threshold_epochs: int
    no_improvement_epochs: int = 0
    best_error: float = float('inf')
    stopped_paths: list = field(default_factory=list)
    stopped_epochs: list = field(default_factory=list)
