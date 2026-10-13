"""Post-warmup total-loss monitoring, independent of raw-score checkpoints."""
from collections import deque
import math

DEFAULT_WINDOW = 10
DEFAULT_MIN_DELTA = 0.001


class MovingAverageEarlyStopping:
    """Count epochs without a qualifying decrease in mean evaluation total loss.

    Only full epochs after warmup enter the window. A decrease must exceed
    min_delta, in absolute objective units, to update best_mean and reset
    patience. Raw checkpoint scores are managed separately by the caller.
    """

    def __init__(self, patience=0, window=DEFAULT_WINDOW, min_delta=DEFAULT_MIN_DELTA):
        if type(window) is not int or window < 1:
            raise ValueError('train.early_stopping_window must be a positive integer.')
        if type(patience) is not int or patience < 0:
            raise ValueError('train.patience must be a nonnegative integer.')
        if (isinstance(min_delta, bool) or not isinstance(min_delta, (int, float))
                or not math.isfinite(min_delta) or min_delta < 0):
            raise ValueError('train.min_delta must be finite and nonnegative.')
        self.window = window
        self.min_delta = float(min_delta)
        self.patience = patience
        self.reset()

    @classmethod
    def from_config(cls, train):
        return cls(patience=train.get('patience', 0),
                   window=train.get('early_stopping_window', DEFAULT_WINDOW),
                   min_delta=train.get('min_delta', DEFAULT_MIN_DELTA))

    def reset(self):
        self.recent_losses = deque(maxlen=self.window)
        self.best_mean = None
        self.best_epoch = None
        self.last_epoch = None
        self.epochs_since_improvement = 0

    @property
    def current_mean(self):
        if len(self.recent_losses) < self.window:
            return None
        return math.fsum(self.recent_losses) / self.window

    def update(self, epoch, evaluation_loss, *, warming_up):
        """Record one epoch and return whether patience is exhausted."""
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Epoch must be a positive integer.')
        evaluation_loss = float(evaluation_loss)
        if not math.isfinite(evaluation_loss):
            raise ValueError('Evaluation total loss is nonfinite; early stopping cannot monitor it.')
        if self.last_epoch is not None and epoch != self.last_epoch + 1:
            raise ValueError('Early-stopping epochs must be consecutive.')
        if warming_up:
            self.reset()
            self.last_epoch = epoch
            return False
        self.last_epoch = epoch
        self.recent_losses.append(evaluation_loss)
        score = self.current_mean
        if score is None:
            return False
        if self.best_mean is None or score < self.best_mean - self.min_delta:
            self.best_mean = score
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        return self.patience > 0 and self.epochs_since_improvement >= self.patience

    def state_dict(self):
        return dict(version=1, metric='eval_total_loss', window=self.window, min_delta=self.min_delta,
                    recent_losses=list(self.recent_losses), best_mean=self.best_mean,
                    best_epoch=self.best_epoch, last_epoch=self.last_epoch,
                    epochs_since_improvement=self.epochs_since_improvement)

    def load_state_dict(self, state):
        """Restore the monitor, or reset when its policy changed or is absent."""
        self.reset()
        if state is None:
            return False
        if state.get('version') != 1:
            raise ValueError('Unsupported early-stopping state version.')
        if (state.get('metric') != 'eval_total_loss' or state['window'] != self.window
                or state['min_delta'] != self.min_delta):
            return False
        losses = [float(value) for value in state['recent_losses']]
        best = state['best_mean']
        count = state['epochs_since_improvement']
        last_epoch, best_epoch = state['last_epoch'], state['best_epoch']
        if (len(losses) > self.window or not all(math.isfinite(x) for x in losses)
                or type(count) is not int or count < 0
                or (last_epoch is not None and (type(last_epoch) is not int or last_epoch < 1))
                or (best is not None and (not math.isfinite(best)
                    or len(losses) != self.window or type(best_epoch) is not int
                    or last_epoch is None or not 1 <= best_epoch <= last_epoch))
                or (best is None and (count != 0 or best_epoch is not None))):
            raise ValueError('Invalid early-stopping checkpoint state.')
        self.recent_losses.extend(losses)
        self.best_mean, self.best_epoch = best, best_epoch
        self.last_epoch = last_epoch
        self.epochs_since_improvement = count
        return True
