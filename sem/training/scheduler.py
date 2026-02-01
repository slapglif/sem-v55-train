"""WSD (Warmup-Stable-Decay) LR scheduler for streaming training.

Unlike cosine scheduling, WSD does not require knowing total_steps upfront.
The stable phase runs indefinitely until externally triggered to decay,
making it ideal for streaming data with curriculum stage transitions.
"""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


class WSDScheduler(LambdaLR):
    """Warmup-Stable-Decay learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Linear warmup from 0 to base_lr
        decay_steps: Cosine decay from base_lr to min_lr
        min_lr_ratio: min_lr = base_lr * min_lr_ratio
    """

    def __init__(self, optimizer, warmup_steps=2000, decay_steps=5000,
                 min_lr_ratio=0.1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self._decay_start_step = None  # None = stable phase active
        self._current_step = 0

        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step):
        self._current_step = step

        # Phase 1: Warmup
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        # Phase 3: Decay (if triggered)
        if self._decay_start_step is not None:
            decay_elapsed = step - self._decay_start_step
            if decay_elapsed >= 0:
                if decay_elapsed >= self.decay_steps:
                    return self.min_lr_ratio
                progress = decay_elapsed / self.decay_steps
                return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        # Phase 2: Stable
        return 1.0

    def begin_decay(self):
        """Trigger transition from stable to decay phase."""
        self._decay_start_step = self._current_step

    def reset(self, new_warmup_steps=None):
        """Reset scheduler for new curriculum stage.

        Args:
            new_warmup_steps: Override warmup steps for this cycle (default: stage_warmup_steps=500)
        """
        if new_warmup_steps is not None:
            self.warmup_steps = new_warmup_steps
        self._decay_start_step = None
        # Reset step counter by adjusting last_epoch
        self.last_epoch = 0
        self._current_step = 0
        # Re-initialize base_lrs if optimizer LR was changed
        self._step_count = 1
        # Update _last_lr so get_last_lr() returns correct values immediately
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        state = super().state_dict()
        state['warmup_steps'] = self.warmup_steps
        state['decay_steps'] = self.decay_steps
        state['min_lr_ratio'] = self.min_lr_ratio
        state['_decay_start_step'] = self._decay_start_step
        state['_current_step'] = self._current_step
        return state

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.decay_steps = state_dict.pop('decay_steps')
        self.min_lr_ratio = state_dict.pop('min_lr_ratio')
        self._decay_start_step = state_dict.pop('_decay_start_step')
        self._current_step = state_dict.pop('_current_step')
        super().load_state_dict(state_dict)
