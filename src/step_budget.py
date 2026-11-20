"""Equal training steps across label fractions.

With 10% of the labels, one epoch has about a tenth of the gradient steps. Giving
every fraction the same number of epochs would starve the small ones and favour
whichever setup learns fastest. So the budget is counted in steps instead: it is
the steps in N epochs at fraction 1.0, and a smaller fraction runs proportionally
more epochs. Multi-GNN's loop only runs whole epochs, so the totals match to
within one batch per epoch; the run record stores the actual step count.

Example (N = 5):
    epochs_for_budget(5, 1.0)    # 5
    epochs_for_budget(5, 0.25)   # 20
    epochs_for_budget(5, 0.10)   # 50
"""

from __future__ import annotations

import math


def epochs_for_budget(step_budget_epochs: int, fraction: float) -> int:
    """Whole epochs at `fraction` that use about as many steps as the budget."""
    if step_budget_epochs < 1:
        raise ValueError("The step budget must be at least one epoch")
    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1], got %r" % fraction)
    return max(1, int(round(step_budget_epochs / fraction)))


def steps_per_epoch(labels_kept: int, batch_size: int) -> int:
    """Gradient steps in one epoch: one per batch of seed transactions."""
    return int(math.ceil(labels_kept / batch_size))
