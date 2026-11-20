"""Decide when Multi-GNN's training loop really evaluates, without editing it.

Multi-GNN scores validation and test after every epoch. That is slow (about
three minutes per validation pass on LI-Small), and with 10% of the labels
there are ten times as many epochs, so ten times as many passes. So:

  * the test loader is never run during training (AGENTS.md rule 2);
  * the validation loader runs only on a fixed schedule: the same number of
    times for every label fraction, spread evenly, always ending on the last
    epoch. Other epochs report "not scored", which never beats the best score,
    so the stock best-checkpoint rule still picks the best *validated* epoch.

Example:
    validation_epochs(epochs=30, validations=3)   # {9, 19, 29}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, Optional


class NotScored(float):
    """A NaN that prints as "not scored".

    Multi-GNN logs `Test F1: {value:.4f}`. Returning this instead of a number
    makes the log line read "Test F1: not scored" without editing its code.
    """

    def __format__(self, format_spec: str) -> str:
        return "not scored"


NOT_SCORED = NotScored("nan")


def validation_epochs(epochs: int, validations: Optional[int]) -> FrozenSet[int]:
    """Epoch numbers (from 0) after which validation runs; the last epoch is included.

    `None` means every epoch. Asking for more validations than epochs gives every epoch.
    """
    count = epochs if validations is None else min(validations, epochs)
    if count < 1:
        raise ValueError("Need at least one validation pass and one epoch")
    return frozenset(((step + 1) * epochs) // count - 1 for step in range(count))


def scheduled_evaluation(
    original: Callable[..., float],
    holder: Dict[str, Any],
    schedule: FrozenSet[int],
) -> Callable[..., float]:
    """Wrap an evaluate function: skip test, and validate only on scheduled epochs.

    `holder` carries the loaders and a counter of validation calls, which is
    also the epoch number because Multi-GNN validates once per epoch.
    """

    def evaluate(loader: Any, *rest: Any) -> float:
        if loader is holder.get("test_loader"):
            return NOT_SCORED
        if loader is holder.get("validation_loader"):
            epoch = holder.get("epoch", 0)
            holder["epoch"] = epoch + 1
            if epoch not in schedule:
                return NOT_SCORED
        return original(loader, *rest)

    return evaluate


def remember_loaders(
    original_get_loaders: Callable[..., Any], holder: Dict[str, Any]
) -> Callable[..., Any]:
    """Wrap get_loaders so we know which loaders are validation and test."""

    def get_loaders(*arguments: Any) -> Any:
        loaders = original_get_loaders(*arguments)
        holder["validation_loader"] = loaders[1]
        holder["test_loader"] = loaders[2]
        return loaders

    return get_loaders
