"""The rule-based reward of the paper, adapted to the addition task.

The paper's GSM8K reward is a sum of a format term and an accuracy term:

    R_format(o)   = 1   if o follows the required format, else 0
    R_accuracy(o) = 2   if o matches the answer directly
                  = 1.5 if o matches after regular parsing
                  = 0   otherwise

On this task "follows the format" means the completion is a non-empty run of
digit tokens ended by <eos> (or by the end of the generation window).  A direct
match is the canonical least-significant-first string; the 1.5 tier is a
completion that is not the canonical string but whose digits read in the usual
most-significant-first order spell the right number, i.e. the model computed the
sum but wrote it in the other order.
"""

from __future__ import annotations

from .data import EOS_ID, Example, digit_text, is_digit_id


def split_completion(completion_ids) -> list[int]:
    """Tokens before the first <eos> (the payload the reward looks at)."""
    payload: list[int] = []
    for token in completion_ids:
        token = int(token)
        if token == EOS_ID:
            break
        payload.append(token)
    return payload


def is_well_formed(payload: list[int]) -> bool:
    return len(payload) > 0 and all(is_digit_id(t) for t in payload)


def reward(completion_ids, example: Example) -> float:
    """Format reward + accuracy reward, in [0, 3]."""
    payload = split_completion(completion_ids)
    if not is_well_formed(payload):
        return 0.0
    format_reward = 1.0
    text = digit_text(payload)
    if text == example.target_digits:
        return format_reward + 2.0
    if int(text) == example.value:
        return format_reward + 1.5
    return format_reward


def accuracy(completion_ids, example: Example) -> float:
    """1.0 when the completion carries the correct answer, else 0.0."""
    payload = split_completion(completion_ids)
    if not is_well_formed(payload):
        return 0.0
    text = digit_text(payload)
    return 1.0 if (text == example.target_digits or int(text) == example.value) else 0.0
