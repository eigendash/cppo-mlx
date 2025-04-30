"""A synthetic, programmatically verifiable reasoning task.

The task is k-digit addition with the answer written **least-significant digit
first**.  Writing the sum in that order means the model has to emit the carry
propagation left to right, so a decoder-only LM can solve it with a single pass
of local computation instead of having to reverse the digits internally.  It is
the same trick as the classic "addition" task in makemore, and it is what makes
a tiny model able to solve this at all.

    prompt      <bos> 4 7 + 5 8 =
    completion  5 0 1 <eos>      (47 + 58 = 105, written 5,0,1)

Everything is a character-level token; the vocabulary is the ten digits, "+",
"=", plus padding / end-of-sequence / begin-of-sequence markers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD = "<pad>"
EOS = "<eos>"
BOS = "<bos>"
PLUS = "+"
EQ = "="

SPECIALS = [PAD, EOS, BOS, PLUS, EQ]
VOCAB = SPECIALS + [str(d) for d in range(10)]

PAD_ID = VOCAB.index(PAD)
EOS_ID = VOCAB.index(EOS)
BOS_ID = VOCAB.index(BOS)
PLUS_ID = VOCAB.index(PLUS)
EQ_ID = VOCAB.index(EQ)
FIRST_DIGIT_ID = VOCAB.index("0")

VOCAB_SIZE = len(VOCAB)


def digit_id(d: int) -> int:
    return FIRST_DIGIT_ID + d


def is_digit_id(token_id: int) -> bool:
    return FIRST_DIGIT_ID <= token_id < VOCAB_SIZE


def digits_of(n: int) -> list[int]:
    """Digits of ``n``, most significant first."""
    return [int(c) for c in str(n)]


def prompt_ids(a: int, b: int) -> list[int]:
    """``<bos> a + b =``."""
    return (
        [BOS_ID]
        + [digit_id(d) for d in digits_of(a)]
        + [PLUS_ID]
        + [digit_id(d) for d in digits_of(b)]
        + [EQ_ID]
    )


def answer_digits_lsb(value: int) -> str:
    """The canonical answer string: least significant digit first."""
    return str(value)[::-1]


def decode(token_ids) -> str:
    return "".join(VOCAB[int(t)] for t in token_ids)


def digit_text(token_ids) -> str:
    return "".join(VOCAB[int(t)] for t in token_ids)


@dataclass(frozen=True)
class Example:
    a: int
    b: int
    prompt: tuple[int, ...]
    target_digits: str  # least significant digit first
    value: int

    @property
    def target_tokens(self) -> list[int]:
        return [digit_id(int(c)) for c in self.target_digits] + [EOS_ID]


def make_example(a: int, b: int) -> Example:
    value = a + b
    return Example(
        a=a,
        b=b,
        prompt=tuple(prompt_ids(a, b)),
        target_digits=answer_digits_lsb(value),
        value=value,
    )


@dataclass
class AdditionTask:
    """Train / test splits of k-digit addition problems."""

    digits: int = 2
    n_train: int = 512
    n_test: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        lo = 10 ** (self.digits - 1)
        hi = 10**self.digits - 1
        rng = random.Random(self.seed)
        total = self.n_train + self.n_test
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        while len(pairs) < total:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))
        self.train = [make_example(a, b) for a, b in pairs[: self.n_train]]
        self.test = [make_example(a, b) for a, b in pairs[self.n_train :]]

    @property
    def max_prompt_len(self) -> int:
        return max(len(e.prompt) for e in self.train + self.test)

    @property
    def max_answer_len(self) -> int:
        return max(len(e.target_tokens) for e in self.train + self.test)


def pad_prompts(examples, pad_id: int = PAD_ID):
    """Right-pad prompts into a (B, P) array plus a (B, P) validity mask."""
    import mlx.core as mx

    width = max(len(e.prompt) for e in examples)
    ids = [[0] * width for _ in examples]
    mask = [[0.0] * width for _ in examples]
    for i, e in enumerate(examples):
        ids[i][: len(e.prompt)] = list(e.prompt)
        mask[i][: len(e.prompt)] = [1.0] * len(e.prompt)
    return mx.array(ids, dtype=mx.int32), mx.array(mask, dtype=mx.float32)
