import numpy as np
import pytest

from cppo.data import (
    AdditionTask,
    BOS_ID,
    EOS_ID,
    EQ_ID,
    PLUS_ID,
    VOCAB,
    VOCAB_SIZE,
    answer_digits_lsb,
    decode,
    digit_id,
    make_example,
    pad_prompts,
    prompt_ids,
)
from cppo.reward import accuracy, is_well_formed, reward, split_completion


def completion(text: str) -> list[int]:
    """Encode a completion such as '501' (EOS appended)."""
    return [digit_id(int(c)) for c in text] + [EOS_ID]


def test_vocab_layout_matches_helpers():
    assert VOCAB[:5] == ["<pad>", "<eos>", "<bos>", "+", "="]
    assert VOCAB_SIZE == 15
    assert [VOCAB[digit_id(d)] for d in range(10)] == [str(d) for d in range(10)]


def test_prompt_encoding_is_exact():
    assert prompt_ids(47, 58) == [
        BOS_ID,
        digit_id(4),
        digit_id(7),
        PLUS_ID,
        digit_id(5),
        digit_id(8),
        EQ_ID,
    ]
    assert decode(prompt_ids(12, 3)) == "<bos>12+3="


def test_answer_is_written_least_significant_first():
    assert answer_digits_lsb(105) == "501"
    assert answer_digits_lsb(7) == "7"
    ex = make_example(47, 58)
    assert ex.value == 105
    assert ex.target_digits == "501"
    assert [VOCAB[t] for t in ex.target_tokens] == ["5", "0", "1", "<eos>"]


def test_task_splits_are_disjoint_and_deterministic():
    t1 = AdditionTask(digits=2, n_train=40, n_test=20, seed=0)
    t2 = AdditionTask(digits=2, n_train=40, n_test=20, seed=0)
    assert [e.a for e in t1.train] == [e.a for e in t2.train]
    assert [e.b for e in t1.train] == [e.b for e in t2.train]
    train_pairs = {(e.a, e.b) for e in t1.train}
    test_pairs = {(e.a, e.b) for e in t1.test}
    assert not (train_pairs & test_pairs)
    assert all(10 <= e.a <= 99 and 10 <= e.b <= 99 for e in t1.train + t1.test)
    assert len(t1.train) == 40 and len(t1.test) == 20


def test_pad_prompts_pads_and_masks():
    ex = [make_example(11, 2), make_example(99, 88)]
    ids, mask = pad_prompts(ex)
    assert ids.shape == mask.shape == (2, max(len(e.prompt) for e in ex))
    assert np.asarray(mask)[0].sum() == len(ex[0].prompt)
    assert np.asarray(mask)[1].sum() == len(ex[1].prompt)
    assert int(np.asarray(ids)[0, len(ex[0].prompt)]) == 0  # PAD


def test_reward_exact_match_beats_parsed_match():
    ex = make_example(47, 58)  # canonical answer "501"
    assert reward(completion("501"), ex) == 3.0
    assert reward(completion("105"), ex) == 2.5  # right number, wrong order
    assert reward(completion("999"), ex) == 1.0  # well formed, wrong
    assert accuracy(completion("501"), ex) == 1.0
    assert accuracy(completion("105"), ex) == 1.0
    assert accuracy(completion("999"), ex) == 0.0


def test_reward_penalises_bad_format():
    ex = make_example(47, 58)
    assert reward([], ex) == 0.0
    assert reward([EOS_ID], ex) == 0.0
    assert reward([PLUS_ID, digit_id(1)], ex) == 0.0
    assert accuracy([PLUS_ID], ex) == 0.0
    # Anything after <eos> is ignored.
    assert reward(completion("501") + [PLUS_ID, PLUS_ID], ex) == 3.0


def test_split_completion_and_well_formed():
    assert split_completion([digit_id(5), digit_id(0), EOS_ID, digit_id(1)]) == [digit_id(5), digit_id(0)]
    assert is_well_formed([digit_id(5)])
    assert not is_well_formed([])
    assert not is_well_formed([EOS_ID])
    assert not is_well_formed([PLUS_ID])


@pytest.mark.parametrize("digits", [1, 2])
def test_value_equals_sum_for_every_example(digits):
    task = AdditionTask(digits=digits, n_train=25, n_test=10, seed=3)
    for ex in task.train + task.test:
        assert ex.value == ex.a + ex.b
        assert ex.target_digits == str(ex.a + ex.b)[::-1]
        assert reward(completion(ex.target_digits), ex) == 3.0
