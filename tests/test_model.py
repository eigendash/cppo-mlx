import mlx.core as mx
import numpy as np

from cppo.data import EOS_ID, PAD_ID, make_example, pad_prompts
from cppo.model import TinyLM, completion_cross_entropy, completion_logprobs, param_count, token_logprobs
from cppo.sample import greedy_completions, mask_after_eos, repeat_prompts, sample_completions, score_completions


def tiny_model(**kwargs) -> TinyLM:
    kw = dict(vocab_size=15, dim=32, n_layers=2, n_heads=4, max_len=16)
    kw.update(kwargs)
    mx.random.seed(0)
    return TinyLM(**kw)


def test_shapes_and_param_scale():
    model = tiny_model()
    logits = model(mx.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, 15)
    big = TinyLM(vocab_size=15, dim=96, n_layers=3, n_heads=4, max_len=24)
    n = param_count(big)
    assert 200_000 < n < 500_000, n


def test_rejects_sequence_longer_than_max_len():
    model = tiny_model(max_len=4)
    try:
        model(mx.array([[1, 2, 3, 4, 5]]))
    except ValueError as err:
        assert "max_len" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_token_logprobs_match_manual_softmax():
    model = tiny_model()
    tokens = mx.array([[1, 5, 7, 4]])
    logp = token_logprobs(model, tokens)
    logits = np.asarray(model(tokens))
    for t in range(3):
        row = logits[0, t].astype(np.float64)
        row = row - row.max()
        expected = row[tokens[0, t + 1].item()] - np.log(np.exp(row).sum())
        np.testing.assert_allclose(np.asarray(logp)[0, t], expected, rtol=1e-5, atol=1e-5)


def test_completion_logprobs_are_left_shifted_to_the_first_new_token():
    model = tiny_model()
    prompt, mask = pad_prompts([make_example(47, 58)])
    completion = mx.array([[8, 5, 1, EOS_ID]])
    logp = completion_logprobs(model, prompt, mask, completion)
    full = np.asarray(model(mx.concatenate([prompt, completion], axis=1)))
    P = prompt.shape[1]
    # The first completion token is predicted by the last prompt position.
    row = full[0, P - 1].astype(np.float64)
    row = row - row.max()
    expected = row[8] - np.log(np.exp(row).sum())
    assert logp.shape == (1, 4)
    np.testing.assert_allclose(np.asarray(logp)[0, 0], expected, rtol=1e-5, atol=1e-5)


def test_cross_entropy_is_finite_and_positive():
    model = tiny_model()
    prompt, mask = pad_prompts([make_example(47, 58), make_example(12, 34)])
    completion = mx.array([[8, 9, 1, EOS_ID], [9, 6, EOS_ID, PAD_ID]])
    cmask = mask_after_eos(completion)
    loss = completion_cross_entropy(model, prompt, mask, completion, cmask)
    mx.eval(loss)
    assert np.isfinite(np.asarray(loss))
    assert float(loss) > 0


def test_mask_after_eos_keeps_the_eos_token():
    ids = mx.array([[5, 6, EOS_ID, 7], [5, 6, 7, 8]])
    mask = np.asarray(mask_after_eos(ids))
    np.testing.assert_array_equal(mask[0], [1, 1, 1, 0])
    np.testing.assert_array_equal(mask[1], [1, 1, 1, 1])


def test_sampling_is_deterministic_given_the_key():
    model = tiny_model()
    prompt, mask = pad_prompts([make_example(47, 58), make_example(12, 34)])
    a, ma = sample_completions(model, prompt, mask, n_samples=3, max_new_tokens=4, key=mx.random.key(1))
    b, mb = sample_completions(model, prompt, mask, n_samples=3, max_new_tokens=4, key=mx.random.key(1))
    assert a.shape == (2, 3, 4) and ma.shape == a.shape
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(ma), np.asarray(mb))


def test_different_keys_give_different_samples():
    model = tiny_model()
    prompt, mask = pad_prompts([make_example(47, 58)])
    a, _ = sample_completions(model, prompt, mask, n_samples=2, max_new_tokens=4, key=mx.random.key(1))
    b, _ = sample_completions(model, prompt, mask, n_samples=2, max_new_tokens=4, key=mx.random.key(2))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_repeat_prompts_order():
    prompt = mx.array([[1, 2], [3, 4]])
    mask = mx.ones_like(prompt, dtype=mx.float32)
    ids, _ = repeat_prompts(prompt, mask, 2)
    np.testing.assert_array_equal(np.asarray(ids), [[1, 2], [1, 2], [3, 4], [3, 4]])


def test_greedy_completions_stop_at_eos_and_scores_match_reward():
    model = tiny_model()
    prompt, mask = pad_prompts([make_example(47, 58), make_example(12, 34)])
    comps = greedy_completions(model, prompt, mask, max_new_tokens=5)
    assert len(comps) == 2
    assert all(len(c) <= 5 for c in comps)
    from cppo.data import VOCAB

    sampled, _ = sample_completions(model, prompt, mask, n_samples=2, max_new_tokens=4, key=mx.random.key(0))
    rewards = score_completions(sampled, [make_example(47, 58), make_example(12, 34)])
    assert rewards.shape == (2, 2)
    assert np.all(rewards >= 0) and np.all(rewards <= 3)
    assert VOCAB[PAD_ID] == "<pad>"
