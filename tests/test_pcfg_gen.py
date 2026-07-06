import pytest

from induce_text.pcfg_gen import Rule, sample, RecordingChoice, ReplayChoice


def test_sample_replay():
    rule = Rule(symbols=["a", "b", ["c", "d"], "e", "f", ["x", "a"]])
    env = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": rule}
    for seed in range(9):
        cs = RecordingChoice(seed=seed)
        data = sample(rule=rule, env=env, choicesource=cs)
        assert isinstance(data, list)
        assert all(isinstance(x, int) for x in data)
        rc = ReplayChoice(cs.choices)
        replayed_data = sample(rule=rule, env=env, choicesource=rc)
        assert replayed_data == data
        assert rc.index == len(cs.choices)


def test_replay_exhausted_raises():
    rule = Rule(symbols=["a", "b", ["c", "d"], "e", "f", ["x", "a"]])
    env = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": rule}
    cs = RecordingChoice(seed=0)
    sample(rule=rule, env=env, choicesource=cs)
    truncated = ReplayChoice(cs.choices[:-1])
    with pytest.raises(ValueError):
        sample(rule=rule, env=env, choicesource=truncated)
