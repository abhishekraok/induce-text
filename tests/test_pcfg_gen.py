from induce_text.pcfg_gen import Rule, sample, ChoiceMaker


def test_generator_sample_produces_different_data():
    rule = Rule(symbols=["a", "b", ["c", "d"], "e", "f", ["x", "a"]])
    env = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": rule}
    for seed in range(4):
        choicemaker = ChoiceMaker(seed=seed)
        data = sample(rule=rule, env=env, choicemaker=choicemaker)
        assert isinstance(data, list)
        assert all(isinstance(x, int) for x in data)
