from induce_text.pcfg_gen import BinaryHexChoice, Rule, DataGenerator, ChoiceMaker


def test_generator_sample_produces_different_data():
    rule_5_then_a_or_b = Rule(symbols=[5, BinaryHexChoice(a=0xA, b=0xB)])
    rule = Rule(symbols=[0x0, 0x0, rule_5_then_a_or_b, 0x0, BinaryHexChoice(a=3, b=4)])
    gen = DataGenerator(rule=rule)
    all_data = []
    for seed in range(100):
        choicemaker = ChoiceMaker(seed=seed)
        data = gen.sample(choicemaker)
        assert isinstance(data, list)
        assert all(isinstance(x, int) for x in data)
        all_data.append(tuple(data))
        assert choicemaker.count == 2
    assert len(set(all_data)) == 4
