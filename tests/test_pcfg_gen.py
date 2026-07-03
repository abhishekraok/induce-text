from induce_text.pcfg_gen import BinaryHexChoice, Rule, DataGenerator


def test_generator_sample_produces_different_data():
    rule = Rule(
        symbols=[0x0, 0x0, BinaryHexChoice(a=1, b=2), 0x0, BinaryHexChoice(a=3, b=4)]
    )
    all_data = []
    for seed in range(100):
        gen = DataGenerator(seed=seed, rules=[rule])
        data = gen.sample()
        print(data)
        data = gen.sample()
        assert isinstance(data, list)
        assert all(isinstance(x, int) for x in data)
        all_data.append(tuple(data))
    assert len(set(all_data)) == 4
