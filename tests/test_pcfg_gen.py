from induce_text import pcfg_gen


def test_generator_sample_produces_data():
    gen = pcfg_gen.DataGenerator(seed=42)
    data = gen.sample()
    assert isinstance(data, pcfg_gen.HexList)
    assert data.data
    assert all(isinstance(x, pcfg_gen.Hex) for x in data.data)
