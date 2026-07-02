from induce_text import pcfg_gen


def test_generator_sample_produces_data():
    gen = pcfg_gen.DataGenerator(seed=42)
    data = gen.sample()
    assert isinstance(data, pcfg_gen.HexListData)
    assert data.data
    assert all(isinstance(x, int) and 0 <= x <= 0xF for x in data.data)
