from induce_text.metrics import Result, bits_per_byte


def test_result_derived_quantities():
    # 1000 bytes coded in 2000 bits = 2.0 bpc.
    r = Result(name="x", data_bytes=1000, total_bits=2000.0)
    assert r.bits_per_byte == 2.0
    assert r.compressed_bytes == 250.0
    assert r.compression_ratio == 0.25
    assert r.space_saving == 0.75


def test_empty_input_is_safe():
    r = Result(name="x", data_bytes=0, total_bits=0.0)
    assert r.bits_per_byte == 0.0
    assert r.compression_ratio == 1.0
    assert bits_per_byte(0.0, 0) == 0.0


def test_no_compression_is_eight_bpc():
    r = Result(name="raw", data_bytes=100, total_bits=800.0)
    assert r.bits_per_byte == 8.0
    assert r.compression_ratio == 1.0
