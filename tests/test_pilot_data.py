import pytest

from trihaera.pilot_data import pack_text_windows


class ToyTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [2 + ord(char) % 43 for char in text]


def test_packing_deterministic_for_same_seed_and_different_with_other_seeds():
    rows = [{"text": f"record {i}: long enough scientific paragraph about data {i}."}
            for i in range(50)]
    first, meta = pack_text_windows(
        rows, ToyTokenizer(), sequence_length=16, max_windows=15,
        max_rows=50, seed=12
    )
    second, meta2 = pack_text_windows(
        rows, ToyTokenizer(), sequence_length=16, max_windows=15,
        max_rows=50, seed=12
    )
    third, _ = pack_text_windows(
        rows, ToyTokenizer(), sequence_length=16, max_windows=15,
        max_rows=50, seed=13
    )
    assert first.shape == (15, 16)
    assert first.equal(second)
    assert meta == meta2
    assert not first.equal(third)


def test_rejects_insufficient_data_and_invalid_budget():
    with pytest.raises(ValueError):
        pack_text_windows([{"text": "tiny"}], ToyTokenizer(),
                          sequence_length=32, max_windows=5, max_rows=1)
    with pytest.raises(ValueError):
        pack_text_windows([], ToyTokenizer(),
                          sequence_length=2, max_windows=5, max_rows=1)


def test_eos_separator_is_present():
    rows = [{"text": "tiny input"}] * 30
    output, _ = pack_text_windows(rows, ToyTokenizer(), sequence_length=8,
                                  max_windows=6, max_rows=30)
    assert output.eq(1).any()
