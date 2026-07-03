import numpy as np

from scripts.train_alignment_adapter import apply_adapter_np, split_gold_items, train_linear_adapter


def test_train_linear_adapter_learns_simple_swap_mapping():
    spec = train_linear_adapter(
        source_vectors=[[1.0, 0.0], [0.0, 1.0]],
        target_vectors=[[0.0, 1.0], [1.0, 0.0]],
        alpha=0.001,
    )

    mapped = apply_adapter_np([1.0, 0.0], spec, mix=1.0)

    assert int(np.argmax(mapped)) == 1
    assert spec["input_dim"] == 2
    assert spec["output_dim"] == 2
    assert spec["num_pairs"] == 2


def test_split_gold_items_is_deterministic_4_to_1():
    items = [{"music_id": str(index)} for index in range(10)]

    train, validation = split_gold_items(items)

    assert [item["music_id"] for item in validation] == ["0", "5"]
    assert len(train) == 8
