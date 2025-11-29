import torch
from src.models.heads import (
    ClassificationHead,
    RegressionHead,
    MultiLabelHead,
    ContrastiveHead,
    SequenceGenerationHead,
    MultiTaskHead,
    create_task_head,
)


def test_classification_and_regression_forward():
    x = torch.randn(2, 512)
    c = ClassificationHead(hidden_dim=512, num_classes=10)
    out = c(x)
    assert out.shape == (2, 10)

    r = RegressionHead(hidden_dim=512, output_dim=3)
    out_r = r(x)
    assert out_r.shape == (2, 3)


def test_multilabel_and_contrastive_and_task_factory():
    x = torch.randn(4, 512)
    m = MultiLabelHead(hidden_dim=512, num_labels=5)
    out_m = m(x)
    assert out_m.shape == (4, 5)

    # Contrastive: return similarity
    c = ContrastiveHead(hidden_dim=512, projection_dim=32)
    img = torch.randn(3, 512)
    txt = torch.randn(3, 512)
    sim = c(img, txt)
    assert sim.shape == (3, 3)

    # Sequence generation: with target ids
    seq = SequenceGenerationHead(
        hidden_dim=64, vocab_size=100, max_seq_length=10, num_layers=1, num_heads=2
    )
    encoder_out = torch.randn(2, 5, 64)
    target_ids = torch.randint(0, 100, (2, 3))
    logits = seq(encoder_out, target_ids=target_ids)
    assert logits.shape == (2, 3, 100)

    # Factory
    h = create_task_head(
        {"type": "classification", "hidden_dim": 128, "num_classes": 7}
    )
    assert h(torch.randn(1, 128)).shape == (1, 7)


def test_multitask_head_combination():
    features = torch.randn(2, 512)
    tasks = {
        "a": {"type": "classification", "num_classes": 3},
        "b": {"type": "regression", "output_dim": 2},
    }
    mt = MultiTaskHead(hidden_dim=512, tasks=tasks)
    out_all = mt(features)
    assert set(out_all.keys()) == set(["a", "b"])
    out_a = mt(features, task_name="a")
    assert "a" in out_a
