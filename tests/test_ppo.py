import torch
import pytest
from ppo_trainer import MultiCategorical

@pytest.fixture
def example_logits_and_nvec():
    """Fixture to create sample logits and nvec for tests."""
    batch_size = 4
    nvec = [5, 7]  # 5 options for action 0, 7 options for action 1
    total_logits = sum(nvec)  # 5 + 7 = 12
    logits = torch.randn(batch_size, total_logits)
    return logits, nvec

def test_sample_shape(example_logits_and_nvec):
    logits, nvec = example_logits_and_nvec
    dist = MultiCategorical(logits, nvec)

    sample_shape = (6,)  # e.g., draw 6 samples
    samples = dist.sample(sample_shape=torch.Size(sample_shape))

    expected_shape = torch.Size(sample_shape + dist.batch_shape + dist.event_shape)
    assert samples.shape == expected_shape, f"Expected shape {expected_shape}, got {samples.shape}"

def test_log_prob_shape(example_logits_and_nvec):
    logits, nvec = example_logits_and_nvec
    dist = MultiCategorical(logits, nvec)

    samples = dist.sample()
    log_probs = dist.log_prob(samples)

    expected_shape = dist.batch_shape
    assert log_probs.shape == expected_shape, f"Expected shape {expected_shape}, got {log_probs.shape}"

def test_entropy_shape(example_logits_and_nvec):
    logits, nvec = example_logits_and_nvec
    dist = MultiCategorical(logits, nvec)

    entropy = dist.entropy()

    expected_shape = dist.batch_shape
    assert entropy.shape == expected_shape, f"Expected shape {expected_shape}, got {entropy.shape}"

def test_mode_shape(example_logits_and_nvec):
    logits, nvec = example_logits_and_nvec
    dist = MultiCategorical(logits, nvec)

    modes = dist.mode()

    expected_shape = dist.batch_shape + dist.event_shape
    assert modes.shape == expected_shape, f"Expected shape {expected_shape}, got {modes.shape}"
