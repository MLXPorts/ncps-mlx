import mlx.core as mx
import mlx.optimizers as optim

from examples import mlx_cfc_regression as cfc_example
from examples import mlx_smnist_training as smnist_example
from examples import mlx_cheetah_training as cheetah_example


def test_cfc_regression_checkpoint(tmp_path):
    inputs, targets = cfc_example.make_dataset(num_sequences=4, seq_len=32)
    model = cfc_example.make_model(input_dim=inputs.shape[-1], hidden_dim=8, output_dim=1)
    weights_path = tmp_path / "cfc_weights.npz"
    cfc_example.train(model, inputs, targets, epochs=2, checkpoint_path=weights_path)
    assert weights_path.exists()

    restored = cfc_example.make_model(input_dim=inputs.shape[-1], hidden_dim=8, output_dim=1)
    restored.load_weights(str(weights_path))
    mx.eval(restored.parameters())
    sample, _ = restored(inputs[:1])
    assert sample.shape[-1] == 1


def test_smnist_training_checkpoint(tmp_path):
    data = smnist_example.SMnistData(max_train=32, max_eval=32)
    weights_path = tmp_path / "smnist_weights.npz"

    model, stats = smnist_example.train_ltc(
        data=data,
        hidden_dim=16,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        log_period=1,
        solver="semi_implicit",
        ode_unfolds=2,
        checkpoint_path=weights_path,
    )

    assert weights_path.exists()
    assert "valid_acc" in stats

    sample_batch = data.full_split("test")
    logits = model(sample_batch.inputs[:4])
    assert logits.shape == (4, 10)


def test_cheetah_training_utils(tmp_path):
    batch, seq_len, features = 8, 12, 5
    inputs = mx.random.uniform(low=-0.5, high=0.5, shape=(batch, seq_len, features))
    targets = inputs  # identity regression keeps loss bounded
    model = cheetah_example.CheetahPredictor(hidden_dim=16, output_dim=features, input_dim=features)
    optimizer = optim.Adam(learning_rate=1e-3)
    mx.eval(model.parameters())
    loss = cheetah_example.train_epoch(model, optimizer, inputs, targets, batch_size=4)
    assert loss >= 0.0
    mae = cheetah_example.evaluate_mae(model, inputs, targets)
    assert mae >= 0.0
