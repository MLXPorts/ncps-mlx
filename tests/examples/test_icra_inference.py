import mlx.core as mx

from examples.icra_lidar_infer import build_model, predict_sequence
from datasets.icra2020_lidar_collision_avoidance import load_data


def test_icra_inference_sequence():
    seq_len = 8
    (train_x, train_y), _ = load_data(seq_len=seq_len)
    train_x = mx.array(train_x)  # ensure MLX array type
    x = mx.reshape(train_x[0], (seq_len, -1))  # [T, L+2]

    model = build_model(total_input_dim=x.shape[-1])
    y_hat, hx_next = predict_sequence(model, x)

    assert y_hat.shape == (seq_len, 1)
    assert hx_next.shape[0] == 1 and hx_next.shape[1] == model.state_size
