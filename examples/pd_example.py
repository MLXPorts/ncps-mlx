"""Policy distillation style example with MLX LTC students."""

from __future__ import annotations

import mlx.core as mx

from ncps.neurons import LTC, CfC
from ncps.wirings import FullyConnected, AutoNCP

from mlx_common import make_sine_dataset, train_sequence_model


def main() -> None:
    inputs, targets = make_sine_dataset(seq_len=96, batch_size=8, double_frequency=True)

    teacher = LTC(
        input_size=inputs.shape[-1],
        units=FullyConnected(units=24, output_dim=1),
        return_sequences=True,
        implicit_param_constraints=False,
    )
    train_sequence_model(
        teacher,
        inputs,
        targets,
        epochs=200,
        learning_rate=5e-3,
        apply_constraints=teacher.apply_weight_constraints,
    )

    with_teacher, _ = teacher(inputs)
    teacher_targets = mx.stop_gradient(with_teacher)

    student = CfC(
        input_size=inputs.shape[-1],
        units=AutoNCP(units=16, output_size=1, sparsity_level=0.5),
        return_sequences=True,
    )
    train_sequence_model(
        student,
        inputs,
        teacher_targets,
        epochs=150,
        learning_rate=5e-3,
    )

    preds, _ = student(inputs)
    teacher_head = mx.slice(
        teacher_targets,
        mx.array([0, 0, 0], dtype=mx.int32),
        (0, 1, 2),
        (1, 5, 1),
    )
    teacher_head = mx.reshape(teacher_head, (5,))
    student_head = mx.slice(
        preds,
        mx.array([0, 0, 0], dtype=mx.int32),
        (0, 1, 2),
        (1, 5, 1),
    )
    student_head = mx.reshape(student_head, (5,))
    print("teacher sample:", teacher_head.tolist())
    print("student sample:", student_head.tolist())


if __name__ == "__main__":
    main()
