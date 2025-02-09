import ncps.mini_keras
from ncps.mini_keras.utils import register_mini_keras_serializable

@register_mini_keras_serializable(package="ncps", name="CfCCell")
class CfCCell(ncps.mini_keras.layers.AbstractRNNCell):
    def __init__(
        self,
        units,
        mode="default",
        activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.1,
        sparsity_mask=None,
        **kwargs,
    ):
        super().__init__(units=units, **kwargs)  # Pass units to parent AbstractRNNCell
        self.sparsity_mask = sparsity_mask
        if sparsity_mask is not None:
            # No backbone is allowed
            if backbone_units > 0:
                raise ValueError("If sparsity of a CfC cell is set, then no backbone is allowed")

        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {str(allowed_modes)}")
        self.mode = mode
        self.backbone_fn = None
        self._activation = ncps.mini_keras.activations.get(activation)
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout
        self._cfc_layers = []

    @property
    def state_size(self):
        return self.units  # Use self.units from parent class

    @property
    def output_size(self):
        return self.units  # Use self.units from parent class

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple) or isinstance(input_shape[0], ncps.mini_keras.KerasTensor):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        if self._backbone_layers > 0:
            backbone_layers = []
            for i in range(self._backbone_layers):
                backbone_layers.append(ncps.mini_keras.layers.Dense(self._backbone_units, self._activation, name=f"backbone{i}"))
                backbone_layers.append(ncps.mini_keras.layers.Dropout(self._backbone_dropout))
                
            self.backbone_fn = ncps.mini_keras.models.Sequential(backbone_layers)
            self.backbone_fn.build((None, self.state_size + input_dim))
            cat_shape = int(self._backbone_units)
        else:
            cat_shape = int(self.state_size + input_dim)

        self.ff1_kernel = self.add_weight(
            shape=(cat_shape, self.state_size), 
            initializer="glorot_uniform",
            name="ff1_weight",
        )
        self.ff1_bias = self.add_weight(
            shape=(self.state_size,),
            initializer="zeros",
            name="ff1_bias",
        )

        if self.mode == "pure":
            self.w_tau = self.add_weight(
                shape=(1, self.state_size), 
                initializer=ncps.mini_keras.initializers.Zeros(),
                name="w_tau", 
            )
            self.A = self.add_weight(
                shape=(1, self.state_size), 
                initializer=ncps.mini_keras.initializers.Ones(),
                name="A", 
            )
        else:
            self.ff2_kernel = self.add_weight(
                shape=(cat_shape, self.state_size), 
                initializer="glorot_uniform",
                name="ff2_weight", 
            )
            self.ff2_bias = self.add_weight(
                shape=(self.state_size,), 
                initializer="zeros",
                name="ff2_bias", 
            )

            self.time_a = ncps.mini_keras.layers.Dense(self.state_size, name="time_a")
            self.time_b = ncps.mini_keras.layers.Dense(self.state_size, name="time_b")
            input_shape = (None, self.state_size + input_dim)
            if self._backbone_layers > 0:
                input_shape = self.backbone_fn.output_shape
            self.time_a.build(input_shape)
            self.time_b.build(input_shape)
        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = ncps.mini_keras.ops.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
             t = kwargs.get("time") or 1.0
        x = ncps.mini_keras.layers.Concatenate()([inputs, states[0]])
        if self._backbone_layers > 0:
            x = self.backbone_fn(x)
        if self.sparsity_mask is not None:
            ff1_kernel = self.ff1_kernel * self.sparsity_mask
            ff1 = ncps.mini_keras.ops.matmul(x, ff1_kernel) + self.ff1_bias
        else:
            ff1 = ncps.mini_keras.ops.matmul(x, self.ff1_kernel) + self.ff1_bias
        if self.mode == "pure":
            # Solution
            new_hidden = (
                -self.A
                * ncps.mini_keras.ops.exp(-t * (ncps.mini_keras.ops.abs(self.w_tau) + ncps.mini_keras.ops.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            if self.sparsity_mask is not None:
                ff2_kernel = self.ff2_kernel * self.sparsity_mask
                ff2 = ncps.mini_keras.ops.matmul(x, ff2_kernel) + self.ff2_bias
            else:
                ff2 = ncps.mini_keras.ops.matmul(x, self.ff2_kernel) + self.ff2_bias
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = ncps.mini_keras.activations.sigmoid(-t_a * t + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]

    def get_config(self):
        config = {
            "units": self.units,  # Use self.units from parent class
            "mode": self.mode,
            "activation": self._activation,
            "backbone_units": self._backbone_units,
            "backbone_layers": self._backbone_layers,
            "backbone_dropout": self._backbone_dropout,
            "sparsity_mask": self.sparsity_mask,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
