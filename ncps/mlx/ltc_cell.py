"""Liquid Time-Constant (LTC) cell implementation faithful to the MIT CSAIL design."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import mlx.core as mx
import mlx.nn as nn

from .base import LiquidCell
from .typing import InitializerCallable


class MappingType(str, Enum):
    IDENTITY = "identity"
    LINEAR = "linear"
    AFFINE = "affine"


class ODESolver(str, Enum):
    SEMI_IMPLICIT = "semi_implicit"
    EXPLICIT = "explicit"
    RUNGE_KUTTA = "runge_kutta"


class LTCCell(LiquidCell):
    """Liquid Time-Constant (LTC) cell mirroring the original TensorFlow implementation."""

    def __init__(
        self,
        wiring,
        activation: str = "identity",
        backbone_units: Optional[List[int]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        initializer: Optional[InitializerCallable] = None,
        *,
        ode_solver: str = "semi_implicit",
        ode_solver_unfolds: int = 6,
        input_mapping: str = "affine",
        erev_init_factor: float = 1.0,
        w_init_min: float = 0.01,
        w_init_max: float = 1.0,
        cm_init_min: float = 0.5,
        cm_init_max: float = 0.5,
        gleak_init_min: float = 1.0,
        gleak_init_max: float = 1.0,
        w_min_value: float = 1e-5,
        w_max_value: float = 1e3,
        gleak_min_value: float = 1e-5,
        gleak_max_value: float = 1e3,
        cm_min_value: float = 1e-6,
        cm_max_value: float = 1e3,
        fix_cm: Optional[float] = None,
        fix_gleak: Optional[float] = None,
        fix_vleak: Optional[float] = None,
    ):
        super().__init__(
            wiring=wiring,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
            initializer=initializer,
        )

        self.input_size = wiring.input_dim if wiring.input_dim is not None else 0
        self.hidden_size = wiring.units

        self._solver = ODESolver(ode_solver.lower())
        self.ode_solver_unfolds = max(1, int(ode_solver_unfolds))
        self._input_mapping = MappingType(input_mapping.lower())
        self._erev_init_factor = erev_init_factor

        self._w_init_min = w_init_min
        self._w_init_max = w_init_max
        self._cm_init_min = cm_init_min
        self._cm_init_max = cm_init_max
        self._gleak_init_min = gleak_init_min
        self._gleak_init_max = gleak_init_max

        self._w_min_value = w_min_value
        self._w_max_value = w_max_value
        self._gleak_min_value = gleak_min_value
        self._gleak_max_value = gleak_max_value
        self._cm_min_value = cm_min_value
        self._cm_max_value = cm_max_value

        self._fix_cm = fix_cm
        self._fix_gleak = fix_gleak
        self._fix_vleak = fix_vleak

        self._ltc_params_built_for: Optional[Tuple[int, int]] = None
        self._mapping_built_for: Optional[int] = None

    # --------------------------------------------------------------------- #
    # Parameter initialisation helpers
    # --------------------------------------------------------------------- #
    def set_solver(self, solver: str) -> None:
        self._solver = ODESolver(solver.lower())

    def set_input_mapping(self, mapping: str) -> None:
        self._input_mapping = MappingType(mapping.lower())
        self._mapping_built_for = None  # force rebuild

    def _ensure_parameters(self, input_dim: int) -> None:
        self.input_size = input_dim
        concat_dim = self.input_size + self.hidden_size
        self.backbone_input_dim = concat_dim

        if self.backbone_layers > 0 and self.backbone_units:
            if self._backbone_built_for != concat_dim:
                self.backbone_input_dim = concat_dim
                self.build_backbone()
                self._backbone_built_for = concat_dim
            effective_dim = self.backbone_output_dim
        else:
            self._backbone_layers = []
            self.backbone = None
            self.backbone_output_dim = concat_dim
            effective_dim = concat_dim

        self._ensure_input_mapping()
        self._ensure_ltc_parameters()

        if self.wiring.output_dim != self.hidden_size:
            need_output = (
                not hasattr(self, "output_proj")
                or self.output_proj.weight.shape[0] != self.wiring.output_dim
                or self.output_proj.weight.shape[1] != self.hidden_size
            )
            if need_output:
                self.output_proj = nn.Linear(self.hidden_size, self.wiring.output_dim)
                self.output_proj.weight = self.initializer((self.wiring.output_dim, self.hidden_size))
                self.output_proj.bias = mx.zeros((self.wiring.output_dim,))
        elif hasattr(self, "output_proj"):
            del self.output_proj

    def _ensure_input_mapping(self) -> None:
        if self._input_mapping is MappingType.IDENTITY:
            if hasattr(self, "input_w"):
                del self.input_w
            if hasattr(self, "input_b"):
                del self.input_b
            self._mapping_built_for = self.input_size
            return

        if self._mapping_built_for == self.input_size:
            return

        self.input_w = mx.ones((self.input_size,))
        if self._input_mapping is MappingType.AFFINE:
            self.input_b = mx.zeros((self.input_size,))
        elif hasattr(self, "input_b"):
            del self.input_b
        self._mapping_built_for = self.input_size

    def _ensure_ltc_parameters(self) -> None:
        key = (self.input_size, self.hidden_size)
        if self._ltc_params_built_for == key:
            return

        sensory_shape = (self.input_size, self.hidden_size)
        recurrent_shape = (self.hidden_size, self.hidden_size)

        if not hasattr(self, "sensory_mu") or self.sensory_mu.shape != sensory_shape:
            self.sensory_mu = mx.random.uniform(
                low=0.3, high=0.8, shape=sensory_shape
            )
        if not hasattr(self, "sensory_sigma") or self.sensory_sigma.shape != sensory_shape:
            self.sensory_sigma = mx.random.uniform(
                low=3.0, high=8.0, shape=sensory_shape
            )
        if not hasattr(self, "sensory_W") or self.sensory_W.shape != sensory_shape:
            self.sensory_W = mx.random.uniform(
                low=self._w_init_min, high=self._w_init_max, shape=sensory_shape
            )
        if not hasattr(self, "sensory_erev") or self.sensory_erev.shape != sensory_shape:
            sensory_sign = mx.random.randint(low=0, high=2, shape=sensory_shape, dtype=mx.uint32)
            sensory_sign = sensory_sign.astype(mx.float32) * 2 - 1
            self.sensory_erev = sensory_sign * self._erev_init_factor

        if not hasattr(self, "mu") or self.mu.shape != recurrent_shape:
            self.mu = mx.random.uniform(low=0.3, high=0.8, shape=recurrent_shape)
        if not hasattr(self, "sigma") or self.sigma.shape != recurrent_shape:
            self.sigma = mx.random.uniform(low=3.0, high=8.0, shape=recurrent_shape)
        if not hasattr(self, "W") or self.W.shape != recurrent_shape:
            self.W = mx.random.uniform(
                low=self._w_init_min, high=self._w_init_max, shape=recurrent_shape
            )
        if not hasattr(self, "erev") or self.erev.shape != recurrent_shape:
            sign = mx.random.randint(low=0, high=2, shape=recurrent_shape, dtype=mx.uint32)
            sign = sign.astype(mx.float32) * 2 - 1
            self.erev = sign * self._erev_init_factor

        if self._fix_vleak is None:
            if not hasattr(self, "vleak") or self.vleak.shape != (self.hidden_size,):
                self.vleak = mx.random.uniform(low=-0.2, high=0.2, shape=(self.hidden_size,))
        else:
            self.vleak = mx.zeros((self.hidden_size,)) + self._fix_vleak

        if self._fix_gleak is None:
            if not hasattr(self, "gleak") or self.gleak.shape != (self.hidden_size,):
                if self._gleak_init_max > self._gleak_init_min:
                    self.gleak = mx.random.uniform(
                        low=self._gleak_init_min,
                        high=self._gleak_init_max,
                        shape=(self.hidden_size,),
                    )
                else:
                    self.gleak = mx.ones((self.hidden_size,)) * self._gleak_init_min
        else:
            self.gleak = mx.zeros((self.hidden_size,)) + self._fix_gleak

        if self._fix_cm is None:
            if not hasattr(self, "cm_t") or self.cm_t.shape != (self.hidden_size,):
                if self._cm_init_max > self._cm_init_min:
                    self.cm_t = mx.random.uniform(
                        low=self._cm_init_min,
                        high=self._cm_init_max,
                        shape=(self.hidden_size,),
                    )
                else:
                    self.cm_t = mx.ones((self.hidden_size,)) * self._cm_init_min
        else:
            self.cm_t = mx.zeros((self.hidden_size,)) + self._fix_cm

        self._ltc_params_built_for = key

    # --------------------------------------------------------------------- #
    # Forward pass helpers
    # --------------------------------------------------------------------- #
    def apply_constraints(self) -> None:
        """Clamp parameters to remain within biologically-plausible ranges."""
        if hasattr(self, "cm_t"):
            self.cm_t = mx.clip(self.cm_t, self._cm_min_value, self._cm_max_value)
        if hasattr(self, "gleak"):
            self.gleak = mx.clip(self.gleak, self._gleak_min_value, self._gleak_max_value)
        if hasattr(self, "W"):
            self.W = mx.clip(self.W, self._w_min_value, self._w_max_value)
        if hasattr(self, "sensory_W"):
            self.sensory_W = mx.clip(self.sensory_W, self._w_min_value, self._w_max_value)

    def _map_inputs(self, inputs: mx.array) -> mx.array:
        if self._input_mapping is MappingType.IDENTITY:
            return inputs
        mapped = inputs
        if self._input_mapping in (MappingType.LINEAR, MappingType.AFFINE):
            mapped = mapped * self.input_w
        if self._input_mapping is MappingType.AFFINE:
            mapped = mapped + self.input_b
        return mapped

    def _sigmoid(self, potentials: mx.array, mu: mx.array, sigma: mx.array) -> mx.array:
        potentials = mx.expand_dims(potentials, axis=-1)  # [batch, features, 1]
        mu = mx.expand_dims(mu, axis=0)  # [1, features, units]
        sigma = mx.expand_dims(sigma, axis=0)
        return nn.sigmoid(sigma * (potentials - mu))

    def _semi_implicit_step(
        self,
        inputs: mx.array,
        state: mx.array,
        sensory_terms: Tuple[mx.array, mx.array],
    ) -> mx.array:
        w_num_sensory, w_den_sensory = sensory_terms
        v_pre = state
        for _ in range(self.ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            w_numerator = mx.sum(rev_activation, axis=1) + w_num_sensory
            w_denominator = mx.sum(w_activation, axis=1) + w_den_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator
            v_pre = numerator / denominator
        return v_pre

    def _explicit_step(
        self,
        inputs: mx.array,
        state: mx.array,
        sensory_terms: Tuple[mx.array, mx.array],
    ) -> mx.array:
        v_pre = state
        for _ in range(self.ode_solver_unfolds):
            f_prime = self._f_prime(inputs, v_pre, sensory_terms)
            v_pre = v_pre + 0.1 * f_prime
        return v_pre

    def _runge_kutta_step(
        self,
        inputs: mx.array,
        state: mx.array,
        sensory_terms: Tuple[mx.array, mx.array],
    ) -> mx.array:
        h = 0.1
        v_pre = state
        for _ in range(self.ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, v_pre, sensory_terms)
            k2 = h * self._f_prime(inputs, v_pre + 0.5 * k1, sensory_terms)
            k3 = h * self._f_prime(inputs, v_pre + 0.5 * k2, sensory_terms)
            k4 = h * self._f_prime(inputs, v_pre + k3, sensory_terms)
            v_pre = v_pre + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return v_pre

    def _f_prime(
        self,
        inputs: mx.array,
        state: mx.array,
        sensory_terms: Tuple[mx.array, mx.array],
    ) -> mx.array:
        w_num_sensory, w_den_sensory = sensory_terms

        w_activation = self.W * self._sigmoid(state, self.mu, self.sigma)
        w_reduced_synapse = mx.sum(w_activation, axis=1)

        sensory_activation = self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_w_activation = sensory_activation * self.sensory_W
        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = (
            mx.sum(sensory_in, axis=1)
            - state * w_den_sensory
            + mx.sum(synapse_in, axis=1)
            - state * w_reduced_synapse
        )
        return (self.gleak * (self.vleak - state) + sum_in) / self.cm_t

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def __call__(self, x: mx.array, state: mx.array, time: float = 1.0) -> Tuple[mx.array, mx.array]:
        self._ensure_parameters(x.shape[-1])

        mapped_inputs = self._map_inputs(x)

        sensory_activation = self._sigmoid(mapped_inputs, self.sensory_mu, self.sensory_sigma)
        sensory_w_activation = sensory_activation * self.sensory_W
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_num_sensory = mx.sum(sensory_rev_activation, axis=1)
        w_den_sensory = mx.sum(sensory_w_activation, axis=1)
        sensory_terms = (w_num_sensory, w_den_sensory)

        if self._solver is ODESolver.SEMI_IMPLICIT:
            new_state = self._semi_implicit_step(mapped_inputs, state, sensory_terms)
        elif self._solver is ODESolver.EXPLICIT:
            new_state = self._explicit_step(mapped_inputs, state, sensory_terms)
        else:
            new_state = self._runge_kutta_step(mapped_inputs, state, sensory_terms)

        output = new_state
        if self.wiring.output_dim != self.hidden_size:
            output = self.output_proj(output)
        return output, new_state

    def _build_from_config(self, config: Dict[str, Any]) -> None:
        solver = config.get("solver")
        if solver:
            self.set_solver(solver)
        mapping = config.get("input_mapping")
        if mapping:
            self.set_input_mapping(mapping)
        input_dim = config.get("input_dim")
        if input_dim is None:
            wiring_cfg = config.get("wiring", {})
            input_dim = wiring_cfg.get("input_dim", self.input_size)
        if input_dim is not None:
            self._ensure_parameters(input_dim)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["config"]["solver"] = self._solver.value
        state["config"]["input_mapping"] = self._input_mapping.value
        return state
