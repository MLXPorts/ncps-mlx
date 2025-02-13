"""Neural Circuit Policy wiring patterns implemented for MLX."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


class Wiring(nn.Module):
    """Base class for neural wiring patterns."""
    
    def __init__(self, units: int):
        super().__init__()
        self.units = units
        # Use float32 for MLX compatibility
        self.adjacency_matrix = mx.zeros((units, units), dtype=mx.float32)
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None
        self.state_size = units

    @property
    def num_layers(self) -> int:
        """Number of layers in the wiring."""
        return 1

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """Get neurons belonging to a specific layer."""
        return list(range(self.units))

    def is_built(self) -> bool:
        """Check if wiring is built."""
        return self.input_dim is not None

    def build(self, input_dim: int):
        """Build the wiring pattern."""
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Conflicting input dimensions. Expected {self.input_dim}, got {input_dim}"
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self) -> mx.array:
        """Initialize reversal potentials for internal connections."""
        return self.adjacency_matrix

    def sensory_erev_initializer(self) -> mx.array:
        """Initialize reversal potentials for sensory connections."""
        return self.sensory_adjacency_matrix

    def set_input_dim(self, input_dim: int):
        """Set input dimension and initialize sensory matrix."""
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = mx.zeros(
            (input_dim, self.units), dtype=mx.float32
        )

    def set_output_dim(self, output_dim: int):
        """Set output dimension."""
        self.output_dim = output_dim

    def get_type_of_neuron(self, neuron_id: int) -> str:
        """Get the type of a neuron (motor/inter)."""
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src: int, dest: int, polarity: int):
        """Add a synapse between internal neurons."""
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid source neuron {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in [-1, 1]:
            raise ValueError(f"Invalid polarity {polarity}")
        
        # Convert to numpy for modification then back to MLX
        adj_matrix = self.adjacency_matrix.tolist()
        adj_matrix[src][dest] = float(polarity)  # Convert to float
        self.adjacency_matrix = mx.array(adj_matrix, dtype=mx.float32)

    def add_sensory_synapse(self, src: int, dest: int, polarity: int):
        """Add a synapse from sensory input to internal neuron."""
        if not self.is_built():
            raise ValueError("Cannot add sensory synapses before build()")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"Invalid source sensory neuron {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in [-1, 1]:
            raise ValueError(f"Invalid polarity {polarity}")
        
        # Convert to numpy for modification then back to MLX
        sensory_matrix = self.sensory_adjacency_matrix.tolist()
        sensory_matrix[src][dest] = float(polarity)  # Convert to float
        self.sensory_adjacency_matrix = mx.array(sensory_matrix, dtype=mx.float32)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            "units": self.units,
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix.tolist() if self.sensory_adjacency_matrix is not None else None,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Wiring':
        """Create wiring from configuration."""
        wiring = cls(config["units"])
        if "adjacency_matrix" in config and config["adjacency_matrix"] is not None:
            wiring.adjacency_matrix = mx.array(config["adjacency_matrix"])
        if "sensory_adjacency_matrix" in config and config["sensory_adjacency_matrix"] is not None:
            wiring.sensory_adjacency_matrix = mx.array(config["sensory_adjacency_matrix"])
        if "input_dim" in config:
            wiring.input_dim = config["input_dim"]
        if "output_dim" in config:
            wiring.output_dim = config["output_dim"]
        return wiring

    @property
    def synapse_count(self) -> int:
        """Count internal synapses."""
        return int(mx.sum(mx.abs(self.adjacency_matrix)))

    @property
    def sensory_synapse_count(self) -> int:
        """Count sensory synapses."""
        return int(mx.sum(mx.abs(self.sensory_adjacency_matrix)))


class FullyConnected(Wiring):
    """Fully connected wiring pattern."""
    
    def __init__(
        self, 
        units: int, 
        output_dim: Optional[int] = None,
        erev_init_seed: int = 1111,
        self_connections: bool = True
    ):
        super().__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        self._erev_init_seed = erev_init_seed
        
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                polarity = self._rng.choice([-1, 1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_dim: int):
        """Build fully connected sensory synapses."""
        super().build(input_dim)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "erev_init_seed": self._erev_init_seed,
            "self_connections": self.self_connections
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FullyConnected':
        """Create from configuration."""
        wiring = cls(
            units=config["units"],
            output_dim=config["output_dim"],
            erev_init_seed=config["erev_init_seed"],
            self_connections=config["self_connections"]
        )
        if config["input_dim"] is not None:
            wiring.build(config["input_dim"])
        return wiring


class Random(Wiring):
    """Random sparse wiring pattern."""
    
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        sparsity_level: float = 0.0,
        random_seed: int = 1111
    ):
        super().__init__(units)
        if output_dim is None:
            output_dim = units
        self.set_output_dim(output_dim)
        self.sparsity_level = sparsity_level

        if not 0.0 <= sparsity_level < 1.0:
            raise ValueError(f"Invalid sparsity level {sparsity_level}")
            
        self._rng = np.random.default_rng(random_seed)
        self._random_seed = random_seed

        number_of_synapses = int(np.round(units * units * (1 - sparsity_level)))
        all_synapses = [(src, dest) for src in range(units) for dest in range(units)]
        
        used_synapses = self._rng.choice(all_synapses, size=number_of_synapses, replace=False)
        for src, dest in used_synapses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_synapse(src, dest, polarity)

    def build(self, input_dim: int):
        """Build random sensory connections."""
        super().build(input_dim)
        number_of_sensory_synapses = int(
            np.round(self.input_dim * self.units * (1 - self.sparsity_level))
        )
        all_sensory_synapses = [
            (src, dest) 
            for src in range(self.input_dim) 
            for dest in range(self.units)
        ]
        
        used_sensory_synapses = self._rng.choice(
            all_sensory_synapses, 
            size=number_of_sensory_synapses, 
            replace=False
        )
        for src, dest in used_sensory_synapses:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "sparsity_level": self.sparsity_level,
            "random_seed": self._random_seed
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Random':
        """Create from configuration."""
        wiring = cls(
            units=config["units"],
            output_dim=config["output_dim"],
            sparsity_level=config["sparsity_level"],
            random_seed=config["random_seed"]
        )
        if config["input_dim"] is not None:
            wiring.build(config["input_dim"])
        return wiring


class NCP(Wiring):
    """Neural Circuit Policy wiring pattern."""
    
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_fanout: int,
        inter_fanout: int,
        recurrent_command_synapses: int,
        motor_fanin: int,
        seed: int = 22222,
    ):
        super().__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin
        self._seed = seed

        # Neuron IDs: [0..motor ... command ... inter]
        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(
            range(
                self._num_motor_neurons,
                self._num_motor_neurons + self._num_command_neurons,
            )
        )
        self._inter_neurons = list(
            range(
                self._num_motor_neurons + self._num_command_neurons,
                self._num_motor_neurons + self._num_command_neurons + self._num_inter_neurons,
            )
        )

        # Validate parameters
        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(
                f"Motor fanin {self._motor_fanin} exceeds command neurons {self._num_command_neurons}"
            )
        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(
                f"Inter fanout {self._inter_fanout} exceeds command neurons {self._num_command_neurons}"
            )

    @property
    def num_layers(self) -> int:
        """Number of layers in NCP."""
        return 3

    def get_neurons_of_layer(self, layer_id: int) -> List[int]:
        """Get neurons for each NCP layer."""
        if layer_id == 0:
            return self._inter_neurons
        elif layer_id == 1:
            return self._command_neurons
        elif layer_id == 2:
            return self._motor_neurons
        raise ValueError(f"Invalid layer {layer_id}")

    def get_type_of_neuron(self, neuron_id: int) -> str:
        """Get neuron type (motor/command/inter)."""
        if neuron_id < self._num_motor_neurons:
            return "motor"
        if neuron_id < self._num_motor_neurons + self._num_command_neurons:
            return "command"
        return "inter"

    def build(self, input_dim: int):
        """Build complete NCP wiring."""
        super().build(input_dim)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(self._num_sensory_neurons))

        # Validate sensory fanout
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(
                f"Sensory fanout {self._sensory_fanout} exceeds inter neurons {self._num_inter_neurons}"
            )

        # Build each layer
        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command_to_motor_layer()

    def _build_sensory_to_inter_layer(self):
        """Build connections from sensory to inter neurons."""
        unreachable = set(self._inter_neurons)
        
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                self._inter_neurons, 
                size=self._sensory_fanout, 
                replace=False
            ):
                unreachable.discard(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

        # Connect any unreached neurons
        if unreachable:
            mean_fanin = max(
                int(self._num_sensory_neurons * self._sensory_fanout / self._num_inter_neurons),
                1
            )
            for dest in unreachable:
                for src in self._rng.choice(
                    self._sensory_neurons,
                    size=min(mean_fanin, self._num_sensory_neurons),
                    replace=False
                ):
                    polarity = self._rng.choice([-1, 1])
                    self.add_sensory_synapse(src, dest, polarity)

    def _build_inter_to_command_layer(self):
        """Build connections from inter to command neurons."""
        unreachable = set(self._command_neurons)
        
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                self._command_neurons,
                size=self._inter_fanout,
                replace=False
            ):
                unreachable.discard(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # Connect any unreached neurons
        if unreachable:
            mean_fanin = max(
                int(self._num_inter_neurons * self._inter_fanout / self._num_command_neurons),
                1
            )
            for dest in unreachable:
                for src in self._rng.choice(
                    self._inter_neurons,
                    size=min(mean_fanin, self._num_inter_neurons),
                    replace=False
                ):
                    polarity = self._rng.choice([-1, 1])
                    self.add_synapse(src, dest, polarity)

    def _build_recurrent_command_layer(self):
        """Build recurrent connections in command layer."""
        for _ in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def _build_command_to_motor_layer(self):
        """Build connections from command to motor neurons."""
        unreachable = set(self._command_neurons)
        
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                self._command_neurons,
                size=self._motor_fanin,
                replace=False
            ):
                unreachable.discard(src)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # Connect any unreached neurons
        if unreachable:
            mean_fanout = max(
                int(self._num_motor_neurons * self._motor_fanin / self._num_command_neurons),
                1
            )
            for src in unreachable:
                for dest in self._rng.choice(
                    self._motor_neurons,
                    size=min(mean_fanout, self._num_motor_neurons),
                    replace=False
                ):
                    polarity = self._rng.choice([-1, 1])
                    self.add_synapse(src, dest, polarity)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "inter_neurons": self._num_inter_neurons,
            "command_neurons": self._num_command_neurons,
            "motor_neurons": self._num_motor_neurons,
            "sensory_fanout": self._sensory_fanout,
            "inter_fanout": self._inter_fanout,
            "recurrent_command_synapses": self._recurrent_command_synapses,
            "motor_fanin": self._motor_fanin,
            "seed": self._seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCP':
        """Create from configuration."""
        wiring = cls(
            inter_neurons=config["inter_neurons"],
            command_neurons=config["command_neurons"],
            motor_neurons=config["motor_neurons"],
            sensory_fanout=config["sensory_fanout"],
            inter_fanout=config["inter_fanout"],
            recurrent_command_synapses=config["recurrent_command_synapses"],
            motor_fanin=config["motor_fanin"],
            seed=config["seed"]
        )
        if config["input_dim"] is not None:
            wiring.build(config["input_dim"])
        return wiring


class AutoNCP(NCP):
    """Automatic NCP wiring with simplified parameters."""
    
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: int = 22222,
    ):
        if output_size >= units - 2:
            raise ValueError(
                f"Output size {output_size} must be less than units-2 ({units-2})"
            )
        if not 0.1 <= sparsity_level <= 0.9:
            raise ValueError(
                f"Sparsity level must be between 0.1 and 0.9 (got {sparsity_level})"
            )

        self._output_size = output_size
        self._sparsity_level = sparsity_level
        self._seed = seed
        
        # Calculate architecture
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons

        # Calculate connectivity
        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(int(command_neurons * density_level * 2), 1)
        motor_fanin = max(int(command_neurons * density_level), 1)

        super().__init__(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=output_size,
            sensory_fanout=sensory_fanout,
            inter_fanout=inter_fanout,
            recurrent_command_synapses=recurrent_command_synapses,
            motor_fanin=motor_fanin,
            seed=seed,
        )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "output_size": self._output_size,
            "sparsity_level": self._sparsity_level,
            "seed": self._seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoNCP':
        """Create from configuration."""
        wiring = cls(
            units=config["units"],
            output_size=config["output_size"],
            sparsity_level=config["sparsity_level"],
            seed=config["seed"]
        )
        if config["input_dim"] is not None:
            wiring.build(config["input_dim"])
        return wiring
