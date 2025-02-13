"""Neural Circuit Policy wiring patterns implemented for Keras."""

import keras
from keras import ops
import numpy as np
from typing import Optional, List, Dict, Any, Union


@keras.saving.register_keras_serializable(package="ncps")
class Wiring(keras.layers.Layer):
    """Base class for neural wiring patterns."""
    
    def __init__(self, units: int):
        """Initialize wiring pattern.
        
        Args:
            units: Number of neurons in the circuit
        """
        super().__init__()
        self.units = units
        self.adjacency_matrix = None
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None
    
    def build(self, input_dim: int):
        """Build wiring pattern.
        
        Args:
            input_dim: Input dimension
        """
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Conflicting input dimensions. Expected {self.input_dim}, got {input_dim}"
            )
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = ops.zeros((input_dim, self.units))
    
    def add_synapse(self, src: int, dest: int, polarity: int = 1):
        """Add synapse between neurons.
        
        Args:
            src: Source neuron index
            dest: Destination neuron index
            polarity: Synapse polarity (1 or -1)
        """
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid source neuron {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in [-1, 1]:
            raise ValueError(f"Invalid polarity {polarity}")
        
        if self.adjacency_matrix is None:
            self.adjacency_matrix = ops.zeros((self.units, self.units))
        
        # Update adjacency matrix
        matrix = ops.convert_to_numpy(self.adjacency_matrix)
        matrix[src, dest] = float(polarity)
        self.adjacency_matrix = ops.convert_to_tensor(matrix)
    
    def add_sensory_synapse(self, src: int, dest: int, polarity: int = 1):
        """Add synapse from sensory input to neuron.
        
        Args:
            src: Source sensory input index
            dest: Destination neuron index
            polarity: Synapse polarity (1 or -1)
        """
        if not self.is_built():
            raise ValueError("Cannot add sensory synapses before build()")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"Invalid source sensory neuron {src}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid destination neuron {dest}")
        if polarity not in [-1, 1]:
            raise ValueError(f"Invalid polarity {polarity}")
        
        # Update sensory adjacency matrix
        matrix = ops.convert_to_numpy(self.sensory_adjacency_matrix)
        matrix[src, dest] = float(polarity)
        self.sensory_adjacency_matrix = ops.convert_to_tensor(matrix)
    
    def is_built(self) -> bool:
        """Check if wiring is built."""
        return self.input_dim is not None
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'class_name': self.__class__.__name__,
            'config': {
                'units': self.units,
                'adjacency_matrix': ops.convert_to_numpy(self.adjacency_matrix).tolist() if self.adjacency_matrix is not None else None,
                'sensory_adjacency_matrix': ops.convert_to_numpy(self.sensory_adjacency_matrix).tolist() if self.sensory_adjacency_matrix is not None else None,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
            }
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Wiring':
        """Create from configuration."""
        # Extract config
        if 'config' in config:
            config = config['config']
        
        # Create instance
        instance = cls(config['units'])
        if config['adjacency_matrix'] is not None:
            instance.adjacency_matrix = ops.convert_to_tensor(config['adjacency_matrix'])
        if config['sensory_adjacency_matrix'] is not None:
            instance.sensory_adjacency_matrix = ops.convert_to_tensor(config['sensory_adjacency_matrix'])
        instance.input_dim = config['input_dim']
        instance.output_dim = config['output_dim']
        return instance


@keras.saving.register_keras_serializable(package="ncps")
class FullyConnected(Wiring):
    """Fully connected wiring pattern."""
    
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        self_connections: bool = True,
        random_seed: int = 1111
    ):
        """Initialize fully connected wiring.
        
        Args:
            units: Number of neurons
            output_dim: Output dimension (default: units)
            self_connections: Allow self connections (default: True)
            random_seed: Random seed for initialization
        """
        super().__init__(units)
        self.output_dim = output_dim or units
        self.self_connections = self_connections
        self._rng = np.random.RandomState(random_seed)
        
        # Initialize connections
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                polarity = self._rng.choice([-1, 1, 1])  # Bias towards excitatory
                self.add_synapse(src, dest, polarity)
    
    def build(self, input_dim: int):
        """Build fully connected sensory synapses."""
        super().build(input_dim)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])  # Bias towards excitatory
                self.add_sensory_synapse(src, dest, polarity)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config['config'].update({
            'output_dim': self.output_dim,
            'self_connections': self.self_connections,
            'random_seed': self._rng.get_state()[1][0],
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FullyConnected':
        """Create from configuration."""
        if 'config' in config:
            config = config['config']
        return cls(
            units=config['units'],
            output_dim=config['output_dim'],
            self_connections=config['self_connections'],
            random_seed=config['random_seed']
        )


@keras.saving.register_keras_serializable(package="ncps")
class Random(Wiring):
    """Random sparse wiring pattern."""
    
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        sparsity_level: float = 0.5,
        random_seed: int = 1111
    ):
        """Initialize random sparse wiring.
        
        Args:
            units: Number of neurons
            output_dim: Output dimension (default: units)
            sparsity_level: Connection sparsity (0.0 to 1.0)
            random_seed: Random seed for initialization
        """
        if not 0.0 <= sparsity_level < 1.0:
            raise ValueError(f"Invalid sparsity level {sparsity_level}")
        
        super().__init__(units)
        self.output_dim = output_dim or units
        self.sparsity_level = sparsity_level
        self._rng = np.random.RandomState(random_seed)
        
        # Initialize sparse connections
        num_synapses = int(np.round(units * units * (1 - sparsity_level)))
        all_synapses = [(i, j) for i in range(units) for j in range(units)]
        used_synapses = self._rng.choice(len(all_synapses), size=num_synapses, replace=False)
        
        for idx in used_synapses:
            src, dest = all_synapses[idx]
            polarity = self._rng.choice([-1, 1, 1])  # Bias towards excitatory
            self.add_synapse(src, dest, polarity)
    
    def build(self, input_dim: int):
        """Build random sensory synapses."""
        super().build(input_dim)
        num_sensory_synapses = int(
            np.round(self.input_dim * self.units * (1 - self.sparsity_level))
        )
        all_sensory_synapses = [
            (i, j) 
            for i in range(self.input_dim) 
            for j in range(self.units)
        ]
        used_sensory_synapses = self._rng.choice(
            len(all_sensory_synapses),
            size=num_sensory_synapses,
            replace=False
        )
        
        for idx in used_sensory_synapses:
            src, dest = all_sensory_synapses[idx]
            polarity = self._rng.choice([-1, 1, 1])  # Bias towards excitatory
            self.add_sensory_synapse(src, dest, polarity)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config['config'].update({
            'output_dim': self.output_dim,
            'sparsity_level': self.sparsity_level,
            'random_seed': self._rng.get_state()[1][0],
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Random':
        """Create from configuration."""
        if 'config' in config:
            config = config['config']
        return cls(
            units=config['units'],
            output_dim=config['output_dim'],
            sparsity_level=config['sparsity_level'],
            random_seed=config['random_seed']
        )


@keras.saving.register_keras_serializable(package="ncps")
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
        seed: int = 22222
    ):
        """Initialize NCP wiring.
        
        Args:
            inter_neurons: Number of interneurons
            command_neurons: Number of command neurons
            motor_neurons: Number of motor neurons
            sensory_fanout: Sensory neuron fanout
            inter_fanout: Interneuron fanout
            recurrent_command_synapses: Recurrent command connections
            motor_fanin: Motor neuron fanin
            seed: Random seed for initialization
        """
        super().__init__(inter_neurons + command_neurons + motor_neurons)
        self.output_dim = motor_neurons
        self._rng = np.random.RandomState(seed)
        
        # Store architecture
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin
        
        # Validate parameters
        if motor_fanin > command_neurons:
            raise ValueError(
                f"Motor fanin {motor_fanin} exceeds command neurons {command_neurons}"
            )
        if inter_fanout > command_neurons:
            raise ValueError(
                f"Inter fanout {inter_fanout} exceeds command neurons {command_neurons}"
            )
        
        # Initialize layers
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
    
    def build(self, input_dim: int):
        """Build complete NCP wiring."""
        super().build(input_dim)
        
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
        
        for src in range(self.input_dim):
            dest_neurons = self._rng.choice(
                self._inter_neurons,
                size=self._sensory_fanout,
                replace=False
            )
            for dest in dest_neurons:
                unreachable.discard(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)
        
        # Connect any unreached neurons
        if unreachable:
            mean_fanin = max(
                int(self.input_dim * self._sensory_fanout / self._num_inter_neurons),
                1
            )
            for dest in unreachable:
                src_neurons = self._rng.choice(
                    range(self.input_dim),
                    size=min(mean_fanin, self.input_dim),
                    replace=False
                )
                for src in src_neurons:
                    polarity = self._rng.choice([-1, 1])
                    self.add_sensory_synapse(src, dest, polarity)
    
    def _build_inter_to_command_layer(self):
        """Build connections from inter to command neurons."""
        unreachable = set(self._command_neurons)
        
        for src in self._inter_neurons:
            dest_neurons = self._rng.choice(
                self._command_neurons,
                size=self._inter_fanout,
                replace=False
            )
            for dest in dest_neurons:
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
                src_neurons = self._rng.choice(
                    self._inter_neurons,
                    size=min(mean_fanin, self._num_inter_neurons),
                    replace=False
                )
                for src in src_neurons:
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
            src_neurons = self._rng.choice(
                self._command_neurons,
                size=self._motor_fanin,
                replace=False
            )
            for src in src_neurons:
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
                dest_neurons = self._rng.choice(
                    self._motor_neurons,
                    size=min(mean_fanout, self._num_motor_neurons),
                    replace=False
                )
                for dest in dest_neurons:
                    polarity = self._rng.choice([-1, 1])
                    self.add_synapse(src, dest, polarity)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config['config'].update({
            'inter_neurons': self._num_inter_neurons,
            'command_neurons': self._num_command_neurons,
            'motor_neurons': self._num_motor_neurons,
            'sensory_fanout': self._sensory_fanout,
            'inter_fanout': self._inter_fanout,
            'recurrent_command_synapses': self._recurrent_command_synapses,
            'motor_fanin': self._motor_fanin,
            'seed': self._rng.get_state()[1][0],
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCP':
        """Create from configuration."""
        if 'config' in config:
            config = config['config']
        return cls(
            inter_neurons=config['inter_neurons'],
            command_neurons=config['command_neurons'],
            motor_neurons=config['motor_neurons'],
            sensory_fanout=config['sensory_fanout'],
            inter_fanout=config['inter_fanout'],
            recurrent_command_synapses=config['recurrent_command_synapses'],
            motor_fanin=config['motor_fanin'],
            seed=config['seed']
        )


@keras.saving.register_keras_serializable(package="ncps")
class AutoNCP(NCP):
    """Automated Neural Circuit Policy wiring pattern."""
    
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: int = 22222
    ):
        """Initialize automated NCP wiring.
        
        Args:
            units: Total number of neurons
            output_size: Number of output neurons
            sparsity_level: Connection sparsity (0.0 to 1.0)
            seed: Random seed for initialization
        """
        if output_size >= units - 2:
            raise ValueError(
                f"Output size {output_size} must be less than units-2 ({units-2})"
            )
        if not 0.1 <= sparsity_level <= 0.9:
            raise ValueError(
                f"Sparsity level must be between 0.1 and 0.9 (got {sparsity_level})"
            )
        
        # Calculate architecture
        density = 1.0 - sparsity_level
        remaining = units - output_size
        command = max(int(0.4 * remaining), 1)
        inter = remaining - command
        
        super().__init__(
            inter_neurons=inter,
            command_neurons=command,
            motor_neurons=output_size,
            sensory_fanout=max(int(inter * density), 1),
            inter_fanout=max(int(command * density), 1),
            recurrent_command_synapses=max(int(command * density * 2), 1),
            motor_fanin=max(int(command * density), 1),
            seed=seed
        )
        
        # Store original parameters
        self._units = units
        self._output_size = output_size
        self._sparsity_level = sparsity_level
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config['config'].update({
            'units': self._units,
            'output_size': self._output_size,
            'sparsity_level': self._sparsity_level,
            'seed': self._rng.get_state()[1][0],
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoNCP':
        """Create from configuration."""
        if 'config' in config:
            config = config['config']
        return cls(
            units=config['units'],
            output_size=config['output_size'],
            sparsity_level=config['sparsity_level'],
            seed=config['seed']
        )