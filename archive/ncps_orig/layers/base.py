"""Base classes for liquid neural network implementations."""

import keras
from keras import ops, layers
from typing import Optional, List, Union, Dict, Any, Tuple

from .liquid_utils import TimeAwareMixin, BackboneMixin, get_activation


class BackboneLayerCell(layers.Layer, TimeAwareMixin, BackboneMixin):
    """Base class for liquid neural network cells.
    
    This serves as the foundation for all liquid neuron implementations,
    providing core functionality for time-based updates, state management,
    and feature extraction.
    
    Args:
        wiring: Neural circuit wiring configuration
        activation: Activation function to use
        backbone_units: Number of units in backbone layers
        backbone_layers: Number of backbone layers
        backbone_dropout: Dropout rate for backbone layers
        solver_method: ODE solver method to use
        **kwargs: Additional keyword arguments for the Layer base class
    """
    
    def __init__(
        self,
        wiring,
        activation: str = "tanh",
        backbone_units: Optional[Union[int, List[int]]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        solver_method: str = "rk4",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Store wiring configuration
        self.wiring = wiring
        self.units = wiring.units
        self.state_size = wiring.units
        self.output_size = wiring.output_dim if hasattr(wiring, 'output_dim') else wiring.units
        
        # Get activation function
        self.activation_name = activation
        self.activation = get_activation(activation)
        
        # Process backbone configuration
        if backbone_units is None:
            backbone_units = []
        elif isinstance(backbone_units, int):
            backbone_units = [backbone_units] * backbone_layers
        elif len(backbone_units) == 1 and backbone_layers > 1:
            backbone_units = backbone_units * backbone_layers
            
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.backbone = None
        
        # Store solver configuration
        self.solver_method = solver_method
        
        # Initialize state
        self.built = False
    
    def build(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) -> None:
        """Build the cell weights.
        
        Args:
            input_shape: Input shape tuple or list of tuples for [inputs, time]
        """
        # Handle input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
            
        # Get input dimension
        if len(input_shape) >= 2:
            input_dim = input_shape[-1]
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")
            
        # Store input size
        self.input_size = input_dim
        
        # Build backbone if needed
        if self.backbone_layers > 0:
            self.backbone = self.build_backbone(
                input_size=self.input_size + self.state_size,
                backbone_units=self.backbone_units[-1],
                backbone_layers=self.backbone_layers,
                backbone_dropout=self.backbone_dropout,
                activation=self.activation
            )
            self.backbone_output_dim = self.backbone_units[-1]
        else:
            self.backbone_output_dim = self.input_size + self.state_size
            
        # Mark as built
        self.built = True
    
    def get_initial_state(
        self,
        batch_size: Optional[int] = None,
        dtype: Optional[Any] = None
    ) -> List[keras.KerasTensor]:
        """Get initial state for RNN.
        
        Args:
            batch_size: Optional batch size for state shape
            dtype: Optional dtype for state tensors
            
        Returns:
            List containing initial state tensor
        """
        return [
            ops.zeros((batch_size, self.state_size), dtype=dtype or self.dtype)
        ]
    
    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        states: List[keras.KerasTensor],
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """Process one timestep.
        
        This should be implemented by subclasses to define their forward pass.
        
        Args:
            inputs: Input tensor or list of [input, time] tensors
            states: List of state tensors
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (output tensor, list of new state tensors)
        """
        raise NotImplementedError()
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "wiring": self.wiring.get_config(),
            "activation": self.activation_name,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout,
            "solver_method": self.solver_method,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BackboneLayerCell':
        """Create layer from configuration."""
        # Handle wiring configuration
        wiring_config = config.pop("wiring")
        wiring_class = cls._get_wiring_class()
        wiring = wiring_class(**wiring_config)
        
        return cls(wiring=wiring, **config)
    
    @staticmethod
    def _get_wiring_class():
        """Get wiring class for configuration.
        
        This should be implemented by subclasses to return their wiring class.
        """
        raise NotImplementedError()