from abc import abstractmethod
from typing import Any, Optional, List, Union, Dict, Tuple

from ncps.layers.base.layer import Layer

class BackboneLayerCellBase(Layer):
    """Base class for liquid neural network cells.
    
    This class defines the interface for liquid neuron implementations across
    different backends. Each backend should provide a concrete implementation
    of the abstract methods.
    
    Args:
        wiring: Neural circuit wiring configuration
        activation: Activation function to use
        backbone_units: Number of units in backbone layers
        backbone_layers: Number of backbone layers
        backbone_dropout: Dropout rate for backbone layers
        solver_method: ODE solver method to use
        dtype: Data type for layer weights
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        wiring: Any,
        activation: str = "tanh",
        backbone_units: Optional[Union[int, List[int]]] = None,
        backbone_layers: int = 0,
        backbone_dropout: float = 0.0,
        solver_method: str = "rk4",
        dtype: str = "float32",
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        
        # Store wiring configuration
        self.wiring = wiring
        self.units = wiring.units
        self.state_size = wiring.units
        self.output_size = wiring.output_dim if hasattr(wiring, 'output_dim') else wiring.units
        
        # Store activation configuration
        self.activation_name = activation
        self.activation = None  # Set by backend implementation
        
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
        
        # Set by build
        self.input_size = None
        self.backbone_output_dim = None
    
    @abstractmethod
    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Build the cell weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        pass
    
    @abstractmethod
    def get_initial_state(
        self,
        batch_size: Optional[int] = None,
        dtype: Optional[Any] = None
    ) -> List[Any]:
        """Get initial state for RNN.
        
        Args:
            batch_size: Optional batch size for state shape
            dtype: Optional dtype for state tensors
            
        Returns:
            List containing initial state tensor
        """
        pass
    
    @abstractmethod
    def __call__(
        self,
        inputs: Any,
        states: List[Any],
        training: Optional[bool] = None,
        **kwargs
    ) -> Tuple[Any, List[Any]]:
        """Process one timestep.
        
        Args:
            inputs: Input tensor
            states: List of state tensors
            training: Whether in training mode
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (output tensor, list of new state tensors)
        """
        pass
    
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
    def from_config(cls, config: Dict[str, Any]) -> 'BackboneLayerCellBase':
        """Create layer from configuration."""
        # Handle wiring configuration
        wiring_config = config.pop("wiring")
        wiring_class = cls._get_wiring_class()
        wiring = wiring_class(**wiring_config)
        
        return cls(wiring=wiring, **config)
    
    @staticmethod
    @abstractmethod
    def _get_wiring_class() -> Any:
        """Get wiring class for configuration.
        
        This should be implemented by subclasses to return their wiring class.
        
        Returns:
            The wiring class to use
        """
        pass