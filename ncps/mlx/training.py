"""Enhanced training utilities for MLX Neural Circuit Policies."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Optional, Dict, Any, Callable, Union, List, Tuple
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for enhanced LTC training."""
    target_accuracy: float = 95.0
    max_epochs: int = 1000
    patience: int = 10
    weight_clip: float = 0.5
    bias_clip: float = 0.1
    default_clip: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 0.01
    min_learning_rate: float = 0.0001  # Minimum learning rate for warmup
    warmup_epochs: int = 100  # Number of epochs for learning rate warmup
    noise_scale: float = 0.01  # Scale of noise to add during training
    noise_decay: float = 0.995  # Decay rate for noise scale
    momentum: float = 0.9  # Momentum for gradient updates
    grad_momentum: float = 0.1  # Momentum for gradient accumulation
    epsilon: float = 1e-8  # Small constant for numerical stability


class EnhancedLTCTrainer:
    """Enhanced trainer incorporating LTC insights."""
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], TrainingConfig]] = None
    ):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        if config is None:
            self.config = TrainingConfig()
        elif isinstance(config, dict):
            self.config = TrainingConfig(**config)
        else:
            self.config = config
            
        self.optimizer = None
        self.best_accuracy = 0.0
        self.best_model_state = None
        self.current_epoch = 0
        self.current_noise_scale = self.config.noise_scale
        self.grad_momentum = None
        
    def _get_learning_rate(self) -> float:
        """Get current learning rate with warmup schedule."""
        if self.current_epoch < self.config.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.config.warmup_epochs
            return self.config.min_learning_rate + alpha * (self.config.learning_rate - self.config.min_learning_rate)
        return self.config.learning_rate
        
    def _init_optimizer(self, model: nn.Module):
        """Initialize the optimizer with proper configuration."""
        self.optimizer = optim.Adam(
            learning_rate=self._get_learning_rate(),
            betas=(self.config.momentum, 0.999),  # Higher momentum
            eps=self.config.epsilon  # Use config epsilon
        )
        
    def process_gradients(
        self,
        grads: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process gradients with parameter-specific handling.
        
        Args:
            grads: Dictionary of parameter gradients
            
        Returns:
            Processed gradients
        """
        def process_grad(grad, param_name=""):
            if isinstance(grad, dict):
                return {k: process_grad(v, f"{param_name}.{k}" if param_name else k) 
                       for k, v in grad.items()}
            elif isinstance(grad, list):
                return [process_grad(g, f"{param_name}[{i}]") 
                       for i, g in enumerate(grad)]
            elif isinstance(grad, mx.array):
                # Add noise to help escape local minima
                noise = self.current_noise_scale * mx.random.normal(grad.shape)
                grad = grad + noise
                
                # Apply gradient momentum
                if self.grad_momentum is None:
                    self.grad_momentum = {param_name: mx.zeros_like(grad)}
                elif param_name not in self.grad_momentum:
                    self.grad_momentum[param_name] = mx.zeros_like(grad)
                
                momentum = self.grad_momentum[param_name]
                grad = self.config.grad_momentum * momentum + (1 - self.config.grad_momentum) * grad
                self.grad_momentum[param_name] = grad
                
                # Scale gradients based on learning rate warmup
                scale = self.config.learning_rate / self._get_learning_rate()
                grad = grad * scale
                
                # Clip gradients based on parameter type
                if 'weight' in param_name or 'kernel' in param_name:
                    clip_value = self.config.weight_clip
                elif 'bias' in param_name:
                    clip_value = self.config.bias_clip
                else:
                    clip_value = self.config.default_clip
                return mx.clip(grad, -clip_value, clip_value)
            else:
                return grad

        # Process gradients recursively
        clipped_grads = process_grad(grads)
        
        # Compute global norm across all array gradients
        def get_array_grads(d):
            if isinstance(d, dict):
                return [g for v in d.values() for g in get_array_grads(v)]
            elif isinstance(d, list):
                return [g for v in d for g in get_array_grads(v)]
            elif isinstance(d, mx.array):
                return [d]
            else:
                return []

        array_grads = get_array_grads(clipped_grads)
        if array_grads:
            grad_norm = mx.sqrt(sum(mx.sum(g * g) for g in array_grads))
            if grad_norm > self.config.max_grad_norm:
                scale = self.config.max_grad_norm / (grad_norm + self.config.epsilon)
                
                def scale_grads(d):
                    if isinstance(d, dict):
                        return {k: scale_grads(v) for k, v in d.items()}
                    elif isinstance(d, list):
                        return [scale_grads(g) for g in d]
                    elif isinstance(d, mx.array):
                        return d * scale
                    else:
                        return d
                
                clipped_grads = scale_grads(clipped_grads)
        
        return clipped_grads
        
    def calculate_accuracy(
        self,
        model: nn.Module,
        data_x: mx.array,
        data_y: mx.array,
        tolerance: float = 0.05
    ) -> float:
        """Calculate model accuracy within tolerance.
        
        Args:
            model: The model to evaluate
            data_x: Input data
            data_y: Target data
            tolerance: Error tolerance (5% by default)
            
        Returns:
            Accuracy as percentage
        """
        predictions = model(data_x)
        
        # Calculate percentage errors
        errors = mx.abs((predictions - data_y) / (data_y + self.config.epsilon))
        accurate = mx.mean(errors <= tolerance)
        
        return float(accurate.item() * 100)
        
    def train_step(
        self,
        model: nn.Module,
        data_x: mx.array,
        data_y: mx.array,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[float, float]:
        """Perform one training step.
        
        Args:
            model: The model to train
            data_x: Input data
            data_y: Target data
            loss_fn: Optional custom loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if loss_fn is None:
            loss_fn = lambda pred, target: mx.mean((pred - target) ** 2)

        def loss_fn_wrapper(model_params):
            """Compute loss for given parameters."""
            model.update(model_params)
            pred = model(data_x)
            return loss_fn(pred, data_y)

        # Get trainable parameters
        params = model.trainable_parameters()
        if not params:
            raise ValueError("Model has no trainable parameters")

        # Update optimizer learning rate
        self.optimizer.learning_rate = self._get_learning_rate()

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn_wrapper)(params)
        
        # Process gradients
        processed_grads = self.process_gradients(grads)
        
        # Update model
        self.optimizer.update(model, processed_grads)
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(model, data_x, data_y)
        
        # Decay noise scale
        self.current_noise_scale *= self.config.noise_decay
        
        return float(loss.item()), accuracy
        
    def train_with_accuracy_target(
        self,
        model: nn.Module,
        data_x: mx.array,
        data_y: mx.array,
        loss_fn: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train until target accuracy is reached.
        
        Args:
            model: The model to train
            data_x: Input data
            data_y: Target data
            loss_fn: Optional custom loss function
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        if self.optimizer is None:
            self._init_optimizer(model)
            
        history = {
            'loss': [],
            'accuracy': [],
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'converged': False
        }
        
        patience_counter = 0
        self.current_epoch = 0
        self.current_noise_scale = self.config.noise_scale
        self.grad_momentum = None
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training step
            loss, accuracy = self.train_step(model, data_x, data_y, loss_fn)
            
            # Update history
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            # Check for improvement
            if accuracy > history['best_accuracy']:
                history['best_accuracy'] = accuracy
                history['best_epoch'] = epoch
                patience_counter = 0
                # Save best model state
                self.best_model_state = {k: v.copy() for k, v in model.parameters().items()}
            else:
                patience_counter += 1
                
            if verbose and (epoch + 1) % 10 == 0:
                lr = self._get_learning_rate()
                print(f"Epoch {epoch + 1}, Loss: {loss:.6f}, Accuracy: {accuracy:.2f}%, LR: {lr:.6f}, Noise: {self.current_noise_scale:.6f}")
                
            # Check stopping conditions
            if accuracy >= self.config.target_accuracy:
                if verbose:
                    print(f"\nTarget accuracy {self.config.target_accuracy}% reached at epoch {epoch + 1}")
                history['converged'] = True
                break
                
            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
                
            # Force evaluation
            mx.eval(model.parameters(), self.optimizer.state)
                
        # Restore best model
        if self.best_model_state is not None:
            model.update(self.best_model_state)
            
        return history
