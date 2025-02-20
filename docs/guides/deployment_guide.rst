Deployment Guide
================
================
================
================
================
================
================
================
================
================
================
================
================
================
================
===============

This guide covers best practices for deploying Neural Circuit Policies in production environments using MLX.

Model Serialization
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-----------------

Saving Models
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~

MLX models can be serialized to JSON format.

.. code-block:: python

    def save_model(model, path):
        """Save model to file."""
        state = {
            'model_state': model.state_dict(),
            'model_config': {
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                # Add other relevant config
            }
        }
        with open(path, 'w') as f:
            json.dump(state, f)

Loading Models
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~

Load saved models for inference.

.. code-block:: python

    def load_model(path):
        """Load model from file."""
        with open(path, 'r') as f:
            state = json.load(f)
            
        model = CfC(**state['model_config'])
        model.load_state_dict(state['model_state'])
        return model

Model Optimization
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
---------------

Compilation
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~~~
~~~~~~~~~

Use MLX's compilation for faster inference.

.. code-block:: python

    @mx.compile
    def optimized_inference(model, x, time_delta=None):
        return model(x, time_delta=time_delta)

Batch Processing
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~

Efficient batch processing for multiple inputs.

.. code-block:: python

    class BatchProcessor:
        def __init__(self, model, batch_size=32):
            self.model = model
            self.batch_size = batch_size
            
        def process_all(self, data):
            results = []
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                results.extend(self.model(batch))
            return results

Memory Management
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~

Optimize memory usage for production.

.. code-block:: python

    class MemoryOptimizedInference:
        def __init__(self, model):
            self.model = model
            
        def __call__(self, x):
            with mx.stream():
                result = self.model(x)
                mx.eval(result)
            return result

Serving Strategies
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
---------------

FastAPI Integration
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~

Create a REST API using FastAPI.

.. code-block:: python

    from fastapi import FastAPI
    import mlx.core as mx
    import numpy as np
    
    app = FastAPI()
    
    class ModelServer:
        def __init__(self, model_path):
            self.model = load_model(model_path)
            
        async def predict(self, data):
            x = mx.array(data)
            return self.model(x)
    
    server = ModelServer('path/to/model.json')
    
    @app.post("/predict")
    async def predict(data: dict):
        result = await server.predict(data['input'])
        return {"prediction": result.tolist()}

gRPC Service
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~~~
~~~~~~~~~~

High-performance gRPC service.

.. code-block:: python

    import grpc
    from concurrent import futures
    import prediction_pb2
    import prediction_pb2_grpc
    
    class PredictionService(prediction_pb2_grpc.PredictorServicer):
        def __init__(self, model_path):
            self.model = load_model(model_path)
            
        def Predict(self, request, context):
            input_data = np.array(request.data)
            prediction = self.model(mx.array(input_data))
            return prediction_pb2.PredictionResponse(
                prediction=prediction.tolist()
            )
    
    def serve():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        prediction_pb2_grpc.add_PredictorServicer_to_server(
            PredictionService('path/to/model.json'), server
        )
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()

Production Considerations
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
-------------------------
----------------------

Error Handling
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~

Robust error handling for production.

.. code-block:: python

    class ProductionModel:
        def __init__(self, model_path):
            self.model = load_model(model_path)
            
        def predict(self, x):
            try:
                # Input validation
                if not self._validate_input(x):
                    raise ValueError("Invalid input format")
                
                # Prediction with timeout
                with timeout(seconds=30):
                    result = self.model(x)
                
                # Output validation
                if not self._validate_output(result):
                    raise ValueError("Invalid model output")
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise

Monitoring
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~~
~~~~~~~~~

Monitor model performance in production.

.. code-block:: python

    class MonitoredModel:
        def __init__(self, model, metrics_client):
            self.model = model
            self.metrics = metrics_client
            
        def predict(self, x):
            start_time = time.time()
            try:
                result = self.model(x)
                self.metrics.increment('predictions.success')
                return result
            except Exception as e:
                self.metrics.increment('predictions.error')
                raise
            finally:
                duration = time.time() - start_time
                self.metrics.timing('prediction.duration', duration)

Scaling
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~~
~~~~~~

Strategies for scaling model serving.

.. code-block:: python

    class LoadBalancedPredictor:
        def __init__(self, model_paths, max_batch_size=32):
            self.models = [load_model(path) for path in model_paths]
            self.current_model = 0
            self.max_batch_size = max_batch_size
            
        def predict(self, x):
            # Round-robin load balancing
            model = self.models[self.current_model]
            self.current_model = (self.current_model + 1) % len(self.models)
            
            # Batch size management
            if len(x) > self.max_batch_size:
                return self._predict_large_batch(x)
            return model(x)
            
        def _predict_large_batch(self, x):
            results = []
            for i in range(0, len(x), self.max_batch_size):
                batch = x[i:i+self.max_batch_size]
                results.append(self.predict(batch))
            return mx.concatenate(results)

Deployment Environments
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
--------------------

Docker Deployment
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~

Containerize your model for deployment.

.. code-block:: dockerfile

    FROM python:3.8-slim
    
    WORKDIR /app
    
    # Install dependencies
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    # Copy model and code
    COPY model.json .
    COPY server.py .
    
    # Run the server
    CMD ["python", "server.py"]

Kubernetes Configuration
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~

Deploy on Kubernetes for scaling.

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: model-service
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: model-service
      template:
        metadata:
          labels:
            app: model-service
        spec:
          containers:

    - name: model-service

            image: model-service:latest
            resources:
              limits:
                memory: "1Gi"
                cpu: "500m"
            ports:

    - containerPort: 8000

Best Practices
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
------------

1. **Model Versioning**

   - Use semantic versioning
   - Track model lineage
   - Version control configurations

2. **Testing**

   - Unit tests for serving code
   - Integration tests for API
   - Load testing for production

3. **Monitoring**

   - Track prediction latency
   - Monitor resource usage
   - Set up alerting

4. **Documentation**

   - API documentation
   - Deployment procedures
   - Troubleshooting guides

Example Deployment
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
---------------

Complete deployment example:

.. code-block:: python

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import mlx.core as mx
    import numpy as np
    import json
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class PredictionRequest(BaseModel):
        data: list
        time_delta: Optional[list] = None
    
    class PredictionResponse(BaseModel):
        prediction: list
        confidence: float
    
    class ProductionModelServer:
        def __init__(self, model_path):
            self.model = self._load_model(model_path)
            self.metrics = MetricsClient()
            
        def _load_model(self, path):
            try:
                return load_model(path)
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
                
        async def predict(self, data, time_delta=None):
            try:
                x = mx.array(data)
                if time_delta is not None:
                    time_delta = mx.array(time_delta)
                
                with self.metrics.timer('prediction.duration'):
                    result = self.model(x, time_delta=time_delta)
                    
                return {
                    'prediction': result.tolist(),
                    'confidence': float(self._compute_confidence(result))
                }
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
        def _compute_confidence(self, result):
            # Implement confidence calculation
            return 0.95
    
    # Create FastAPI app
    app = FastAPI()
    model_server = ProductionModelServer('model.json')
    
    @app.post("/predict")
    async def predict(request: PredictionRequest):
        return await model_server.predict(
            request.data,
            request.time_delta
        )

Getting Help
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
----------

If you need deployment assistance:

1. Check deployment examples
2. Review best practices
3. Consult MLX documentation
4. Join community discussions
