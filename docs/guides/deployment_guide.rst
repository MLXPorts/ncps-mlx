Deployment Guide
================

This guide covers best practices for deploying Neural Circuit Policies in production environments using MLX.

Model Serialization
-------------------

Saving Models
~~~~~~~~~~~~~

MLX models can be serialized to JSON format.

.. code-block:: python

def save_model(
    model,
        path)::,
    ))))))))
    """Save model to file."""
        state = {
        'model_state': model.state_dict(
        'model_config': {
            'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                    'num_layers': model.num_layers,
                # Add other relevant config

                with open(
                json.dump(

            Loading Models
            ~~~~~~~~~~~~~~

            Load saved models for inference.

            .. code-block:: python

            def load_model(
                path)::,
            )
            """Load model from file."""
            with open(
            state = json.load(

            model = CfC(
            model.load_state_dict(
        return model

        Model Optimization
        ------------------

        Compilation
        ~~~~~~~~~~~

        Use MLX's compilation for faster inference.

        .. code-block:: python

        @mx.compile
        def optimized_inference(
            model,
                x,
                    time_delta=None)::,
                )
                return model(

            Batch Processing
            ~~~~~~~~~~~~~~~~

            Efficient batch processing for multiple inputs.

            .. code-block:: python

            class BatchProcessor::
                def __init__(
                    self,
                        model,
                            batch_size=32)::,
                        )
                        self.model = model
                        self.batch_size = batch_size

                        def process_all(
                            self,
                                data)::,
                            )
                            results = [
                            for i in range(
                                0,
                                len(
                                    data),
                                )
                                    self.batch_size)::,
                                )
                                batch = data[i:i+self.batch_size
                                results.extend(
                            return results

                            Memory Management
                            ~~~~~~~~~~~~~~~~~

                            Optimize memory usage for production.

                            .. code-block:: python

                            class MemoryOptimizedInference::
                                def __init__(
                                    self,
                                        model)::,
                                    )
                                    self.model = model

                                    def __call__(
                                        self,
                                            x)::,
                                        )
                                        with mx.stream(
                                        result = self.model(
                                        mx.eval(
                                    return result

                                    Serving Strategies
                                    ------------------

                                    FastAPI Integration
                                    ~~~~~~~~~~~~~~~~~~~

                                    Create a REST API using FastAPI.

                                    .. code-block:: python

                                    from fastapi import FastAPI
                                    import mlx.core as mx
                                    import numpy as np

                                    app = FastAPI(

                                class ModelServer::
                                    def __init__(
                                        self,
                                            model_path)::,
                                        )
                                        self.model = load_model(

                                        async def predict(
                                        x = mx.array(
                                        return self.model(

                                        server = ModelServer(

                                        @app.post(
                                        async def predict(
                                        result = await server.predict(
                                            return {"prediction": result.tolist(

                                        gRPC Service
                                        ~~~~~~~~~~~~

                                        High-performance gRPC service.

                                        .. code-block:: python

                                        import grpc
                                        from concurrent import futures
                                        import prediction_pb2
                                        import prediction_pb2_grpc

                                        class PredictionService(
                                            prediction_pb2_grpc.PredictorServicer)::,
                                        )
                                        def __init__(
                                            self,
                                                model_path)::,
                                            )
                                            self.model = load_model(

                                            def Predict(
                                                self,
                                                    request,
                                                        context)::,
                                                    )))))))))))
                                                    input_data = np.array(
                                                    prediction = self.model(
                                                    return prediction_pb2.PredictionResponse(
                                                    prediction=prediction.tolist(

                                                    def serve(
                                                        )::,
                                                    ))))
                                                    server = grpc.server(
                                                    prediction_pb2_grpc.add_PredictorServicer_to_server(
                                                    PredictionService(

                                                    server.add_insecure_port(
                                                    server.start(
                                                    server.wait_for_termination(

                                                Production Considerations
                                                -------------------------

                                                Error Handling
                                                ~~~~~~~~~~~~~~

                                                Robust error handling for production.

                                                .. code-block:: python

                                                class ProductionModel::
                                                    def __init__(
                                                        self,
                                                            model_path)::,
                                                        )
                                                        self.model = load_model(

                                                        def predict(
                                                            self,
                                                                x)::,
                                                            )))))
                                                            try:
                                                            # Input validation
                                                            if not self._validate_input(
                                                                x)::,
                                                            )))))
                                                            raise ValueError(

                                                        # Prediction with timeout
                                                        with timeout(
                                                        result = self.model(

                                                    # Output validation
                                                    if not self._validate_output(
                                                        result)::,
                                                    ))))))))))
                                                    raise ValueError(

                                                return result

                                                except Exception as e:
                                                logger.error(
                                            raise

                                            Monitoring
                                            ~~~~~~~~~~

                                            Monitor model performance in production.

                                            .. code-block:: python

                                            class MonitoredModel::
                                                def __init__(
                                                    self,
                                                        model,
                                                            metrics_client)::,
                                                        )
                                                        self.model = model
                                                        self.metrics = metrics_client

                                                        def predict(
                                                            self,
                                                                x)::,
                                                            )
                                                            start_time = time.time(
                                                        try:
                                                        result = self.model(
                                                        self.metrics.increment(
                                                    return result
                                                    except Exception as e:
                                                    self.metrics.increment(
                                                raise
                                                finally:
                                                duration = time.time(
                                                self.metrics.timing(

                                            Scaling
                                            ~~~~~~~

                                            Strategies for scaling model serving.

                                            .. code-block:: python

                                            class LoadBalancedPredictor::
                                                def __init__(
                                                    self,
                                                        model_paths,
                                                            max_batch_size=32)::,
                                                        )
                                                        self.models = [load_model(
                                                    self.current_model = 0
                                                    self.max_batch_size = max_batch_size

                                                    def predict(
                                                        self,
                                                            x)::,
                                                        )))))
                                                        # Round-robin load balancing
                                                        model = self.models[self.current_model
                                                        self.current_model = (

                                                    # Batch size management
                                                    if len(
                                                        x) > self.max_batch_size::,
                                                    )))))))))))))))))))))))))))
                                                    return self._predict_large_batch(
                                                    return model(

                                                    def _predict_large_batch(
                                                        self,
                                                            x)::,
                                                        )))))
                                                        results = [
                                                        for i in range(
                                                            0,
                                                            len(
                                                                x),
                                                            )
                                                                self.max_batch_size)::,
                                                            )))))))))))))))))))))))
                                                            batch = x[i:i+self.max_batch_size
                                                            results.append(
                                                            return mx.concatenate(

                                                        Deployment Environments
                                                        -----------------------

                                                        Docker Deployment
                                                        ~~~~~~~~~~~~~~~~~

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
                                                        CMD ["python", "server.py"

                                                        Kubernetes Configuration
                                                        ~~~~~~~~~~~~~~~~~~~~~~~~

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
                                                        pass

                                                        - name: model-service

                                                        image: model-service:latest
                                                        resources:
                                                        limits:
                                                        memory: "1Gi"
                                                        cpu: "500m"
                                                        ports:
                                                        pass

                                                        - containerPort: 8000

                                                        Best Practices
                                                        --------------

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

                                                        Complete deployment example:

                                                        .. code-block:: python

                                                        from fastapi import FastAPI, HTTPException
                                                        from pydantic import BaseModel
                                                        import mlx.core as mx
                                                        import numpy as np
                                                        import json
                                                        import logging

                                                        # Configure logging
                                                        logging.basicConfig(
                                                        logger = logging.getLogger(

                                                        class PredictionRequest(
                                                            BaseModel)::,
                                                        )))))))))))))
                                                        data: list
                                                        time_delta: Optional[list] = None

                                                        class PredictionResponse(
                                                            BaseModel)::,
                                                        )))))))))))))
                                                        prediction: list
                                                        confidence: float

                                                        class ProductionModelServer::
                                                            def __init__(
                                                                self,
                                                                    model_path)::,
                                                                )
                                                                self.model = self._load_model(
                                                                self.metrics = MetricsClient(

                                                                def _load_model(
                                                                    self,
                                                                        path)::,
                                                                    ))))))))
                                                                    try:
                                                                    return load_model(
                                                                except Exception as e:
                                                                logger.error(
                                                            raise

                                                            async def predict(
                                                        try:
                                                        x = mx.array(
                                                    if time_delta is not None::
                                                        pass
                                                        time_delta = mx.array(

                                                        with self.metrics.timer(
                                                    pass
                                                    result = self.model(

                                                    return {
                                                    'prediction': result.tolist(
                                                    'confidence': float(

                                                except Exception as e:
                                                pass
                                                logger.error(
                                                raise HTTPException(

                                                def _compute_confidence(
                                                    self,
                                                        result)::,
                                                    ))))))))))
                                                    pass
                                                    # Implement confidence calculation
                                                    return 0.95

                                                    # Create FastAPI app
                                                    app = FastAPI(
                                                    model_server = ProductionModelServer(

                                                    @app.post(
                                                    async def predict(
                                                    return await model_server.predict(
                                                        request.data,
                                                    request.time_delta

                                                    Getting Help
                                                    ------------

                                                    If you need deployment assistance:

                                                    1. Check deployment examples
                                                    2. Review best practices
                                                    3. Consult MLX documentation
                                                    4. Join community discussions

