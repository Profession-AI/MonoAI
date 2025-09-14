from typing import List, Optional, Dict, Any, Callable, Union
from datetime import datetime, timezone
import json
import asyncio

from monoai.models import Model
from monoai.agents import Agent
from .rate_limiter import RateLimiter

class Application:

    def __init__(self, name: str, model: Optional[Model] = None, agents: Optional[List[Agent]] = None, 
                 rate_limiter: Optional[RateLimiter] = None, user_validator: Optional[Callable[[str], Union[bool, str]]] = None):
        """
        Initialize the application.
        
        Parameters
        ----------
        name : str
            Application name
        model : Optional[Model], default None
            AI model to use
        agents : Optional[List[Agent]], default None
            List of available agents
        rate_limiter : Optional[RateLimiter], default None
            Rate limiter to control API usage
        user_validator : Optional[Callable[[str], Union[bool, str]]], default None
            Function to validate user_id. Must return:
            - True: user_id is valid
            - False: user_id is invalid
            - str: user_id is valid but normalized (e.g. "user123" -> "user_123")
        """
        self.name = name
        self._model = model
        self._agents: Optional[Dict[str, Agent]] = (
            {a.name: a for a in agents} if agents else None
        )
        self._rate_limiter = rate_limiter
        self._user_validator = user_validator
        self._started_at = datetime.now(timezone.utc)

    @staticmethod
    async def _maybe_await(fn: Callable[..., Any], *args, **kwargs) -> Any:
        result = fn(*args, **kwargs)
        if hasattr(result, "__await__"):
            return await result 
        return result

    def _get_user_identifier(self, request, data: Dict[str, Any]) -> str:
        """
        Extract user identifier from the request.
        First looks for user_id in the body, then uses IP as fallback.
        If a user_validator is configured, validates the user_id.
        
        Parameters
        ----------
        request : Request
            FastAPI request object
        data : Dict[str, Any]
            Request body data
            
        Returns
        -------
        str
            User identifier (user_id or IP-based)
        """
        # Look for user_id in the request body
        user_id = data.get("user_id")
        if user_id:
            user_id_str = str(user_id)
            
            # Validate user_id if validator is configured
            if self._user_validator:
                try:
                    validation_result = self._user_validator(user_id_str)
                    
                    if validation_result is True:
                        # user_id is valid, use as is
                        return user_id_str
                    elif validation_result is False:
                        # user_id is invalid, fallback to IP
                        pass
                    elif isinstance(validation_result, str):
                        # user_id is valid but normalized
                        return validation_result
                    else:
                        # Unknown validation result, fallback to IP
                        pass
                except Exception:
                    # Error during validation, fallback to IP
                    pass
            else:
                # No validator configured, use user_id as is
                return user_id_str
        
        # Fallback to client IP
        # Try different headers to get the real IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, take the first one
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                client_ip = real_ip
            else:
                # Handle both Request and WebSocket
                if hasattr(request, 'client') and request.client:
                    client_ip = request.client.host
                elif hasattr(request, 'url') and hasattr(request.url, 'hostname'):
                    client_ip = request.url.hostname
                else:
                    client_ip = "unknown"
        
        return f"ip:{client_ip}"

    def validate_user_id(self, user_id: str) -> Union[bool, str]:
        """
        Validate a user_id using the configured validator.
        
        Parameters
        ----------
        user_id : str
            user_id to validate
        
        Returns
        -------
        Union[bool, str]
            - True: user_id is valid
            - False: user_id is invalid
            - str: user_id is valid but normalized
        """
        if not self._user_validator:
            return True  # No validator, always consider valid
        
        try:
            return self._user_validator(user_id)
        except Exception:
            return False  # Error during validation, consider invalid

    def _build_app(self):

        try:
            from fastapi import FastAPI, Request, HTTPException, status
        except ImportError as e:
            raise ImportError(
                "fastapi is required to build the application. "
                "Install it with: pip install fastapi"
            ) from e

        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(title=self.name)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/", tags=["meta"], summary="Ping app")
        async def root():
            return {
                "msg": f"App {self.name} successfully started",
                "started_at": self._started_at.isoformat() + "Z",
            }

        @app.get("/health", tags=["meta"], summary="Health check")
        async def health():
            return {"status": "ok", "app": self.name}

        @app.get("/rate-limit/stats", tags=["rate-limit"], summary="Rate limiter statistics")
        async def rate_limit_stats():
            if not self._rate_limiter:
                return {"message": "Rate limiter not configured"}
            
            stats = self._rate_limiter.get_stats()
            return {
                "rate_limiter": str(self._rate_limiter),
                "global_stats": stats
            }

        @app.get("/rate-limit/stats/{user_id}", tags=["rate-limit"], summary="Statistics for a specific user")
        async def rate_limit_user_stats(user_id: str):
            if not self._rate_limiter:
                return {"message": "Rate limiter not configured"}
            
            user_stats = self._rate_limiter.get_stats(user_id)
            usage = self._rate_limiter.get_usage(user_id)
            remaining = self._rate_limiter.get_remaining(user_id)
            
            return {
                "user_id": user_id,
                "usage": usage,
                "remaining": remaining,
                "stats": user_stats
            }

        @app.post("/validate-user", tags=["auth"], summary="Validate a user_id")
        async def validate_user_endpoint(request: Request):
            try:
                data = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            
            user_id = data.get("user_id")
            if not user_id:
                raise HTTPException(status_code=400, detail="'user_id' is required")
            
            validation_result = self.validate_user_id(str(user_id))
            
            if validation_result is True:
                return {
                    "valid": True,
                    "user_id": user_id,
                    "normalized": None
                }
            elif validation_result is False:
                return {
                    "valid": False,
                    "user_id": user_id,
                    "error": "Invalid user_id"
                }
            elif isinstance(validation_result, str):
                return {
                    "valid": True,
                    "user_id": user_id,
                    "normalized": validation_result
                }
            else:
                return {
                    "valid": False,
                    "user_id": user_id,
                    "error": "Unknown validation result"
                }

        if self._model is not None:
            @app.post(
                "/model",
                tags=["model"],
                summary="Ask the model",
                status_code=status.HTTP_200_OK,
            )
            async def model_route(request: Request):
                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Execute the request to the model
                result = await self._maybe_await(self._model.ask, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        if self._agents is not None:
            @app.post(
                "/agent/{agent_name}",
                tags=["agents"],
                summary="Execute an agent",
                status_code=status.HTTP_200_OK,
            )
            async def agent_route(agent_name: str, request: Request):
                if agent_name not in self._agents:
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Execute the agent
                agent = self._agents[agent_name]
                result = await self._maybe_await(agent.run, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        # Model streaming endpoint
        if self._model is not None:
            @app.post(
                "/model/stream",
                tags=["model"],
                summary="Ask the model with streaming",
                status_code=status.HTTP_200_OK,
            )
            async def model_stream_route(request: Request):
                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Create a function to handle streaming
                def stream_handler(content: str):
                    return content
                
                # Enable streaming on the model
                if hasattr(self._model, 'enable_streaming'):
                    self._model.enable_streaming(stream_handler)
                
                # Execute the request to the model with streaming
                result = await self._maybe_await(self._model.ask, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        # Agent streaming endpoint
        if self._agents is not None:
            @app.post(
                "/agent/{agent_name}/stream",
                tags=["agents"],
                summary="Execute an agent with streaming",
                status_code=status.HTTP_200_OK,
            )
            async def agent_stream_route(agent_name: str, request: Request):
                if agent_name not in self._agents:
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON body")

                prompt = (data or {}).get("prompt")
                if not prompt:
                    raise HTTPException(status_code=400, detail="'prompt' is required")

                # Extract user identifier
                user_identifier = self._get_user_identifier(request, data or {})
                
                # Create a function to handle streaming
                def stream_handler(content: str):
                    return content
                
                # Enable streaming on the agent
                agent = self._agents[agent_name]
                if hasattr(agent, 'enable_streaming'):
                    agent.enable_streaming(stream_handler)
                
                # Execute the agent with streaming
                result = await self._maybe_await(agent.run, prompt)
                
                # Check and update rate limit if configured
                if self._rate_limiter:
                    # Check rate limit based on the response
                    if not self._rate_limiter.check_with_response(user_identifier, result):
                        raise HTTPException(
                            status_code=429, 
                            detail="Rate limit exceeded. Please try again later."
                        )

                    # Update the rate limiter with the response
                    self._rate_limiter.update_with_response(user_identifier, result)
                
                return result

        # WebSocket endpoint for real-time streaming
        if self._model is not None:
            @app.websocket("/model/ws")
            async def model_websocket(websocket):
                try:
                    from fastapi import WebSocket
                    await websocket.accept()
                    
                    while True:
                        # Receive prompt from client
                        data = await websocket.receive_text()
                        try:
                            request_data = json.loads(data)
                        except json.JSONDecodeError:
                            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                            continue
                        
                        prompt = request_data.get("prompt")
                        if not prompt:
                            await websocket.send_text(json.dumps({"error": "Prompt is required"}))
                            continue
                        
                        # Extract user identifier
                        user_identifier = self._get_user_identifier(websocket, request_data)
                        
                        # Create a handler to send chunks via WebSocket
                        def ws_stream_handler(content: str):
                            asyncio.create_task(websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": content
                            })))
                        
                        # Enable streaming on the model
                        if hasattr(self._model, 'enable_streaming'):
                            self._model.enable_streaming(ws_stream_handler)
                        
                        # Execute the request to the model
                        result = await self._maybe_await(self._model.ask, prompt)
                        
                        # Send the final result
                        await websocket.send_text(json.dumps({
                            "type": "complete",
                            "result": result
                        }))
                        
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                finally:
                    await websocket.close()

        if self._agents is not None:
            @app.websocket("/agent/{agent_name}/ws")
            async def agent_websocket(websocket, agent_name: str):
                try:
                    from fastapi import WebSocket
                    await websocket.accept()
                    
                    if agent_name not in self._agents:
                        await websocket.send_text(json.dumps({"error": f"Agent '{agent_name}' not found"}))
                        await websocket.close()
                        return
                    
                    agent = self._agents[agent_name]
                    
                    while True:
                        # Receive prompt from client
                        data = await websocket.receive_text()
                        try:
                            request_data = json.loads(data)
                        except json.JSONDecodeError:
                            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                            continue
                        
                        prompt = request_data.get("prompt")
                        if not prompt:
                            await websocket.send_text(json.dumps({"error": "Prompt is required"}))
                            continue
                        
                        # Extract user identifier
                        user_identifier = self._get_user_identifier(websocket, request_data)
                        
                        # Create a handler to send chunks via WebSocket
                        def ws_stream_handler(content: str):
                            asyncio.create_task(websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": content
                            })))
                        
                        # Enable streaming on the agent
                        if hasattr(agent, 'enable_streaming'):
                            agent.enable_streaming(ws_stream_handler)
                        
                        # Execute the agent
                        result = await self._maybe_await(agent.run, prompt)
                        
                        # Send the final result
                        await websocket.send_text(json.dumps({
                            "type": "complete",
                            "result": result
                        }))
                        
                except Exception as e:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                finally:
                    await websocket.close()

        return app

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: Optional[int] = None,
        log_level: str = "info",
    ):

        """
        Serve the application creating endpoints for the model and agents.

        API Endpoints:
            When the server is running, the following endpoints are available:
            
            Model Endpoints:
                POST /model                    Ask the model
                POST /model/stream             Ask the model with streaming
                WS  /model/ws                  Ask the model with WebSocket
            
            Agent Endpoints:
                POST /agent/{agent_name}       Execute an agent
                POST /agent/{agent_name}/stream Execute an agent with streaming
                WS  /agent/{agent_name}/ws     Execute an agent with WebSocket
            
            Authentication:
                POST /validate-user            Validate a user_id
            
            Rate Limiting:
                GET  /rate-limit/stats         Rate limiter statistics
                GET  /rate-limit/stats/{user_id} Statistics for a specific user
            
            Meta:
                GET  /                         Ping the application
                GET  /health                   Health check

        Parameters
        ----------
        host : str, default "0.0.0.0"
            Host to serve the application
        port : int, default 8000
            Port to serve the application
        reload : bool, default False
            Whether to reload the application when code changes are detected
        workers : Optional[int], default None
            Number of workers to use
        log_level : str, default "info"
            Log level
        """

        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "uvicorn is required to serve the application. "
                "Install it with: pip install uvicorn"
            ) from e

        uvicorn.run(
            self._build_app(),
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
        )
