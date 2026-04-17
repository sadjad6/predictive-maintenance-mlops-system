"""FastAPI application setup with CORS, exception handling, and lifespan."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.inference import InferencePipeline
from src.api.routes import router

# Global inference pipeline instance
pipeline = InferencePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: load models on startup."""
    logger.info("Starting Predictive Maintenance API")
    try:
        pipeline.load_models()
        logger.info("Models loaded successfully")
    except Exception:
        logger.warning("No trained models found — running in mock mode")
    yield
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Predictive Maintenance API",
        description=(
            "Production-grade API for industrial machinery failure prediction, "
            "Remaining Useful Life estimation, and anomaly detection."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
