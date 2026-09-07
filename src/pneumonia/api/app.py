"""FastAPI application exposing the validated pneumonia model."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from pneumonia import __version__
from pneumonia.api.inference import InvalidImageError, PredictionService
from pneumonia.api.monitoring import ApiMetrics
from pneumonia.api.schemas import HealthResponse, PredictionResponse

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def create_app(service: PredictionService | None = None) -> FastAPI:
    """Create an application with an injectable inference service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.prediction_service = service
        else:
            app.state.prediction_service = PredictionService()
        yield

    application = FastAPI(
        title="Pneumonia Detection API",
        version=__version__,
        description=(
            "Prototype académique de classification de radiographies thoraciques. "
            "Ne remplace pas un diagnostic médical."
        ),
        lifespan=lifespan,
    )
    application.state.metrics = ApiMetrics()

    @application.middleware("http")
    async def observe_requests(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started_at = perf_counter()
        response = await call_next(request)
        path = request.url.path
        metrics: ApiMetrics = request.app.state.metrics
        metrics.requests.labels(request.method, path, str(response.status_code)).inc()
        metrics.latency.labels(request.method, path).observe(perf_counter() - started_at)
        return response

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        prediction_service: PredictionService = request.app.state.prediction_service
        return HealthResponse(
            status="healthy",
            model=prediction_service.model_name,
            threshold=prediction_service.threshold,
        )

    @application.get("/metrics", include_in_schema=False, tags=["system"])
    async def metrics(request: Request) -> Response:
        api_metrics: ApiMetrics = request.app.state.metrics
        return Response(api_metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        status_code=status.HTTP_200_OK,
    )
    async def predict(
        request: Request,
        file: Annotated[UploadFile, File()],
        include_gradcam: Annotated[bool, Query()] = True,
    ) -> PredictionResponse:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Formats acceptés : JPEG et PNG.",
            )
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="La taille maximale est de 10 Mo.",
            )
        prediction_service: PredictionService = request.app.state.prediction_service
        try:
            result = prediction_service.predict(content, include_gradcam)
        except InvalidImageError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        api_metrics: ApiMetrics = request.app.state.metrics
        api_metrics.predictions.labels(result["model"], result["prediction"]).inc()
        return PredictionResponse.model_validate(result)

    return application


app = create_app()
