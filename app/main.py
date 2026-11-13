import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.database import async_session_maker, engine
from app.models import ServiceLog

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get()
        return True


logger = logging.getLogger("nightwatch")
logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | request_id=%(request_id)s | %(name)s | %(message)s"
        )
    )
    _handler.addFilter(RequestIDFilter())
    logger.addHandler(_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("Application started, database connection verified")
    yield
    await engine.dispose()
    logger.info("Database engine disposed")


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    token = request_id_ctx_var.set(req_id)
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time)*1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        
        return response

    finally:
        request_id_ctx_var.reset(token)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = getattr(request.state, "request_id", None) or request_id_ctx_var.get()
    tb_text = traceback.format_exc()

    token = request_id_ctx_var.set(req_id)
    try:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        try:
            async with async_session_maker() as session:
                log_entry = ServiceLog(
                    request_id=req_id,
                    service_name="nightwatch-api",
                    level="ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                    metadata_json={
                        "method": request.method,
                        "path": request.url.path,
                        "traceback": tb_text,
                    },
                )
                session.add(log_entry)
                await session.commit()
        except Exception:
            logger.exception("Failed to persist error log to database")
    finally:
        request_id_ctx_var.reset(token)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": req_id},
        headers={"X-Request-ID": req_id},
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok", "service": settings.APP_NAME}

