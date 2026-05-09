"""FastAPI application entry point."""

from fastapi import FastAPI

from .config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="teletop", version="0.1.0")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "teletop"}

    return app


app = create_app()


def cli() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "teletop_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
