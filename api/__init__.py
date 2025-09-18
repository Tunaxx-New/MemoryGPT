from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.routes import router
from settings import Settings


def create_app() -> FastAPI:
    settings = Settings() # noqa

    app = FastAPI(
        openapi_tags=[
            {'name': 'Memory GPT', 'description': 'Memory GPT that remembers something.'}
        ]
    )

    origins = [
        "http://localhost:8080",
        "http://frontend:8080",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix='/api', tags=['API'])

    return app
