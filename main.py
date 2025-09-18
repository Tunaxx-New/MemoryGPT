import uvicorn

from api import create_app

app = create_app()
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)