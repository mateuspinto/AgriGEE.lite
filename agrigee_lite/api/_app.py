# Entrypoint for `fastapi dev` (debug mode).
# Do not import this from application code — use create_app() instead.
from agrigee_lite.api import create_app

app = create_app()
