"""AWS Lambda entry point: adapts the FastAPI app to Lambda's event/response
shape via Mangum, so the same app.py serves both the container/ECS path
(uvicorn) and the Lambda path (this handler) with no route duplication."""

from mangum import Mangum

from finsight.api.app import app

handler = Mangum(app)
