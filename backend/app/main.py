from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {settings.app_name}...")
    print(f"Device: {settings.device}")
    print(f"CLIP model: {settings.clip_model}")
    print("Server ready.")

    yield

    print("Shutting down...")


app = FastAPI(
    title="GeoGuessr AI",
    description="Upload a Street View image and get a location prediction.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "app": settings.app_name}


# Import and include routers
from app.api.v1.analyze import router as analyze_router
app.include_router(analyze_router, prefix="/api/v1")