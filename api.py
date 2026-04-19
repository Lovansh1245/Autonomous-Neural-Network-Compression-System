"""
api.py — FastAPI Model Service for the Self-Pruning Neural Network System.

Endpoints:
  /train      — POST: Train model with given λ (async background task)
  /evaluate   — GET:  Returns accuracy + sparsity for a trained model
  /predict    — POST: Inference on uploaded image (base64 or file)
  /metrics    — GET:  Returns model stats, gate distributions
  /query      — POST: RAG query endpoint — ask about experiments
  /agent/recommend — POST: Full agent pipeline analysis + recommendation

All endpoints use Pydantic schemas for validation and documentation.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import SystemConfig, get_config
from model import PrunableCNN
from train import train_model, evaluate_accuracy, get_cifar10_loaders
from analysis import AnalysisEngine
from rag import ExperimentStore
from agent import PruningAgent

logger = logging.getLogger("api")

# ─── App Setup ───────────────────────────────────────────────────────────────────

config = get_config()
app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    description=(
        "Production-grade API for training, evaluating, and querying "
        "self-pruning neural networks with learnable gates."
    ),
)

# Add CORS middleware to allow the Vite frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared State ────────────────────────────────────────────────────────────────

# Model registry: stores trained models by experiment ID
model_registry: dict[str, dict[str, Any]] = {}
training_tasks: dict[str, str] = {}  # task_id → status
agent_instance: Optional[PruningAgent] = None
rag_store: Optional[ExperimentStore] = None

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

@app.on_event("startup")
async def startup_event():
    """Load persistent models from disk on server boot to maintain state."""
    logger.info("Initializing API and loading persistent models...")
    models_dir = config.paths.models_dir
    if not models_dir.exists():
        return
        
    loaded = 0
    for pt_file in models_dir.glob("model_lambda_*.pt"):
        if "_hard" in pt_file.name: 
            continue
        try:
            checkpoint = torch.load(pt_file, map_location="cpu", weights_only=False)
            lam = checkpoint["lambda_value"]
            task_id = f"exp_{lam}_loaded"
            
            # Check if rich JSON log exists to append full telemetry (epoch arrays, gate maths, flops)
            json_path = config.paths.logs_dir / f"experiment_lambda_{lam}.json"
            if json_path.exists():
                import json
                with open(json_path) as f:
                    log_data = json.load(f)
                # Ensure the loaded log data is assigned locally
                model_registry[task_id] = log_data
            else:
                # Reconstruct the expected registry payload fallback
                model_registry[task_id] = {
                    "lambda_value": lam,
                    "final_accuracy": checkpoint.get("final_accuracy", 0.0),
                    "final_sparsity": checkpoint.get("final_sparsity", 0.0),
                    "flops_reduction": {"total_reduction_pct": 0.0}, # Estimated
                    "config": checkpoint.get("config", {}),
                    "inference_ms_baseline": checkpoint.get("inference_ms_baseline", 0.0),
                    "inference_ms_pruned": checkpoint.get("inference_ms_pruned", 0.0),
                }
            training_tasks[task_id] = "completed (loaded)"
            loaded += 1
        except Exception as e:
            logger.warning(f"Failed to load {pt_file}: {e}")
            
    logger.info(f"Loaded {loaded} continuous models from disk into registry.")


def get_agent() -> PruningAgent:
    global agent_instance
    if agent_instance is None:
        agent_instance = PruningAgent(config=config)
    return agent_instance


def get_rag() -> ExperimentStore:
    global rag_store
    if rag_store is None:
        rag_store = ExperimentStore(
            embedding_model_name=config.rag.embedding_model,
            embedding_dim=config.rag.embedding_dim,
            persist_dir=config.paths.rag_dir,
        )
    return rag_store


# ─── Schemas ─────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    lambda_value: float = Field(..., description="Sparsity penalty (λ)", ge=0, examples=[0.001])
    epochs: Optional[int] = Field(None, description="Number of training epochs", ge=1, le=200)
    batch_size: Optional[int] = Field(None, description="Batch size", ge=8, le=512)
    lr: Optional[float] = Field(None, description="Learning rate", gt=0)
    lambda_schedule: Optional[str] = Field(None, description="Lambda schedule: constant, linear_warmup, cosine")


class TrainResponse(BaseModel):
    task_id: str
    status: str
    message: str


class EvalRequest(BaseModel):
    task_id: str = Field(..., description="ID of the trained model to evaluate")


class EvalResponse(BaseModel):
    task_id: str
    lambda_value: float
    accuracy: float
    sparsity: float
    flops_reduction_pct: float
    gate_stats: dict[str, Any]


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (PNG/JPEG)")
    task_id: Optional[str] = Field(None, description="Specific model to use (latest if None)")


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class MetricsResponse(BaseModel):
    total_models: int
    models: list[dict[str, Any]]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about experiments", min_length=3)
    top_k: int = Field(3, description="Number of results to retrieve", ge=1, le=10)


class QueryResponse(BaseModel):
    question: str
    answer: str
    num_results: int


class AgentRecommendRequest(BaseModel):
    lambda_values: Optional[list[float]] = Field(
        None,
        description="Lambda values to experiment with (uses defaults if None)",
    )


class AgentRecommendResponse(BaseModel):
    recommended_lambda: float
    accuracy: float
    sparsity: float
    composite_score: float
    observations: list[str]
    total_time: float


# ─── Background Training ────────────────────────────────────────────────────────

def _run_training_background(task_id: str, request: TrainRequest) -> None:
    """Background task for model training."""
    try:
        training_tasks[task_id] = "running"
        logger.info(f"Background training started: {task_id}")

        # Override config with request params
        train_config = config.train
        if request.epochs:
            train_config.epochs = request.epochs
        if request.batch_size:
            train_config.batch_size = request.batch_size
        if request.lr:
            train_config.lr = request.lr
        if request.lambda_schedule:
            train_config.lambda_schedule = request.lambda_schedule

        result = train_model(
            config=train_config,
            paths=config.paths,
            lambda_val=request.lambda_value,
        )

        # Store in registry
        model_registry[task_id] = vars(result)

        # Add to RAG store
        store = get_rag()
        store.add_experiment(vars(result))

        training_tasks[task_id] = "completed"
        logger.info(f"Training completed: {task_id} — acc={result.final_accuracy:.2%}")

    except Exception as e:
        training_tasks[task_id] = f"failed: {str(e)}"
        logger.error(f"Training failed: {task_id} — {e}")


# ─── Endpoints ───────────────────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse)
async def train_endpoint(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    """Train a model with the given λ value. Returns immediately — training runs in background."""
    import uuid
    task_id = f"exp_{request.lambda_value}_{uuid.uuid4().hex[:8]}"

    background_tasks.add_task(_run_training_background, task_id, request)
    training_tasks[task_id] = "queued"

    return TrainResponse(
        task_id=task_id,
        status="queued",
        message=f"Training started with λ={request.lambda_value}. Check /evaluate?task_id={task_id}",
    )


@app.get("/evaluate", response_model=EvalResponse)
async def evaluate_endpoint(task_id: str) -> EvalResponse:
    """Get evaluation metrics for a trained model."""
    if task_id not in model_registry:
        status = training_tasks.get(task_id, "not found")
        raise HTTPException(
            status_code=404,
            detail=f"Model '{task_id}' not found. Status: {status}",
        )

    result = model_registry[task_id]
    flops = result.get("flops_reduction", {})

    return EvalResponse(
        task_id=task_id,
        lambda_value=result["lambda_value"],
        accuracy=result["final_accuracy"],
        sparsity=result["final_sparsity"],
        flops_reduction_pct=flops.get("total_reduction_pct", 0.0),
        gate_stats=result.get("gate_stats", {}),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Run inference on a base64-encoded image using a trained model."""
    # Find model
    if request.task_id and request.task_id in model_registry:
        result_data = model_registry[request.task_id]
    elif model_registry:
        # Use latest model
        result_data = list(model_registry.values())[-1]
    else:
        raise HTTPException(status_code=404, detail="No trained models available")

    # Load model
    model_path = config.paths.models_dir / f"model_lambda_{result_data['lambda_value']}.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    model = PrunableCNN(num_classes=10).to(config.train.device)
    checkpoint = torch.load(model_path, map_location=config.train.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Decode and preprocess image
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tensor = transform(image).unsqueeze(0).to(config.train.device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = probs.max(0)

    probabilities = {
        CIFAR10_CLASSES[i]: float(probs[i]) for i in range(10)
    }

    return PredictResponse(
        predicted_class=CIFAR10_CLASSES[predicted.item()],
        class_index=predicted.item(),
        confidence=float(confidence),
        probabilities=probabilities,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint() -> MetricsResponse:
    """Get metrics for all trained models."""
    models = []
    for task_id, result in model_registry.items():
        models.append({
            "task_id": task_id,
            "lambda_value": result["lambda_value"],
            "accuracy": result["final_accuracy"],
            "sparsity": result["final_sparsity"],
            "flops_reduction_pct": result.get("flops_reduction", {}).get(
                "total_reduction_pct", 0.0
            ),
            "training_time": result.get("training_time_seconds", 0.0),
            "status": training_tasks.get(task_id, "unknown"),
        })

    return MetricsResponse(
        total_models=len(models),
        models=models,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Query the RAG system about past experiments."""
    store = get_rag()

    if not store.documents:
        # Load persisted data if available
        try:
            store.load(config.paths.rag_dir)
        except Exception:
            pass

    if not store.documents:
        raise HTTPException(
            status_code=404,
            detail="No experiments in the knowledge base. Train models first.",
        )

    docs = store.query(request.question, top_k=request.top_k)
    answer = store.generate_response(request.question, docs)

    return QueryResponse(
        question=request.question,
        answer=answer,
        num_results=len(docs),
    )


@app.post("/agent/recommend", response_model=AgentRecommendResponse)
async def agent_recommend_endpoint(
    request: AgentRecommendRequest,
) -> AgentRecommendResponse:
    """
    Run the full agent pipeline: train → store → analyze → recommend.
    Note: This is a synchronous long-running operation.
    """
    agent = get_agent()
    result = agent.run_full_pipeline(lambda_values=request.lambda_values)

    analysis = result.get("analysis", {})
    recommendation = analysis.get("recommendation", {})
    observations = analysis.get("observations", [])

    return AgentRecommendResponse(
        recommended_lambda=recommendation.get("recommended_lambda", 0.0),
        accuracy=recommendation.get("accuracy", 0.0),
        sparsity=recommendation.get("sparsity", 0.0),
        composite_score=recommendation.get("composite_score", 0.0),
        observations=observations,
        total_time=result.get("total_time", 0.0),
    )


@app.post("/reload")
async def reload_models():
    """Force an active disk scan to reload all checkpoints into the API registry."""
    model_registry.clear()
    await startup_event()
    return {"status": "reloaded", "models_loaded": len(model_registry)}

@app.get("/models")
async def list_models():
    """List all currently loaded checkpoints in memory."""
    return {"loaded_count": len(model_registry), "registry_keys": list(model_registry.keys())}


@app.get("/dashboard/data")
async def dashboard_data():
    """Return explicit unfiltered raw arrays for React Recharts telemetry."""
    # Convert dict values to list of dictionaries
    return list(model_registry.values())


# ─── Root / Health Check ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": config.api.title,
        "version": config.api.version,
        "device": str(config.train.device),
        "models_loaded": len(model_registry),
        "status": "healthy",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "device": str(config.train.device),
        "registry_size": len(model_registry)
    }


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def favicon():
    from fastapi import Response
    return Response(status_code=204)
