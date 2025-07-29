import os
import typing as t
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
import modal

# ------------------------------------------------------------------
# Configuration (edit as you like)
# ------------------------------------------------------------------
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "d802ae16c9caed4d197895d27c6d529434cd8c6d"
GPU_CONFIG = "H100"
CACHE_DIR = "/cache"
API_KEY = os.environ.get("EMBED_API_KEY")  # set modal secret before deploy
# ------------------------------------------------------------------

image = modal.Image.debian_slim().pip_install(
    "torch==2.6.0",
    "sentence-transformers==3.4.1",
    "einops==0.8.1",
    "fastapi[standard]",
)

app = modal.App("nomic-embed-service", image=image)

cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


# ------------------------------------------------------------------
# Modal model class
# ------------------------------------------------------------------
@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    scaledown_window=60 * 10,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.enter()
    def setup(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_folder=CACHE_DIR,
            trust_remote_code=True,
        )

    @modal.method()
    def embed(self, sentences: t.List[str]) -> t.List[t.List[float]]:
        return self.model.encode(sentences, normalize_embeddings=True).tolist()


# ------------------------------------------------------------------
# FastAPI app (runs inside Modal)
# ------------------------------------------------------------------
web = FastAPI(
    title="Nomic-Embed OpenAI-compatible embedding service",
    version="1.0.0",
)


# ------------------------------------------------------------------
# Security dependency
# ------------------------------------------------------------------
def verify_key(authorization: t.Annotated[str, Header(alias="authorization")]):
    if not API_KEY:
        # no key configured -> allow all
        return
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ------------------------------------------------------------------
# Request/response models
# ------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    input: t.Union[str, t.List[str]] = Field(
        description="Text or list of texts to embed"
    )
    model: t.Optional[str] = Field(
        None, description="Model name (ignored; always uses nomic-embed-text-v1.5)"
    )


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: t.List[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: t.List[EmbeddingObject]
    model: str
    usage: Usage


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@web.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(
    req: EmbeddingRequest,
    _: t.Annotated[None, Depends(verify_key)],
):
    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(status_code=400, detail="No input provided")

    embeddings = Model().embed.remote(texts)

    data = [
        EmbeddingObject(index=i, embedding=emb) for i, emb in enumerate(embeddings)
    ]
    return EmbeddingResponse(
        data=data,
        model=MODEL_ID,
        usage=Usage(prompt_tokens=0, total_tokens=0),  # token counts not tracked
    )


# Health check
@web.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Modal endpoint
# ------------------------------------------------------------------
@app.function()
@modal.fastapi_endpoint()
def serve():
    return web


# ------------------------------------------------------------------
# Local dev helper
# ------------------------------------------------------------------
@app.local_entrypoint()
def main():
    sentences = ["search_document: TSNE is a dimensionality reduction algorithm"]
    print(Model().embed.remote(sentences))