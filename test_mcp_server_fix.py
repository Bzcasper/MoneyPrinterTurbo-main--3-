import modal
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "d802ae16c9caed4d197895d27c6d529434cd8c6d"

image = modal.Image.debian_slim().pip_install(
    "torch==2.6.0",
    "sentence-transformers==3.4.1",
    "einops==0.8.1",
    "fastapi[standard]"
)
app = modal.App("example-base-nomic-embed", image=image)

GPU_CONFIG = "H100"

CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Define API key header
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# FastAPI app instance for endpoint
fastapi_app = FastAPI()

# Validate API key
async def get_api_key(api_key: str = Security(api_key_header)):
    # Fetch the expected API key from Modal Secret
    expected_api_key = modal.Secret.from_name("embed-api-key").get("API_KEY")
    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

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
    def embed(self, sentences: list):
        return self.model.encode(sentences)

# Web endpoint with API key authentication
@app.function(secrets=[modal.Secret.from_name("embed-api-key")])
@modal.fastapi_endpoint(method="POST")
async def api(body: dict, api_key: str = Security(get_api_key)):
    sentences = body.get("sentences", [])
    if not isinstance(sentences, list) or not sentences:
        return {"error": "Provide a list of sentences in the JSON body."}
    
    embeddings = Model().embed.remote(sentences)
    return {"embeddings": embeddings.tolist()}

# Run the model locally for testing
@app.local_entrypoint()
def main():
    sentences = [
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"
    ]

    print(Model().embed.remote(sentences))