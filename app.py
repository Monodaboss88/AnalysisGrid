"""
BARE MINIMUM deploy test - zero project imports.
If this fails: problem is Railway config / Dockerfile / requirements.
If this works: problem is in our code imports.
"""
import os
import time

_start = time.time()
print(f"[BOOT] app.py PID={os.getpid()} PORT={os.environ.get('PORT','?')}", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AnalysisGrid", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
async def health():
    return {"status": "ok", "uptime": round(time.time() - _start, 1)}

@app.get("/api/status")
async def status():
    return {"status": "running", "message": "bare minimum test"}

@app.get("/")
async def root():
    return {"message": "AnalysisGrid API running"}

print(f"[BOOT] app ready in {time.time()-_start:.2f}s", flush=True)
