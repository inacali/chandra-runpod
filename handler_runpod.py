"""
RunPod vLLM handler for Chandra OCR
Compatible with RunPod's serverless endpoint format
"""
import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io

from chandra.input import load_file
from chandra.model import InferenceManager

app = FastAPI(title="Chandra OCR RunPod Endpoint")

# Initialize manager
manager = None

@app.on_event("startup")
async def startup():
    global manager
    method = os.environ.get("CHANDRA_METHOD", "vllm")
    vllm_base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    manager = InferenceManager(method=method, vllm_base=vllm_base)

@app.post("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ocr")
async def process_ocr(request: Request):
    """
    Process OCR request
    Accepts: PDF/image files via multipart form or base64 JSON
    """
    try:
        content_type = request.headers.get("content-type", "")
        
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if not file:
                raise HTTPException(status_code=400, detail="No file provided")
            
            file_bytes = await file.read()
            filename = file.filename or "document.pdf"
        else:
            # JSON with base64
            data = await request.json()
            if "file" not in data:
                raise HTTPException(status_code=400, detail="No file in request")
            
            file_bytes = base64.b64decode(data["file"])
            filename = data.get("filename", "document.pdf")
        
        # Save to temp file
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        
        try:
            # Load document
            config = {}
            if "page_range" in data:
                config["page_range"] = data["page_range"]
            
            images = load_file(str(tmp_path), config)
            
            if not images:
                raise HTTPException(status_code=400, detail="No pages detected")
            
            # Process with Chandra
            from chandra.model.schema import BatchInputItem
            
            batch = [
                BatchInputItem(
                    image=img,
                    prompt=data.get("prompt"),
                    prompt_type=data.get("prompt_type", "ocr_layout"),
                )
                for img in images
            ]
            
            results = manager.generate(
                batch,
                max_output_tokens=int(data.get("max_output_tokens", 8192)),
                include_images=bool(data.get("include_images", False)),
                include_headers_footers=bool(data.get("include_headers_footers", False)),
            )
            
            # Format response
            pages = []
            for idx, result in enumerate(results):
                page_data = {
                    "page_index": idx,
                    "markdown": result.markdown,
                    "html": result.html,
                    "token_count": result.token_count,
                    "chunks": result.chunks,
                    "error": result.error,
                }
                pages.append(page_data)
            
            return {
                "filename": filename,
                "num_pages": len(pages),
                "markdown": "\n\n".join([p["markdown"] for p in pages]),
                "html": "\n".join([p["html"] for p in pages]),
                "pages": pages,
            }
        
        finally:
            # Cleanup
            try:
                tmp_path.unlink()
            except:
                pass
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
