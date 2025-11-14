"""
RunPod handler for the Chandra OCR worker.
Creates a queue-compatible handler that processes RunPod jobs and
optionally launches a local vLLM server for inference.
"""
import atexit
import base64
import binascii
import logging
import os
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import runpod

from chandra.input import load_file
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

logger = logging.getLogger("chandra_runpod_handler")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

manager: Optional[InferenceManager] = None
manager_lock = threading.Lock()

vllm_process: Optional[subprocess.Popen] = None
vllm_lock = threading.Lock()


class InputError(Exception):
    """Raised when the incoming job payload is invalid."""


def should_launch_vllm() -> bool:
    """Determine if the embedded vLLM server should be started."""
    method = os.environ.get("CHANDRA_METHOD", "vllm").lower()
    start_flag = os.environ.get("START_VLLM_SERVER", "1").lower()
    return method == "vllm" and start_flag not in {"0", "false", "no"}


def start_vllm_server() -> None:
    """Launch vLLM in a background process if it is not already running."""
    global vllm_process
    with vllm_lock:
        if vllm_process and vllm_process.poll() is None:
            return

        model_name = os.environ.get("CHANDRA_MODEL", "datalab-to/chandra")
        port = str(os.environ.get("VLLM_PORT", "8000"))
        host = os.environ.get("VLLM_HOST", "0.0.0.0")
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--trust-remote-code",
            "--port",
            port,
            "--host",
            host,
        ]

        logger.info("Launching vLLM server for %s on %s:%s", model_name, host, port)
        vllm_process = subprocess.Popen(cmd)

    wait_for_vllm_ready()


def wait_for_vllm_ready(timeout: int = 180) -> None:
    """Poll the vLLM server until it responds or timeout occurs."""
    api_base = os.environ.get(
        "VLLM_API_BASE",
        f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '8000')}/v1",
    ).rstrip("/")
    status_url = f"{api_base}/models"
    deadline = time.time() + timeout

    logger.info("Waiting for vLLM server to become ready at %s", api_base)
    while time.time() < deadline:
        if vllm_process and vllm_process.poll() is not None:
            raise RuntimeError("vLLM server terminated during startup.")

        try:
            with urllib.request.urlopen(status_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    logger.info("vLLM server is ready")
                    return
        except Exception:
            time.sleep(2)

    raise RuntimeError(f"Timed out waiting for vLLM server at {status_url}")


def ensure_manager() -> InferenceManager:
    """Initialize the inference manager once and reuse it for every job."""
    global manager
    if manager is not None:
        return manager

    with manager_lock:
        if manager is not None:
            return manager

        if should_launch_vllm():
            start_vllm_server()

        method = os.environ.get("CHANDRA_METHOD", "vllm")
        default_base = f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '8000')}/v1"
        vllm_base = os.environ.get("VLLM_API_BASE", default_base)
        manager = InferenceManager(method=method, vllm_base=vllm_base)
        logger.info("Initialized InferenceManager with method=%s", method)

    return manager


def bool_from_input(value: Any, default: bool = False) -> bool:
    """Convert various input formats to a boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def int_from_input(value: Any, default: int) -> int:
    """Convert inputs to an integer, raising InputError on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise InputError("max_output_tokens must be an integer.") from exc


def get_file_payload(job_input: Dict[str, Any]) -> Tuple[bytes, str]:
    """Return the file bytes and filename from the job input."""
    if not job_input:
        raise InputError("Job input cannot be empty.")

    file_b64 = job_input.get("file") or job_input.get("file_b64")
    if file_b64:
        try:
            file_bytes = base64.b64decode(file_b64)
        except (binascii.Error, ValueError) as exc:
            raise InputError("Unable to decode 'file' payload.") from exc
        filename = job_input.get("filename", "document.pdf")
        return file_bytes, filename

    file_url = job_input.get("file_url")
    if file_url:
        try:
            with urllib.request.urlopen(file_url) as response:
                file_bytes = response.read()
        except Exception as exc:
            raise InputError(f"Failed to download file from {file_url}") from exc
        parsed = urllib.parse.urlparse(file_url)
        filename = job_input.get("filename") or Path(parsed.path).name or "document.pdf"
        return file_bytes, filename

    raise InputError("Provide either 'file' (base64) or 'file_url' in the input.")


def process_job(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Core OCR pipeline shared by all jobs."""
    file_bytes, filename = get_file_payload(job_input)
    suffix = Path(filename).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        config: Dict[str, Any] = {}
        if "page_range" in job_input:
            config["page_range"] = job_input["page_range"]

        images = load_file(str(tmp_path), config)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    if not images:
        raise InputError("No pages detected in the provided document.")

    batch = [
        BatchInputItem(
            image=image,
            prompt=job_input.get("prompt"),
            prompt_type=job_input.get("prompt_type", "ocr_layout"),
        )
        for image in images
    ]

    inference_manager = ensure_manager()
    results = inference_manager.generate(
        batch,
        max_output_tokens=int_from_input(job_input.get("max_output_tokens"), 8192),
        include_images=bool_from_input(job_input.get("include_images")),
        include_headers_footers=bool_from_input(
            job_input.get("include_headers_footers")
        ),
    )

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
        "markdown": "\n\n".join(page["markdown"] for page in pages),
        "html": "\n".join(page["html"] for page in pages),
        "pages": pages,
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler entry point."""
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return {"error": "Job payload is missing the 'input' dictionary."}

    try:
        return process_job(job_input)
    except InputError as exc:
        logger.warning("Validation error for job %s: %s", job.get("id"), exc)
        return {"error": str(exc)}


def shutdown_vllm() -> None:
    """Terminate the vLLM process when the worker exits."""
    global vllm_process
    proc = vllm_process
    if not proc or proc.poll() is not None:
        return

    logger.info("Stopping vLLM server.")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


atexit.register(shutdown_vllm)

runpod.serverless.start({"handler": handler})
