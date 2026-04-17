"""
Bulk Folder Image Upscaler Web App (FastAPI)

Frontend:
- Single-page `index.html` (Tailwind CDN + vanilla JS)
- Select an entire folder via <input webkitdirectory>
- Uploads files to backend

Backend pipeline per image:
1) Upload local file to ImgBB to obtain a public URL (CRITICAL: A1D needs public URL)
2) Call A1D image upscaler API with imageUrl
3) Download the upscaled image to a temporary folder
4) Continue for the next image (sequential or limited concurrency)
5) Zip all successful upscaled images and return as a downloadable file

Environment variables (.env):
- A1D_API_KEY=...
- IMGBB_API_KEY=...

Install:
    py -m pip install fastapi uvicorn requests python-dotenv python-multipart

Run:
    uvicorn app:app --reload

Open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

# Persistent export directory (avoids temp cleanup races with FileResponse)
EXPORTS_DIR = Path(__file__).parent / "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

A1D_ENDPOINT = "https://api.a1d.ai/api/image-upscaler"
IMGBB_UPLOAD_ENDPOINT = "https://api.imgbb.com/1/upload"

# Reasonable timeouts: (connect, read)
HTTP_TIMEOUT: Tuple[int, int] = (15, 120)


class BatchUpscaleError(RuntimeError):
    """Raised for known failures in the batch upscaling pipeline."""


@dataclass(frozen=True)
class Settings:
    a1d_api_key: str
    imgbb_api_key: str
    concurrency: int = 2  # limited concurrency to reduce rate-limit risk
    scale: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        a1d_api_key = (os.getenv("A1D_API_KEY") or "").strip()
        imgbb_api_key = (os.getenv("IMGBB_API_KEY") or "").strip()

        if not a1d_api_key:
            raise ValueError("Missing A1D_API_KEY in environment/.env")
        if not imgbb_api_key:
            raise ValueError("Missing IMGBB_API_KEY in environment/.env")

        # Optional overrides
        concurrency = int((os.getenv("CONCURRENCY") or "2").strip())
        scale = int((os.getenv("SCALE") or "2").strip())

        # Safety caps
        concurrency = max(1, min(concurrency, 5))
        scale = max(1, scale)

        return cls(
            a1d_api_key=a1d_api_key,
            imgbb_api_key=imgbb_api_key,
            concurrency=concurrency,
            scale=scale,
        )


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        preview = (resp.text or "").strip()
        raise ValueError(f"Non-JSON response (status {resp.status_code}): {preview}")
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON type: {type(data).__name__}")
    return data


def _raise_for_status_with_details(resp: requests.Response, *, service_name: str) -> None:
    if 200 <= resp.status_code < 300:
        return
    try:
        detail = json.dumps(resp.json(), ensure_ascii=False)
    except ValueError:
        detail = (resp.text or "").strip()
    raise HTTPError(f"{service_name} HTTP {resp.status_code}: {detail}")


def upload_to_imgbb(*, imgbb_api_key: str, filename: str, content: bytes) -> str:
    """
    Upload image bytes to ImgBB to obtain a publicly accessible URL.

    ImgBB API expects multipart form-data with:
      - key: your API key
      - image: base64 OR binary file content (we use multipart file upload)
    """
    files = {
        "image": (filename, content),
    }
    data = {
        "key": imgbb_api_key,
    }

    try:
        resp = requests.post(
            IMGBB_UPLOAD_ENDPOINT,
            data=data,
            files=files,
            timeout=HTTP_TIMEOUT,
        )
        _raise_for_status_with_details(resp, service_name="ImgBB")
        payload = _safe_json(resp)

        # Expected: payload["data"]["url"] (or "display_url")
        url = (
            payload.get("data", {}).get("url")
            or payload.get("data", {}).get("display_url")
        )
        if not url or not isinstance(url, str):
            raise BatchUpscaleError(f"ImgBB response missing URL: {json.dumps(payload, ensure_ascii=False)}")

        return url

    except (Timeout, ConnectionError) as exc:
        raise BatchUpscaleError(f"ImgBB network error: {exc}") from exc
    except HTTPError as exc:
        raise BatchUpscaleError(str(exc)) from exc
    except RequestException as exc:
        raise BatchUpscaleError(f"ImgBB request failed: {exc}") from exc


def call_a1d_upscaler_start(*, a1d_api_key: str, image_url: str, scale: int) -> Dict[str, Any]:
    """
    Starts an A1D upscaling task.

    NOTE: Per your observation, this endpoint returns a taskId and does not immediately
    return the final upscaled image URL.
    """
    headers = {
        "Authorization": f"KEY {a1d_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "imageUrl": image_url,
        "scale": scale,
        "source": "api",
    }

    try:
        resp = requests.post(
            A1D_ENDPOINT,
            headers=headers,
            json=body,
            timeout=HTTP_TIMEOUT,
        )
        _raise_for_status_with_details(resp, service_name="A1D")
        return _safe_json(resp)

    except (Timeout, ConnectionError) as exc:
        raise BatchUpscaleError(f"A1D network error: {exc}") from exc
    except HTTPError as exc:
        raise BatchUpscaleError(str(exc)) from exc
    except RequestException as exc:
        raise BatchUpscaleError(f"A1D request failed: {exc}") from exc


def poll_a1d_task_result(
    *,
    a1d_api_key: str,
    task_id: str,
    poll_interval_seconds: int = 3,
    max_attempts: int = 40,
) -> Dict[str, Any]:
    """
    Poll A1D task status endpoint until completion (or timeout).

    Typical endpoint (per requirement):
        GET https://api.a1d.ai/api/task/{taskId}

    Behavior:
      - If status indicates pending/processing, sleep and retry.
      - If completed, return the final JSON payload.
      - If failed, raise with the API's error payload.
      - Timeout after max_attempts to avoid infinite loops.
    """
    if not task_id or not task_id.strip():
        raise BatchUpscaleError("Missing/empty taskId from A1D start response.")

    url = f"https://api.a1d.ai/api/task/{task_id.strip()}"
    headers = {
        "Authorization": f"KEY {a1d_api_key}",
        "Content-Type": "application/json",
    }

    last_payload: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            _raise_for_status_with_details(resp, service_name="A1D Task")
            payload = _safe_json(resp)
            last_payload = payload
        except (Timeout, ConnectionError) as exc:
            # transient error: continue polling unless we've exhausted attempts
            if attempt >= max_attempts:
                raise BatchUpscaleError(f"A1D task polling network error (final attempt): {exc}") from exc
            time.sleep(poll_interval_seconds)
            continue
        except HTTPError as exc:
            raise BatchUpscaleError(str(exc)) from exc
        except RequestException as exc:
            raise BatchUpscaleError(f"A1D task polling failed: {exc}") from exc

        # Interpret status
        status_val = payload.get("status") or payload.get("state")
        if isinstance(status_val, str):
            status = status_val.strip().lower()
        else:
            status = ""

        # A1D success state (per your payload): "FINISHED"
        if status == "finished":
            # Ensure expected result field exists (per your payload: imageUrl)
            image_url = payload.get("imageUrl")
            if isinstance(image_url, str) and image_url.strip().startswith("http"):
                return payload
            raise BatchUpscaleError(
                "A1D task status FINISHED but missing imageUrl: "
                + json.dumps(payload, ensure_ascii=False)
            )

        if status in {"failed", "error", "canceled", "cancelled"}:
            raise BatchUpscaleError(
                "A1D task failed: " + json.dumps(payload, ensure_ascii=False)
            )

        # Default: pending/processing/unknown -> wait and retry
        if attempt < max_attempts:
            time.sleep(poll_interval_seconds)

    raise BatchUpscaleError(
        "A1D task polling timed out after "
        f"{max_attempts} attempts (~{max_attempts * poll_interval_seconds}s). "
        f"Last payload: {json.dumps(last_payload or {}, ensure_ascii=False)}"
    )


def extract_output_image_url(a1d_response: Dict[str, Any]) -> str:
    """
    Attempt to extract upscaled image URL from A1D response.

    Because API response shape can vary, we check common keys.
    Adjust this function if the API returns a different schema.
    """
    candidates: List[Optional[str]] = [
        a1d_response.get("upscaledImageUrl"),  # common
        a1d_response.get("imageUrl"),          # sometimes reused
        a1d_response.get("resultUrl"),
        a1d_response.get("url"),
    ]

    # Some APIs nest under "data"
    data = a1d_response.get("data")
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("upscaledImageUrl"),
                data.get("imageUrl"),
                data.get("resultUrl"),
                data.get("url"),
                data.get("outputUrl"),
                data.get("output"),
            ]
        )

    for c in candidates:
        if isinstance(c, str) and c.strip().startswith("http"):
            return c.strip()

    raise BatchUpscaleError(
        "Could not find upscaled image URL in A1D response. "
        f"Response: {json.dumps(a1d_response, ensure_ascii=False)}"
    )


def download_file(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        _raise_for_status_with_details(resp, service_name="Download")
        return resp.content
    except (Timeout, ConnectionError) as exc:
        raise BatchUpscaleError(f"Download network error: {exc}") from exc
    except HTTPError as exc:
        raise BatchUpscaleError(str(exc)) from exc
    except RequestException as exc:
        raise BatchUpscaleError(f"Download failed: {exc}") from exc


def sanitize_filename(name: str) -> str:
    """
    Basic filename sanitization for safe zip writing.
    """
    keep = "._- "
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in keep).strip()
    return cleaned or f"image_{uuid4().hex}.png"


async def process_one_image(
    *,
    settings: Settings,
    upload: UploadFile,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
    index: int,
) -> Dict[str, Any]:
    """
    Process a single image end-to-end.
    Runs blocking network calls in a thread to avoid blocking the event loop.
    """
    async with semaphore:
        original_name = upload.filename or f"image_{index}.png"
        original_basename = sanitize_filename(Path(original_name).name)

        # Preserve the user's original filename in the final output.
        # If a name collision happens, append a short suffix.
        out_name = original_basename
        out_path = out_dir / out_name
        if out_path.exists():
            stem = Path(original_basename).stem
            suffix = Path(original_basename).suffix or ".png"
            out_name = f"{stem}_upscaled_{uuid4().hex[:8]}{suffix}"

        out_path = out_dir / out_name

        try:
            content = await upload.read()
            if not content:
                raise BatchUpscaleError("Empty file")

            # 1) Upload to ImgBB -> public URL
            public_url = await asyncio.to_thread(
                upload_to_imgbb,
                imgbb_api_key=settings.imgbb_api_key,
                filename=original_basename,
                content=content,
            )

            # 2) Start A1D task -> returns taskId
            start_resp = await asyncio.to_thread(
                call_a1d_upscaler_start,
                a1d_api_key=settings.a1d_api_key,
                image_url=public_url,
                scale=settings.scale,
            )

            task_id = start_resp.get("taskId") or start_resp.get("task_id") or start_resp.get("id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise BatchUpscaleError(
                    "A1D start response missing taskId: "
                    + json.dumps(start_resp, ensure_ascii=False)
                )

            # 3) Poll A1D task result until completed (or timeout)
            final_resp = await asyncio.to_thread(
                poll_a1d_task_result,
                a1d_api_key=settings.a1d_api_key,
                task_id=task_id,
                poll_interval_seconds=3,
                max_attempts=40,
            )

            # 4) Extract upscaled URL + download the file
            upscaled_url = extract_output_image_url(final_resp)
            upscaled_bytes = await asyncio.to_thread(download_file, upscaled_url)

            # 4) Save locally
            # Save with the preserved original filename (or derived _upscaled_ name).
            out_path.write_bytes(upscaled_bytes)

            return {
                "filename": out_name,
                "original_filename": original_basename,
                "status": "success",
                "public_url": public_url,
                "upscaled_url": upscaled_url,
            }

        except Exception as exc:
            # Continue on error; return error info
            return {
                "filename": out_name,
                "original_filename": original_basename,
                "status": "error",
                "error": str(exc),
            }
        finally:
            await upload.close()


def make_zip_from_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.glob("*")):
            if p.is_file():
                zf.write(p, arcname=p.name)


app = FastAPI(title="A1D Bulk Image Upscaler")

# Serve index.html and any static assets from ./static
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>Missing static/index.html</h1><p>Create static/index.html and refresh.</p>",
            status_code=500,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/batch-upscale")
async def batch_upscale(
    files: List[UploadFile] = File(...),
    # Optional override per request (frontend can send it)
    scale: int = Form(2),
    concurrency: int = Form(2),
) -> JSONResponse:
    """
    Accepts multiple files (from folder selection) and processes them with limited concurrency.
    Returns a JSON payload containing:
      - results per file
      - a downloadUrl for a zip (if any successes)
    """
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Apply safe overrides
    settings = Settings(
        a1d_api_key=settings.a1d_api_key,
        imgbb_api_key=settings.imgbb_api_key,
        concurrency=max(1, min(int(concurrency), 5)),
        scale=max(1, int(scale)),
    )

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Create per-request temp working directory for intermediate files
    request_id = uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"batch_{request_id}_"))
    out_dir = work_dir / "upscaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(settings.concurrency)

    # Process with limited concurrency
    tasks = [
        process_one_image(
            settings=settings,
            upload=f,
            out_dir=out_dir,
            semaphore=semaphore,
            index=i,
        )
        for i, f in enumerate(files, start=1)
    ]

    results = await asyncio.gather(*tasks)

    successes = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") == "error"]

    zip_url: Optional[str] = None
    if successes:
        # Write ZIP to a persistent exports directory so it is guaranteed to exist
        # when FileResponse tries to read it.
        export_zip_path = EXPORTS_DIR / f"upscaled_{request_id}.zip"
        make_zip_from_dir(out_dir, export_zip_path)
        zip_url = f"/api/download/{request_id}"

        # Cleanup intermediate temp work dir (safe; ZIP already persisted)
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        # No successes -> cleanup immediately
        shutil.rmtree(work_dir, ignore_errors=True)

    return JSONResponse(
        {
            "requestId": request_id,
            "total": len(files),
            "successCount": len(successes),
            "errorCount": len(errors),
            "results": results,
            "downloadUrl": zip_url,
        }
    )


@app.get("/api/download/{request_id}")
def download_zip(request_id: str) -> FileResponse:
    """
    Downloads the ZIP for a completed batch from the persistent `exports/` folder.

    Note: We intentionally do not delete the ZIP yet (leave it for verification).
    """
    zip_path = EXPORTS_DIR / f"upscaled_{request_id}.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Zip not found in exports folder.")

    return FileResponse(
        path=str(zip_path),
        filename="upscaled_images.zip",
        media_type="application/zip",
    )
