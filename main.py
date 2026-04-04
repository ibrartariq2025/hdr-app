from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, uuid, os, shutil
from hdr_merge import merge_hdr
from tone_mapper import tone_map_real_estate

app = FastAPI(title="HDR Real Estate API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

@app.get("/")
def root():
    return {"status": "HDR API running"}

@app.post("/process")
async def process_hdr(files: list[UploadFile] = File(...)):
    if len(files) < 2:
        return JSONResponse(status_code=400, content={"error": "Upload at least 2 bracketed images"})

    job_id = str(uuid.uuid4())[:8]
    images = []
    for f in files:
        contents = await f.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return JSONResponse(status_code=400, content={"error": "Could not decode images"})

    merged = merge_hdr(images)
    final = tone_map_real_estate(merged)

    out_path = f"outputs/{job_id}_hdr.jpg"
    cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {"job_id": job_id, "output": out_path, "images_used": len(images)}

@app.get("/download/{job_id}")
def download(job_id: str):
    path = f"outputs/{job_id}_hdr.jpg"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(path, media_type="image/jpeg", filename=f"{job_id}_hdr.jpg")
