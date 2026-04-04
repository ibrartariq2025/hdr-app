from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, uuid, os, tempfile
from hdr_merge import merge_hdr
from tone_mapper import tone_map_real_estate

app = FastAPI(title="HDR Real Estate API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

RAW_EXTENSIONS = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2'}

def decode_image(contents: bytes, filename: str) -> np.ndarray:
    """Decode both RAW and JPEG/PNG files to numpy array"""
    ext = os.path.splitext(filename.lower())[1]
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            with rawpy.imread(tmp_path) as raw:
                # Use camera white balance, no auto-bright for HDR accuracy
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    half_size=False
                )
            os.unlink(tmp_path)
            # Convert 16-bit to 8-bit for processing
            rgb_8bit = (rgb / 256).astype(np.uint8)
            return cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"RAW decode error: {e}")
            return None
    else:
        arr = np.frombuffer(contents, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def save_as_tiff(img: np.ndarray, path: str):
    """Save as 16-bit TIFF (highest quality lossless output)"""
    img_16 = (img.astype(np.float32) * 257).astype(np.uint16)
    cv2.imwrite(path, img_16)

@app.get("/")
def root():
    return {"status": "HDR API running — RAW support enabled"}

@app.post("/process")
async def process_hdr(files: list[UploadFile] = File(...)):
    images = []
    for f in files:
        contents = await f.read()
        img = decode_image(contents, f.filename)
        if img is not None:
            images.append(img)
        else:
            print(f"Could not decode: {f.filename}")

    if len(images) < 2:
        return JSONResponse(status_code=400, content={
            "error": f"Could not decode enough images. Decoded {len(images)}/{ len(files)}"
        })

    merged = merge_hdr(images)
    final = tone_map_real_estate(merged)

    job_id = str(uuid.uuid4())[:8]

    # Save full resolution TIFF (lossless)
    tiff_path = f"outputs/{job_id}_hdr.tiff"
    save_as_tiff(final, tiff_path)

    # Also save JPEG preview
    jpg_path = f"outputs/{job_id}_hdr.jpg"
    cv2.imwrite(jpg_path, final, [cv2.IMWRITE_JPEG_QUALITY, 98])

    return {
        "job_id": job_id,
        "images_used": len(images),
        "resolution": f"{final.shape[1]}x{final.shape[0]}",
        "outputs": {
            "tiff": f"/download/{job_id}/tiff",
            "jpeg": f"/download/{job_id}/jpeg"
        }
    }

@app.get("/download/{job_id}/tiff")
def download_tiff(job_id: str):
    path = f"outputs/{job_id}_hdr.tiff"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(path, media_type="image/tiff", filename=f"{job_id}_hdr.tiff")

@app.get("/download/{job_id}/jpeg")
def download_jpeg(job_id: str):
    path = f"outputs/{job_id}_hdr.jpg"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(path, media_type="image/jpeg", filename=f"{job_id}_hdr.jpg")
