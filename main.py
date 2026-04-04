from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, uuid, os, tempfile
from hdr_merge import merge_hdr
from tone_mapper import tone_map_real_estate

app = FastAPI(title="HDR Real Estate API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RAW_EXTENSIONS = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2'}

def decode_image(contents: bytes, filename: str) -> np.ndarray:
    ext = os.path.splitext(filename.lower())[1]
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            with rawpy.imread(tmp_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    half_size=False
                )
            os.unlink(tmp_path)
            rgb_8bit = (rgb / 256).astype(np.uint8)
            return cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"RAW decode error: {e}")
            return None
    else:
        arr = np.frombuffer(contents, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.get("/")
def root():
    return {"status": "HDR API running"}

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
            "error": f"Could not decode enough images. Decoded {len(images)}/{len(files)}"
        })

    merged = merge_hdr(images)
    final = tone_map_real_estate(merged)

    # Encode directly to JPEG in memory — no disk needed
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 97]
    success, buffer = cv2.imencode('.jpg', final, encode_params)

    if not success:
        return JSONResponse(status_code=500, content={"error": "Failed to encode image"})

    # Return image directly in response
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "attachment; filename=hdr_result.jpg",
            "X-Resolution": f"{final.shape[1]}x{final.shape[0]}",
            "X-Images-Used": str(len(images))
        }
    )
