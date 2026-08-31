import base64
import io
import time

import runpod
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from rembg import new_session, remove

CANVAS_SIZE = 2000
AI_MAX_SIZE = 2000
BACKGROUND = (237, 241, 239)  # #EDF1EF
MARGIN = 300
JPEG_QUALITY = 90
MAX_IMAGE_BYTES = 25 * 1024 * 1024

print("Loading BiRefNet model...", flush=True)
model_started = time.time()
session = new_session("birefnet-general")
print(f"BiRefNet loaded in {time.time() - model_started:.1f}s", flush=True)


def process_image(image_bytes: bytes) -> tuple[bytes, str]:
    try:
        source = Image.open(io.BytesIO(image_bytes))
        source = ImageOps.exif_transpose(source).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Input is not a valid image.") from exc

    if source.width < 10 or source.height < 10:
        raise ValueError("Image dimensions are too small.")

    ai_image = source.copy()
    ai_image.thumbnail((AI_MAX_SIZE, AI_MAX_SIZE), Image.Resampling.LANCZOS)

    cutout = remove(ai_image, session=session).convert("RGBA")

    alpha = cutout.getchannel("A")
    alpha = alpha.filter(ImageFilter.MinFilter(5))
    cutout.putalpha(alpha)

    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("AI could not detect a product in the image.")

    cutout = cutout.crop(bbox)
    if cutout.width <= 0 or cutout.height <= 0:
        raise ValueError("Detected product area is empty.")

    aspect_ratio = cutout.height / cutout.width
    is_long = aspect_ratio >= 1.65

    if is_long:
        max_width = int(CANVAS_SIZE * 0.85)
        max_height = int(CANVAS_SIZE * 0.94)
        mode = "LONG"
    else:
        max_width = CANVAS_SIZE - (MARGIN * 2)
        max_height = CANVAS_SIZE - (MARGIN * 2)
        mode = "NORMAL"

    print(
        f"Detected object: {cutout.width}x{cutout.height}, "
        f"ratio={aspect_ratio:.2f}, mode={mode}",
        flush=True,
    )

    scale = min(max_width / cutout.width, max_height / cutout.height)
    new_width = max(1, round(cutout.width * scale))
    new_height = max(1, round(cutout.height * scale))

    cutout = cutout.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND)
    x = (CANVAS_SIZE - cutout.width) // 2
    y = (CANVAS_SIZE - cutout.height) // 2
    canvas.paste(cutout, (x, y), cutout)

    output = io.BytesIO()
    canvas.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
        progressive=True,
    )
    return output.getvalue(), mode


def handler(event):
    started = time.time()

    try:
        job_input = event.get("input") or {}
        image_b64 = job_input.get("image_base64")

        if not image_b64 or not isinstance(image_b64, str):
            return {"ok": False, "error": "Missing input.image_base64"}

        # Priimame ir gryną base64, ir data:image/...;base64,...
        if image_b64.startswith("data:") and "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except Exception:
            return {"ok": False, "error": "Invalid base64 image"}

        if not image_bytes:
            return {"ok": False, "error": "Image is empty"}

        if len(image_bytes) > MAX_IMAGE_BYTES:
            return {"ok": False, "error": "Image exceeds 25 MB limit"}

        result, mode = process_image(image_bytes)
        seconds = time.time() - started

        return {
            "ok": True,
            "image_base64": base64.b64encode(result).decode("ascii"),
            "content_type": "image/jpeg",
            "mode": mode,
            "processing_seconds": round(seconds, 2),
        }

    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        print(f"Auto Fit error: {exc}", flush=True)
        return {"ok": False, "error": "Image processing failed"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
