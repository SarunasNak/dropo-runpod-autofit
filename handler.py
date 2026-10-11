import base64
import io
import time

import runpod
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from rembg import new_session, remove


CANVAS_SIZE = 2000
AI_MAX_SIZE = 2000
BACKGROUND = (237, 241, 239)  # #EDF1EF
MARGIN = 150
JPEG_QUALITY = 90
MAX_IMAGE_BYTES = 25 * 1024 * 1024


print("Loading BiRefNet model...", flush=True)
model_started = time.time()
session = new_session("birefnet-general")
print(
    f"BiRefNet loaded in {time.time() - model_started:.1f}s",
    flush=True,
)


# ============================================================
# SMART IMAGE CLASSIFIER (CLIP)
# ============================================================

print("Loading CLIP classifier...", flush=True)
clip_started = time.time()
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

SMART_PROMPTS = [
    "A product photograph of a complete pair of jeans.",
    "A product photograph of a complete jacket.",
    "A product photograph of a complete shirt.",
    "A product photograph of a complete coat.",
    "A product photograph of a complete sweater.",
    "A product photograph of a complete pair of trousers.",
    "A close-up photograph of a clothing brand label.",
    "A close-up photograph of a clothing size or care tag.",
    "A close-up photograph of denim fabric texture.",
    "A close-up photograph of clothing stitching or a seam.",
    "A close-up photograph of a button or zipper on clothing.",
    "A close-up photograph of a clothing defect or damage.",
    "A close-up photograph of a clothing pocket or logo.",
]
SMART_FULL_COUNT = 6
# Conservative starting value, not a validated accuracy guarantee.
SMART_FULL_THRESHOLD = 0.75

with torch.inference_mode():
    clip_text_inputs = clip_processor(
        text=SMART_PROMPTS, return_tensors="pt", padding=True
    )
    clip_text_features = clip_model.get_text_features(**clip_text_inputs)
    clip_text_features = clip_text_features / clip_text_features.norm(
        dim=-1, keepdim=True
    )

print(f"CLIP loaded in {time.time() - clip_started:.1f}s", flush=True)


def detect_image_type(source: Image.Image) -> str:
    image_inputs = clip_processor(images=source, return_tensors="pt")
    with torch.inference_mode():
        image_features = clip_model.get_image_features(
            pixel_values=image_inputs["pixel_values"]
        )
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        logits = 100.0 * (image_features @ clip_text_features.T)
        probabilities = logits.softmax(dim=-1)[0]

    full_score = probabilities[:SMART_FULL_COUNT].sum().item()
    detail_score = probabilities[SMART_FULL_COUNT:].sum().item()
    selected = "auto" if full_score >= SMART_FULL_THRESHOLD else "simple"
    print(
        f"SMART: full={full_score:.3f}, detail={detail_score:.3f}, "
        f"selected={selected}", flush=True
    )
    return selected


# ============================================================
# HELPERS
# ============================================================

def load_source_image(image_bytes: bytes) -> Image.Image:
    """
    Atidaro originalią nuotrauką ir sutvarko EXIF orientation.
    """

    try:
        source = Image.open(io.BytesIO(image_bytes))
        source = ImageOps.exif_transpose(source).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Input is not a valid image.") from exc

    if source.width < 10 or source.height < 10:
        raise ValueError("Image dimensions are too small.")

    return source


def save_jpeg(image: Image.Image) -> bytes:
    """
    Išsaugo galutinį rezultatą kaip JPEG.
    """

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
        progressive=True,
    )

    return output.getvalue()


# ============================================================
# AUTO MODE
# Visos SMART atpazintos pilno drabuzio nuotraukos
# BiRefNet + background removal + Auto Fit
# ============================================================

def process_auto(source: Image.Image) -> tuple[bytes, str]:

    ai_image = source.copy()

    ai_image.thumbnail(
        (AI_MAX_SIZE, AI_MAX_SIZE),
        Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------------
    # AI background removal
    # --------------------------------------------------------

    cutout = remove(
        ai_image,
        session=session,
    ).convert("RGBA")

    alpha = cutout.getchannel("A")

    alpha = alpha.filter(
        ImageFilter.MinFilter(5)
    )

    cutout.putalpha(alpha)

    bbox = alpha.getbbox()

    if bbox is None:
        raise ValueError(
            "AI could not detect a product in the image."
        )

    cutout = cutout.crop(bbox)

    if cutout.width <= 0 or cutout.height <= 0:
        raise ValueError(
            "Detected product area is empty."
        )

    # --------------------------------------------------------
    # Nustatome LONG / NORMAL
    # --------------------------------------------------------

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
        f"AUTO: detected object "
        f"{cutout.width}x{cutout.height}, "
        f"ratio={aspect_ratio:.2f}, "
        f"mode={mode}",
        flush=True,
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    scale = min(
        max_width / cutout.width,
        max_height / cutout.height,
    )

    new_width = max(
        1,
        round(cutout.width * scale),
    )

    new_height = max(
        1,
        round(cutout.height * scale),
    )

    cutout = cutout.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    # --------------------------------------------------------
    # Background canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "RGB",
        (CANVAS_SIZE, CANVAS_SIZE),
        BACKGROUND,
    )

    x = (
        CANVAS_SIZE - cutout.width
    ) // 2

    y = (
        CANVAS_SIZE - cutout.height
    ) // 2

    canvas.paste(
        cutout,
        (x, y),
        cutout,
    )

    return save_jpeg(canvas), mode



def process_simple(source: Image.Image) -> tuple[bytes, str]:
    """
    DETAIL režimas:
    - Jokio BiRefNet
    - Jokio fono šalinimo
    - Išsaugomas visas originalus kadras
    - 2000x2000 rezultatas
    - Foną užpildo iš originalo sukurta
      išplaukinta versija
    """

    # Originalus kadras
    original = source.copy().convert("RGB")

    # Fonui naudojame tą pačią nuotrauką,
    # kad nebūtų svetimos spalvos stačiakampio
    background = ImageOps.fit(
        original,
        (CANVAS_SIZE, CANVAS_SIZE),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )

    # Išplaukiname tik foną
    background = background.filter(
        ImageFilter.GaussianBlur(radius=65)
    )

    # Originalą sutalpiname be apkirpimo
    fitted = original.copy()
    fitted.thumbnail(
        (CANVAS_SIZE, CANVAS_SIZE),
        Image.Resampling.LANCZOS,
    )

    x = (CANVAS_SIZE - fitted.width) // 2
    y = (CANVAS_SIZE - fitted.height) // 2

    background.paste(fitted, (x, y))

    print(
        f"DETAIL: {source.width}x{source.height} "
        f"-> {fitted.width}x{fitted.height}, "
        f"no AI, no crop",
        flush=True,
    )

    return save_jpeg(background), "DETAIL"



# ============================================================
# MAIN PROCESSOR
# ============================================================

def process_image(
    image_bytes: bytes,
    requested_mode: str = "smart",
) -> tuple[bytes, str]:

    source = load_source_image(
        image_bytes
    )

    requested_mode = (
        requested_mode or "smart"
    ).strip().lower()

    # SMART selects processing automatically for every image.
    if requested_mode == "smart":
        requested_mode = detect_image_type(source)

    # --------------------------------------------------------
    # SIMPLE
    # --------------------------------------------------------

    if requested_mode == "simple":

        return process_simple(
            source
        )

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    if requested_mode == "auto":

        return process_auto(
            source
        )

    raise ValueError(
        f"Unsupported processing mode: {requested_mode}"
    )


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(event):

    started = time.time()

    try:

        job_input = (
            event.get("input") or {}
        )

        image_b64 = job_input.get(
            "image_base64"
        )

        requested_mode = (
            job_input.get("mode") or "smart"
        )

        # ----------------------------------------------------
        # Validate mode
        # ----------------------------------------------------

        if requested_mode not in {
            "auto",
            "simple",
            "smart",
        }:

            return {
                "ok": False,
                "error": (
                    f"Invalid mode: "
                    f"{requested_mode}"
                ),
            }

        print(
            f"Processing image with "
            f"requested mode={requested_mode}",
            flush=True,
        )

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if (
            not image_b64
            or not isinstance(image_b64, str)
        ):

            return {
                "ok": False,
                "error": (
                    "Missing input.image_base64"
                ),
            }

        # Priimame ir:
        #
        # gryną base64
        #
        # ir:
        #
        # data:image/jpeg;base64,...
        #

        if (
            image_b64.startswith("data:")
            and "," in image_b64
        ):

            image_b64 = image_b64.split(
                ",",
                1,
            )[1]

        try:

            image_bytes = base64.b64decode(
                image_b64,
                validate=True,
            )

        except Exception:

            return {
                "ok": False,
                "error": "Invalid base64 image",
            }

        if not image_bytes:

            return {
                "ok": False,
                "error": "Image is empty",
            }

        if len(image_bytes) > MAX_IMAGE_BYTES:

            return {
                "ok": False,
                "error": (
                    "Image exceeds 25 MB limit"
                ),
            }

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        result, actual_mode = process_image(
            image_bytes,
            requested_mode=requested_mode,
        )

        seconds = (
            time.time() - started
        )

        print(
            f"Finished mode={requested_mode} "
            f"in {seconds:.2f}s",
            flush=True,
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "ok": True,
            "image_base64": base64.b64encode(
                result
            ).decode("ascii"),
            "content_type": "image/jpeg",
            "mode": actual_mode,
            "requested_mode": requested_mode,
            "processing_seconds": round(
                seconds,
                2,
            ),
        }

    except ValueError as exc:

        return {
            "ok": False,
            "error": str(exc),
        }

    except Exception as exc:

        print(
            f"Auto Fit error: {exc}",
            flush=True,
        )

        return {
            "ok": False,
            "error": (
                "Image processing failed"
            ),
        }


# ============================================================
# START RUNPOD SERVERLESS
# ============================================================

if __name__ == "__main__":

    runpod.serverless.start(
        {
            "handler": handler,
        }
    )
