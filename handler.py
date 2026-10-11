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
