from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import cv2
import numpy as np
import base64
import os
from core.image_processor import process_image
from config import supabase
import uuid
import json


router = APIRouter()

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode()
    return img_str

def save_to_supabase(image_np, image_name, pest_counts, source_id=None):
    # Convert image to bytes
    _, buffer = cv2.imencode('.jpg', image_np)
    img_bytes = buffer.tobytes()

    # Upload to Supabase Storage
    file_path = f"pest_results/{uuid.uuid4()}_{image_name}"
    supabase.storage.from_("analysis-results").upload(file_path, img_bytes)

    # Get public URL
    public_url = supabase.storage.from_("analysis-results").get_public_url(file_path)

    # Insert into DB
    data = {
        "source_id": source_id,
        "image_name": image_name,
        "pest_counts": json.dumps(pest_counts),
        "annotated_image_url": public_url
    }
    supabase.table("pest_results").insert(data).execute()

    return public_url


@router.post("/detect-pests")
async def detect_pests(image: UploadFile = File(...), source_id: str = None):
    try:
        # Save uploaded file to temp
        suffix = os.path.splitext(image.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            temp_path = tmp.name

        # Run pest detection
        annotated_image, pest_counts, error = process_image(temp_path)
        os.unlink(temp_path)

        if error:
            raise HTTPException(status_code=400, detail=error)

        # Save to Supabase
        annotated_url = save_to_supabase(
            annotated_image,
            image.filename,
            pest_counts,
            source_id
        )

        # Encode annotated image as base64 (still return for frontend preview)
        annotated_base64 = encode_image_to_base64(annotated_image)

        return {
            "pest_counts": pest_counts,
            "annotated_image_base64": annotated_base64,
            "annotated_image_url": annotated_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
