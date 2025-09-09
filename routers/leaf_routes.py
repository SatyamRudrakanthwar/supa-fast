from fastapi import APIRouter, HTTPException, Query
from PIL import Image
import requests
import tempfile
import os
import uuid
from config import supabase, SUPABASE_URL
from io import BytesIO
from core.leaf_extraction import run_inference_from_pil


# Supabase connection


router = APIRouter()

@router.post("/extract-leaves")
async def extract_leaves(source_id: str = Query(..., description="Source ID from sources table")):
    """
    Runs leaf extraction on an image from any source (upload, url, zip).
    Stores results in leaf_results + logs run in analysis_log.
    """
    try:
        # Find the source type
        source_res = supabase.table("sources").select("*").eq("id", source_id).execute()
        if not source_res.data:
            raise HTTPException(status_code=404, detail="Source not found")
        source = source_res.data[0]
        kind = source["kind"]

        pil_image = None

        # Fetch image based on kind
        if kind == "upload":
            upload_res = supabase.table("source_uploads").select("*").eq("source_id", source_id).execute()
            if not upload_res.data:
                raise HTTPException(status_code=404, detail="Upload not found")
            url = upload_res.data[0]["storage_url"]
            response = requests.get(url)
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

        elif kind == "url":
            url_res = supabase.table("source_urls").select("*").eq("source_id", source_id).execute()
            if not url_res.data:
                raise HTTPException(status_code=404, detail="URL not found")
            url = url_res.data[0]["url"]
            response = requests.get(url)
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

        elif kind == "zip":
            zip_res = supabase.table("zip_images").select("*").eq("zip_id", source_id).execute()
            if not zip_res.data:
                raise HTTPException(status_code=404, detail="No images found in zip")
            # For now → just take the first image from zip
            url = zip_res.data[0]["storage_url"]
            response = requests.get(url)
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source kind: {kind}")

        # Run leaf extraction
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_paths, err = run_inference_from_pil(pil_image, output_dir=output_dir)

        if err:
            raise HTTPException(status_code=400, detail=err)

        # Upload extracted images to Supabase Storage
        storage_urls = []
        for path in output_paths:
            filename = f"leaf/{uuid.uuid4()}.png"
            with open(path, "rb") as f:
                supabase.storage.from_("leaf_extracted").upload(filename, f)
            url = f"{SUPABASE_URL}/storage/v1/object/public/leaf_extracted/{filename}"
            storage_urls.append(url)

        leaf_count = len(storage_urls)

        # Insert into leaf_extracted
        res = supabase.table("leaf_extracted").insert({
            "source_id": source_id,
            "leaf_count": leaf_count,
            "extracted_urls": storage_urls
        }).execute()

        if not res.data:
            raise HTTPException(status_code=500, detail="Failed to insert leaf results")

        result_id = res.data[0]["id"]

        # Log into analysis_log
        supabase.table("analysis_log").insert({
            "source_id": source_id,
            "module": "leaf",
            "result_id": result_id
        }).execute()

        return {
            "leaf_count": leaf_count,
            "extracted_leaf_images": storage_urls
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
