import uuid
import requests
from fastapi import APIRouter, UploadFile, File
from config import supabase

router = APIRouter()

ROBOFLOW_API_KEY = "mtvd7s7PwBMsrJZnnRCq"
ROBOFLOW_UPLOAD_URL = f"https://api.roboflow.com/dataset/disease-0ge01/upload?api_key={ROBOFLOW_API_KEY}"

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Step 1: Insert into sources table
    source = supabase.table("sources").insert({"kind": "upload"}).execute()
    source_id = source.data[0]["id"]

    # Step 2: Create unique filename
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    storage_path = f"{source_id}/{unique_name}"

    # Read file content
    file_bytes = await file.read()

    # Step 3: Upload file to Supabase storage
    supabase.storage.from_("images_for_annotation").upload(storage_path, file_bytes, {"upsert": "true"})
    public_url = supabase.storage.from_("images_for_annotation").get_public_url(storage_path)

    # Step 4: Forward file to Roboflow
    files = {
        "file": (file.filename, file_bytes, file.content_type),
    }
    data = {
        "name": file.filename,
        "split": "train"
    }
    rf_response = requests.post(ROBOFLOW_UPLOAD_URL, files=files, data=data)
    rf_result = rf_response.json()

    # Step 5: Insert into source_uploads table
    supabase.table("source_uploads").insert({
        "source_id": source_id,
        "filename": file.filename,
        "storage_url": public_url
    }).execute()

    return {
        "source_id": source_id,
        "filename": file.filename,
        "supabase_url": public_url,
        "roboflow_result": rf_result
    }
