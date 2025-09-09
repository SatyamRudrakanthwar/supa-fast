import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from config import supabase

router = APIRouter()

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Step 1: Insert into sources table
        source = supabase.table("sources").insert({"kind": "upload"}).execute()
        source_id = source.data[0]["id"]

        # Step 2: Create unique filename
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        storage_path = f"{source_id}/{unique_name}"

        # Step 3: Upload file to Supabase storage (note: .storage not .storage())
        file_bytes = await file.read()
        supabase.storage.from_("images").upload(storage_path, file_bytes)

        # Get public URL
        public_url = supabase.storage.from_("images").get_public_url(storage_path)

        # Step 4: Insert into source_uploads table
        supabase.table("source_uploads").insert({
            "source_id": source_id,
            "filename": file.filename,
            "storage_url": public_url
        }).execute()

        return {
            "source_id": source_id,
            "filename": file.filename,
            "url": public_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
