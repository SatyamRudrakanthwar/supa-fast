import os
import shutil
import tempfile
import zipfile
from supabase import create_client, Client
from config import supabase
from fastapi import APIRouter, File, HTTPException, UploadFile
from postgrest import APIError

router = APIRouter()

@router.post("/upload-file/")
async def upload_file(zip_file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    source_id = None
    zip_id = None

    try:
        # Save uploaded ZIP to temporary directory
        zip_path = os.path.join(tmp_dir, zip_file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)

        # Insert record into sources
        source_response = supabase.table("sources").insert({"kind": "zip"}).execute()
        if not source_response.data:
            raise HTTPException(status_code=500, detail="Failed to create source entry.")
        source_id = source_response.data[0]["id"]

        # Insert record into source_zips
        zip_row_response = (
            supabase.table("source_zips")
            .insert(
                {
                    "source_id": source_id,
                    "zip_name": zip_file.filename,  # store only name
                }
            )
            .execute()
        )

        if not zip_row_response.data:
            raise HTTPException(
                status_code=500, detail="Failed to create source_zips entry."
            )
        zip_id = zip_row_response.data[0]["id"]

        # Extract files and upload to Supabase storage
        extracted_files = []
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
            for filename in zip_ref.namelist():
                if filename.endswith("/"):  # skip folders
                    continue

                file_path = os.path.join(tmp_dir, filename)
                if not os.path.isfile(file_path):
                    continue

                with open(file_path, "rb") as f:
                    storage_path = f"{zip_id}/{filename.strip()}"
                    supabase.storage.from_("images").upload(file=f, path=storage_path)
                    public_url = supabase.storage.from_("images").get_public_url(
                        storage_path
                    )

                    # Insert into zip_images table
                    supabase.table("zip_images").insert(
                        {
                            "filename": filename,
                            "zip_id": zip_id,
                            "storage_url": public_url,
                        }
                    ).execute()

                    extracted_files.append({"filename": filename, "url": public_url})

        return {
            "zip_name": zip_file.filename,
            "source_id": source_id,
            "zip_id": zip_id,
            "extracted_files": extracted_files,
        }

    except (APIError, Exception) as e:
        # Rollback if something fails
        if source_id:
            supabase.table("sources").delete().eq("id", source_id).execute()
        detail = f"An error occurred: {getattr(e, 'message', str(e))}"
        raise HTTPException(status_code=500, detail=detail)

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
