from fastapi import APIRouter
from supabase import create_client, Client
from config import supabase

router = APIRouter()

@router.post("/add-url/")
async def add_url(image_url: str):
    source = supabase.table("sources").insert({"kind": "url"}).execute()
    source_id = source.data[0]["id"]

    supabase.table("source_urls").insert({
        "source_id": source_id,
        "url": image_url
    }).execute()

    return {"source_id": source_id, "url": image_url}