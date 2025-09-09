from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile, os, uuid, cv2, io, matplotlib.pyplot as plt
from supabase import create_client, Client
from core.color_analysis import (
    leaf_vein_skeleton,
    leaf_boundary_dilation,
    extract_colors_around_mask,
    cluster_and_mark_palette,
    extract_leaf_colors_with_locations
)
import numpy as np

router = APIRouter()

# --- Supabase client ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _upload_to_supabase(bucket: str, file_path: str, prefix: str) -> str:
    """Upload local file to Supabase bucket and return public URL"""
    filename = f"{prefix}/{uuid.uuid4()}{os.path.splitext(file_path)[1]}"
    with open(file_path, "rb") as f:
        supabase.storage.from_(bucket).upload(filename, f, {"upsert": True})
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"


@router.post("/color-analysis")
async def run_color_analysis(
    image: UploadFile = File(...),
    source_id: str = None,
    num_clusters: int = 6
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(await image.read())
    tmp.close()
    image_path = tmp.name
    image_name = image.filename

    try:
        # --- 1. Vein + Boundary mask ---
        vein_mask = leaf_vein_skeleton(image_path)
        boundary_mask = leaf_boundary_dilation(image_path)

        vein_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        boundary_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(vein_path, vein_mask)
        cv2.imwrite(boundary_path, boundary_mask)

        vein_mask_url = _upload_to_supabase("analysis-results", vein_path, "vein_masks")
        boundary_mask_url = _upload_to_supabase("analysis-results", boundary_path, "boundary_masks")

        # --- 2. Extract colors ---
        stats_veins, vein_labels, vein_perc, vein_colors = extract_colors_around_mask(
            image_path, vein_mask, buffer_ratio=0.5, num_colors=num_clusters, color_type="vein"
        )
        stats_boundary, boundary_labels, boundary_perc, boundary_colors = extract_colors_around_mask(
            image_path, boundary_mask, buffer_ratio=0.15, num_colors=num_clusters, color_type="boundary"
        )

        vein_colors_list = [v["rgb"] for v in stats_veins.values()]
        boundary_colors_list = [v["rgb"] for v in stats_boundary.values()]

        # --- 3. Palette image ---
        palette_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cluster_and_mark_palette(
            vein_colors_list, boundary_colors_list, num_clusters=num_clusters, output_path=palette_path
        )
        palette_url = _upload_to_supabase("analysis-results", palette_path, "color_palettes")

        # --- 4. Bar plot for extracted colors ---
        fig_main, region_figs = extract_leaf_colors_with_locations(
            image_path, num_colors=num_clusters, save_dir=None
        )

        # Save main bar chart
        bar_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig_main.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig_main)
        bar_url = _upload_to_supabase("analysis-results", bar_path, "color_bars")

        # Save region overlays
        region_urls = []
        for i, fig in enumerate(region_figs):
            region_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_region{i+1}.png").name
            fig.savefig(region_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            url = _upload_to_supabase("analysis-results", region_path, "color_regions")
            region_urls.append(url)

        # --- 5. Insert into DB ---
        meta = {
            "vein_mask_pixels": int(cv2.countNonZero(vein_mask)),
            "boundary_mask_pixels": int(cv2.countNonZero(boundary_mask)),
        }

        result = supabase.table("color_results").insert({
            "source_id": source_id,
            "image_name": image_name,
            "meta": meta,
            "vein": list(stats_veins.values()),  
            "boundary": list(stats_boundary.values()),  
            "vein_mask_url": vein_mask_url,
            "boundary_mask_url": boundary_mask_url,
            "palette_url": palette_url,
            "bar_chart_url": bar_url,
            "region_urls": region_urls
        }).execute()

        return {
            "message": "Color analysis completed and stored",
            "image_name": image_name,
            "vein": stats_veins,
            "boundary": stats_boundary,
            "urls": {
                "vein_mask": vein_mask_url,
                "boundary_mask": boundary_mask_url,
                "palette": palette_url,
                "bar_chart": bar_url,
                "regions": region_urls
            },
            "db_response": result.data
        }

    finally:
        os.unlink(image_path)
