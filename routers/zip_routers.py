# from fastapi import APIRouter, UploadFile, File, HTTPException
# import zipfile
# import tempfile
# import os
# import shutil
# from typing import List, Dict, Any
# from core.image_processor import process_image
# from core.leaf_extraction import run_inference_from_pil
# from core.color_analysis import leaf_vein_skeleton, leaf_boundary_dilation, extract_colors_around_mask, cluster_and_mark_palette
# from core.etl_predictor import predict_etl_days
# from PIL import Image
# import cv2
# import numpy as np
# import base64
# import io
# from datetime import datetime

# router = APIRouter()

# BASE_OUTPUT_DIR = "batch_outputs"
# os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# def encode_image_to_base64_cv2(image: np.ndarray) -> str:
#     _, buffer = cv2.imencode('.jpg', image)
#     return base64.b64encode(buffer).decode()

# def encode_pil_to_base64(pil_img: Image.Image) -> str:
#     buffered = io.BytesIO()
#     pil_img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

# def get_timestamped_dir(name: str) -> str:
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path = os.path.join(BASE_OUTPUT_DIR, name, timestamp)
#     os.makedirs(path, exist_ok=True)
#     return path

# @router.post("/process-zip/pest-detection")
# async def batch_pest_detection(zip_file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
#         tmp.write(await zip_file.read())
#         tmp_path = tmp.name

#     extract_dir = tempfile.mkdtemp()
#     output_dir = get_timestamped_dir("pest_detection")

#     try:
#         with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_dir)

#         image_extensions = {".png", ".jpg", ".jpeg"," bmp", ".tiff"}
#         images = [
#             os.path.join(root, f) 
#             for root, _, files in os.walk(extract_dir)
#             for f in files if os.path.splitext(f)[1].lower() in image_extensions
#         ]

#         if not images:
#             raise HTTPException(status_code=400, detail="No images found in ZIP")

#         results = []
#         for img_path in images:
#             annotated_img, pest_counts, error = process_image(img_path)
#             if error:
#                 continue

#             image_name = os.path.basename(img_path)
#             output_img_path = os.path.join(output_dir, f"annotated_{image_name}")
#             cv2.imwrite(output_img_path, annotated_img)

#             results.append({
#                 "image_name": image_name,
#                 "pest_counts": pest_counts,
#                 "annotated_image_base64": encode_image_to_base64_cv2(annotated_img)
#             })

#         return {"total_images": len(images), "results": results}

#     finally:
#         os.unlink(tmp_path)
#         shutil.rmtree(extract_dir, ignore_errors=True)


# @router.post("/process-zip/leaf-extraction")
# async def batch_leaf_extraction(zip_file: UploadFile = File(...), conf_thresh: float = 0.5):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
#         tmp.write(await zip_file.read())
#         tmp_path = tmp.name

#     extract_dir = tempfile.mkdtemp()
#     output_dir = get_timestamped_dir("leaf_extraction")
#     # output_dir = os.path.join("leaf_extraction")
#     # os.makedirs(output_dir, exist_ok=True)


#     try:
#         with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_dir)

#         image_extensions = {".png", ".jpg", ".jpeg"}
#         images = [
#             os.path.join(root, f)
#             for root, _, files in os.walk(extract_dir)
#             for f in files if os.path.splitext(f)[1].lower() in image_extensions
#         ]

#         if not images:
#             raise HTTPException(status_code=400, detail="No images found in ZIP")

#         leaf_results = []
#         for img_path in images:
#             pil_img = Image.open(img_path).convert("RGBA")
#             extracted_paths, err = run_inference_from_pil(pil_img, conf_thresh=conf_thresh, output_dir=output_dir)

#             base64_leaves = []
#             for leaf_path in extracted_paths:
#                 leaf_img = Image.open(leaf_path)
#                 base64_leaves.append(encode_pil_to_base64(leaf_img))

#             leaf_results.append({
#                 "image_name": os.path.basename(img_path),
#                 "error": err,
#                 "extracted_leaf_count": len(extracted_paths),
#                 "extracted_leaf_images_base64": base64_leaves
#             })

#         return {"total_images": len(images), "leaf_extraction_results": leaf_results}

#     finally:
#         os.unlink(tmp_path)
#         shutil.rmtree(extract_dir, ignore_errors=True)


# @router.post("/process-zip/color-analysis")
# async def batch_color_analysis(zip_file: UploadFile = File(...)):
#     from datetime import datetime

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
#         content = await zip_file.read()
#         tmp.write(content)
#         tmp_path = tmp.name

#     extract_dir = tempfile.mkdtemp(prefix="zip_extract_")
#     output_root = "batch_outputs/color_analysis"
#     timestamped_dir = os.path.join(output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(timestamped_dir, exist_ok=True)

#     try:
#         with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_dir)

#         image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
#         images = []
#         for root, _, files in os.walk(extract_dir):
#             for file in files:
#                 ext = os.path.splitext(file)[1].lower()
#                 if ext in image_extensions:
#                     images.append(os.path.join(root, file))

#         if not images:
#             raise HTTPException(status_code=400, detail="No images found in ZIP")

#         color_results = []
#         for img_path in images:
#             image_name = os.path.basename(img_path)

#             try:
#                 vein_mask = leaf_vein_skeleton(img_path)
#                 boundary_mask = leaf_boundary_dilation(img_path)

#                 if vein_mask is None or boundary_mask is None:
#                     raise Exception("Empty mask")

#                 _, labels_vein, percs_vein, _ = extract_colors_around_mask(img_path, vein_mask)
#                 _, labels_boundary, percs_boundary, _ = extract_colors_around_mask(img_path, boundary_mask)

#                 color_results.append({
#                     "image_name": image_name,
#                     "vein_colors": [{"label": l, "percentage": round(p, 2)} for l, p in zip(labels_vein, percs_vein)],
#                     "boundary_colors": [{"label": l, "percentage": round(p, 2)} for l, p in zip(labels_boundary, percs_boundary)]
#                 })

#             except Exception as e:
#                 color_results.append({
#                     "image_name": image_name,
#                     "error": f"Failed to process: {str(e)}"
#                 })

#         return {
#             "total_images": len(images),
#             "color_analysis_results": color_results
#         }

#     finally:
#         os.unlink(tmp_path)
#         shutil.rmtree(extract_dir, ignore_errors=True)



# # If you don't already have this helper:
# def encode_image_to_base64_cv2_path(path: str):
#     img = cv2.imread(path)
#     if img is None:
#         return None
#     ok, buf = cv2.imencode(".png", img)
#     if not ok:
#         return None
#     return base64.b64encode(buf.tobytes()).decode("utf-8")

# @router.post("/process-zip/colour_palette")
# async def batch_color(uploaded_file: UploadFile = File(...)) -> Dict[str, Any]:
#     tmp_path = None
#     extract_dir = None
#     try:
#         # --- Save uploaded ZIP temporarily ---
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
#             tmp.write(await uploaded_file.read())
#             tmp_path = tmp.name

#         # --- Validate ZIP ---
#         if not zipfile.is_zipfile(tmp_path):
#             raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
#         with zipfile.ZipFile(tmp_path, "r") as zf:
#             bad = zf.testzip()
#             if bad:
#                 raise HTTPException(status_code=400, detail=f"ZIP is corrupted at entry: {bad}")

#         # --- Prep dirs ---
#         extract_dir = tempfile.mkdtemp(prefix="zip_extract_")
#         output_dir = os.path.join("batch_outputs", "color_palette")
#         os.makedirs(output_dir, exist_ok=True)

#         # --- Extract ---
#         with zipfile.ZipFile(tmp_path, "r") as zf:
#             zf.extractall(extract_dir)

#         # --- Find images recursively ---
#         image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
#         images = [
#             os.path.join(root, f)
#             for root, _, files in os.walk(extract_dir)
#             for f in files if f.lower().endswith(image_exts)
#         ]
#         if not images:
#             raise HTTPException(status_code=400, detail="No images found in ZIP.")

#         results = []
#         errors = []

#         for img_path in images:
#             image_name = os.path.basename(img_path)
#             try:
#                 # Quick read check
#                 if cv2.imread(img_path) is None:
#                     raise RuntimeError("cv2 could not read the image")

#                 # --- Generate masks ---
#                 vein_mask = leaf_vein_skeleton(image_path=img_path)
#                 boundary_mask = leaf_boundary_dilation(image_path=img_path)

#                 if vein_mask is None or boundary_mask is None:
#                     raise RuntimeError("mask generation failed")
#                 if cv2.countNonZero(vein_mask) == 0 or cv2.countNonZero(boundary_mask) == 0:
#                     raise RuntimeError("empty mask")

#                 # --- Extract colors (UNPACK the 4-tuple) ---
#                 stats_veins, _, _, _ = extract_colors_around_mask(
#                     img_path, vein_mask, buffer_ratio=0.5, num_colors=8, color_type="vein"
#                 )
#                 stats_boundary, _, _, _ = extract_colors_around_mask(
#                     img_path, boundary_mask, buffer_ratio=0.1, num_colors=8, color_type="boundary"
#                 )

#                 if not stats_veins and not stats_boundary:
#                     raise RuntimeError("no colors extracted")

#                 # Use actual RGB tuples from stats (NOT the labels)
#                 vein_colors = [tuple(map(int, v["rgb"])) for v in (stats_veins or {}).values()]
#                 boundary_colors = [tuple(map(int, v["rgb"])) for v in (stats_boundary or {}).values()]
#                 if not vein_colors and not boundary_colors:
#                     raise RuntimeError("parsed color lists are empty")

#                 # --- Save palette image ---
#                 palette_path = os.path.join(
#                     output_dir, f"{os.path.splitext(image_name)[0]}_color_palette.png"
#                 )
#                 saved_path = cluster_and_mark_palette(
#                     vein_colors=vein_colors,
#                     boundary_colors=boundary_colors,
#                     num_clusters=8,
#                     output_path=palette_path,
#                     # zip_path optional; set to None to skip zipping
#                     zip_path=None
#                 ) or palette_path

#                 if not os.path.exists(saved_path):
#                     raise RuntimeError(f"palette not found at {saved_path}")

#                 b64 = encode_image_to_base64_cv2_path(saved_path)
#                 if b64 is None:
#                     raise RuntimeError("failed to encode palette to base64")

#                 results.append({
#                     "image_name": image_name,
#                     "palette_image_base64": b64
#                 })

#             except Exception as e:
#                 # Don’t fail the whole batch if one image trips
#                 errors.append({"image_name": image_name, "error": f"{type(e).__name__}: {e}"})
#                 continue

#         if not results:
#             # Return the collected errors to help you debug from the client side
#             raise HTTPException(status_code=400, detail={"message": "No valid results", "errors": errors})

#         return {"results": results, "errors": errors}

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             try: os.unlink(tmp_path)
#             except: pass
#         if extract_dir and os.path.exists(extract_dir):
#             shutil.rmtree(extract_dir, ignore_errors=True)


# @router.post("/process-zip/etl-prediction")
# async def batch_etl_prediction(pest_data: List[Dict]):
#     """
#     ETL Prediction endpoint
    
#     Expected input format:
#     [
#         {
#             "pest_name": "Name of the pest",
#             "N_current": "Current pest population count",
#             "I": "Damage index",
#             "pesticides_cost": "Cost of pesticide treatment",
#             "market_cost_per_kg": "Market price",
#             "fev_con": "Environmental factor"
#         }
#     ]
#     """
#     rows = []
#     for data in pest_data:
#         rows.append([
#             data.get("pest_name", ""),
#             data.get("N_current", 1),
#             data.get("I", 0.0),
#             0,
#             data.get("pesticides_cost", 0.0),
#             data.get("market_cost_per_kg", 0.0),
#             0, 0,
#             data.get("fev_con", 0.0)
#         ])

#     df_etl, df_progress = predict_etl_days(rows)
#     return {
#         "etl_predictions": df_etl.to_dict(orient="records"),
#         "etl_progression": df_progress.to_dict(orient="records")
#     }
