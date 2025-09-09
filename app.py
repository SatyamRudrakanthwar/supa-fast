from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import image_routes, leaf_routes, color_routes, etl_routes, zip_routers, singleimage_router, roboflow, urllist_router,zipextraction_router


app = FastAPI(title="AgriSavant Pest ETL API")

# Allow CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(singleimage_router.router, prefix="/singleimage", tags=["Single Image Upload"])
app.include_router(urllist_router.router, prefix="/urllist", tags=["URL List Upload"])
app.include_router(zipextraction_router.router, prefix="/zip", tags=["ZIP Extraction"])
app.include_router(image_routes.router, prefix="/image", tags=["Image"])
app.include_router(leaf_routes.router, prefix="/leaf", tags=["Leaf Extraction"])
app.include_router(color_routes.router, prefix="/color", tags=["Color Analysis"])
app.include_router(etl_routes.router, prefix="/etl", tags=["ETL Prediction"])
# app.include_router(zip_routers.router, prefix="/zip", tags=["ZIP Batch Processing"])
app.include_router(roboflow.router, prefix="/roboflow", tags=["Roboflow Upload"])
@app.get("/")
def root():
    return {"message": "Welcome to AgriSavant Pest ETL API"}
