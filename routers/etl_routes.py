from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from core.etl_predictor import calculate_value_loss, predict_etl_days
from supabase import create_client, Client
import os

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter()

# ---- Request Models ----
class PestDataItem(BaseModel):
    pest_name: str
    N_current: float
    I: float
    pesticides_cost: float
    market_cost_per_kg: float
    rh: float

class PestDataRequest(BaseModel):
    data: List[PestDataItem]

# ---- Route ----
@router.post("/etl-prediction")
def pest_analysis(request: PestDataRequest, source_id: str = Query(..., description="Source ID from sources table")):
    """
    Run ETL (Economic Threshold Level) prediction and store results in Supabase.
    """
    try:
        combined_results = []
        data_rows = []

        # Step 1: Loop through each pest entry
        for item in request.data:
            # ---- Yield + Value loss calculation ----
            yield_lost, value_loss = calculate_value_loss(item.I, item.market_cost_per_kg)
            result = value_loss / item.pesticides_cost if item.pesticides_cost != 0 else 0

            combined_results.append({
                "pest_name": item.pest_name,
                "yield_loss": yield_lost,
                "value_loss": value_loss,
                "result": result
            })

            # ---- Prepare row for ETL prediction ----
            row = [
                item.pest_name,
                item.N_current,
                item.I,
                yield_lost,
                item.pesticides_cost,
                item.market_cost_per_kg,
                value_loss,
                result,
                item.rh
            ]
            data_rows.append(row)

        # Step 2: Run ETL prediction
        df_etl, df_progress = predict_etl_days(data_rows)

        etl_days = df_etl.to_dict(orient="records")
        progress_data = df_progress.to_dict(orient="records")

        # Step 3: Store in etl_results
        res = supabase.table("etl_results").insert({
            "source_id": source_id,
            "per_pest_results": combined_results,
            "etl_days": etl_days,
            "progress_data": progress_data
        }).execute()

        if not res.data:
            raise HTTPException(status_code=500, detail="Failed to insert ETL results")

        result_id = res.data[0]["id"]

        # Step 4: Log run in analysis_log
        supabase.table("analysis_log").insert({
            "source_id": source_id,
            "module": "etl",
            "result_id": result_id
        }).execute()

        return {
            "per_pest_results": combined_results,
            "etl_days": etl_days,
            "progress_data": progress_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETL Prediction failed: {str(e)}")
