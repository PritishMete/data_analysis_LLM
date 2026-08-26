from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import json

# Assuming AgenticBacktracker is imported
# from cleaning_agent import AgenticBacktracker

router = APIRouter()

class BacktrackRequest(BaseModel):
    data: list[dict] # The raw sheet data
    target_column: str

@router.post("/api/clean/dynamic_backtrack")
async def dynamic_backtrack(request: BacktrackRequest):
    try:
        # 1. Convert incoming JSON to Pandas DataFrame
        df = pd.DataFrame(request.data)
        
        # 2. Initialize your LLM client (replace with your actual initialization)
        llm_client = MyLLMClient() 
        agent = AgenticBacktracker(llm_client)
        
        # 3. Run the agentic backtrack
        cleaned_df, success, message = agent.apply_dynamic_backtrack(df, request.target_column)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
            
        # 4. Return the updated data to the Dart frontend
        # Replace NaN/NaT with None so it serializes properly to JSON
        cleaned_df = cleaned_df.where(pd.notnull(cleaned_df), None)
        
        return {
            "success": True,
            "message": message,
            "cleaned_data": cleaned_df.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
