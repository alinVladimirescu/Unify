from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import requests
from dotenv import load_dotenv
import os

app = FastAPI()

WATSONX_API_KEY = "apikeyplaceholder"
WATSONX_PROJECT_ID = "projectidplaceholder"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
MODEL_ID = "ibm/granite-3-8b-instruct"


class FeatureData(BaseModel):
    feature: str
    revenue: float
    cost: float
    net_profit: float


class PnLRequest(BaseModel):
    features: List[FeatureData]


class FeatureAdvice(BaseModel):
    feature_name: str
    status: str  
    financial_metric: str 
    reason: str


class PnLAnalysisResponse(BaseModel):
    executive_summary: str
    features_to_keep: List[FeatureAdvice]
    features_to_ditch: List[FeatureAdvice]
    raw_ai_response: Optional[str] = None


def get_watsonx_token():
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY}
    response = requests.post(token_url, headers=headers, data=data, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to get token: {response.text}")
    return response.json()["access_token"]


def call_watsonx(prompt: str, max_tokens: int = 1024) -> str:
    try:
        token = get_watsonx_token()
        headers = {
            "Accept": "application/json", 
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {token}"
        }
        body = {
            "model_id": MODEL_ID,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy", 
                "max_new_tokens": max_tokens, 
                "temperature": 0.2,
                "repetition_penalty": 1.1
            },
            "project_id": WATSONX_PROJECT_ID
        }
        
        response = requests.post(WATSONX_URL, headers=headers, json=body, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            return result["results"][0]["generated_text"].strip()
        else:
            return None
    except Exception as e:
        return None


def generate_financial_prompt(features: List[FeatureData]) -> str:
    feature_list = "\n".join([
        f"- {f.feature}: Revenue ${f.revenue:,.0f}, Cost ${f.cost:,.0f}, Net Profit ${f.net_profit:,.0f}"
        for f in features
    ])
    
    return f"""You are a ruthless business strategist and financial analyst.
Analyze the following Profit & Loss (P&L) data for our product features.

Features:
{feature_list}

Your Goal: Determine which features are profitable (Keep) and which are money pits (Ditch).

Task:
1. Identify features that should be KEPT (High profit, strategic growth).
2. Identify features that should be DITCHED (High cost, low revenue, negative margin).
3. Provide a strictly formatted JSON response.

Output JSON Structure:
{{
    "summary": "A 2-sentence executive summary of the overall financial health.",
    "analysis": [
        {{
            "name": "Feature Name",
            "action": "Keep" or "Ditch",
            "metric": "Net Profit: $50k" or "Loss: -$10k",
            "reason": "Brief financial justification."
        }}
    ]
}}

Return ONLY the JSON. No markdown formatting.
Your JSON Response:"""


@app.post("/analyze-pnl", response_model=PnLAnalysisResponse)
async def analyze_pnl(request: PnLRequest):
    try:
        if not request.features or len(request.features) == 0:
            raise HTTPException(status_code=400, detail="No features provided")
        
        
        prompt = generate_financial_prompt(request.features)
        ai_response = call_watsonx(prompt)
        
        if not ai_response:
            raise HTTPException(status_code=502, detail="Watsonx AI failed to analyze data")

        try:
            clean_json = ai_response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            keep_list = []
            ditch_list = []
            
            for item in data.get("analysis", []):
                advice = FeatureAdvice(
                    feature_name=item.get("name", "Unknown"),
                    status=item.get("action", "Review"),
                    financial_metric=item.get("metric", "N/A"),
                    reason=item.get("reason", "No reason provided")
                )
                
                if advice.status.lower() == "keep":
                    keep_list.append(advice)
                else:
                    ditch_list.append(advice)
            
            return {
                "executive_summary": data.get("summary", "Analysis complete."),
                "features_to_keep": keep_list,
                "features_to_ditch": ditch_list,
                "raw_ai_response": clean_json
            }
            
        except json.JSONDecodeError:
            return {
                "executive_summary": "AI generated a response but it wasn't strict JSON. See raw response.",
                "features_to_keep": [],
                "features_to_ditch": [],
                "raw_ai_response": ai_response
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "P&L Strategic Advisor Agent",
        "status": "active",
        "version": "5.0.0 - Pure JSON",
        "endpoint": "/analyze-pnl"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)