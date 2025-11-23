from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
import os
import json
import requests
import uvicorn

# Load environment variables first
load_dotenv()

# Define the standalone app
app = FastAPI(
    title="HR Strategic Restructuring Agent",
    version="3.0.0",
    description="Calculates weighted scores and generates strategic reports using 3-shot learning."
)

# Add CORS middleware (Required for standalone testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DEBUGGING HANDLER ---
# This prints the specific validation error to your console when you get a 422
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = exc.errors()
    print(f"\n❌ VALIDATION ERROR CAUGHT: {json.dumps(error_details, indent=2)}\n")
    return JSONResponse(
        status_code=422,
        content={"detail": error_details, "body": exc.body},
    )

# --- CONFIGURATION ---
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_URL = os.environ.get("WATSONX_URL")
MODEL_ID = os.environ.get("MODEL_ID")

# --- DATA MODELS ---

class EmployeeInput(BaseModel):
    # Using aliases allows the API to accept your CSV/JSON headers directly
    employee_id: Optional[Union[str, int]] = Field(None, alias="employee_id")
    name: str = Field(..., alias="name")
    rating: float = Field(..., description="Performance Rating 0.0-5.0")
    talent: Union[str, int] = Field(..., description="Talent Score (High/Medium/Low or 1-5)")
    studies: str = Field(..., description="Education Level")
    salary: float = Field(..., description="Current Salary")
    job: str = Field(..., description="Job Title")

    class Config:
        populate_by_name = True

class RestructuringRequest(BaseModel):
    employees: List[EmployeeInput]

class RestructuringResponse(BaseModel):
    status: str
    statistics: Dict[str, Any]
    strategic_report: str
    raw_algorithm_output: Dict[str, Any]

# --- WATSONX HELPERS ---

def get_watsonx_token():
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY}
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=10)
        if response.status_code != 200:
            print(f"Auth Error: {response.text}")
            return None
        return response.json()["access_token"]
    except Exception as e:
        print(f"Auth Connection Error: {e}")
        return None

def call_watsonx(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    try:
        token = get_watsonx_token()
        if not token:
            return "Error: Could not authenticate with IBM Cloud."

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
                "min_new_tokens": 50,
                "repetition_penalty": 1.1
            },
            "project_id": WATSONX_PROJECT_ID
        }
        
        response = requests.post(WATSONX_URL, headers=headers, json=body, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            return result["results"][0]["generated_text"].strip()
        else:
            print(f"WatsonX API Error ({response.status_code}): {response.text}")
            return f"Error from AI Provider: {response.text}"
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

# --- ALGORITHMIC LOGIC ---

def execute_scoring_algorithm(employees: List[EmployeeInput]) -> Dict[str, Any]:
    # 1. Weights
    WEIGHT_PERFORMANCE = 0.5
    WEIGHT_TALENT = 0.3
    WEIGHT_EDUCATION = 0.2

    # 2. Mappings
    education_map = {
        "PhD": 5, "Doctorate": 5, "MD": 5, "JD": 5,
        "Master": 4, "Masters": 4, "MSc": 4,
        "Bachelor": 3, "Bachelors": 3, "BSc": 3, "BSN": 3,
        "Associate": 2, "High School": 1
    }
    talent_map = {
        "High": 5, "Medium": 3, "Low": 1,
        "5": 5, "4": 4, "3": 3, "2": 2, "1": 1
    }

    processed_staff = []

    # 3. Calculation Loop
    for emp in employees:
        try:
            # Map Text to Score
            talent_str = str(emp.talent).capitalize()
            if talent_str in talent_map:
                talent_score = talent_map[talent_str]
            elif talent_str in ["1","2","3","4","5"]:
                talent_score = int(talent_str)
            else:
                talent_score = 1 # Default

            edu_score = education_map.get(emp.studies, 1)

            # Weighted Formula
            final_score = (emp.rating * WEIGHT_PERFORMANCE) + \
                          (talent_score * WEIGHT_TALENT) + \
                          (edu_score * WEIGHT_EDUCATION)

            processed_staff.append({
                "name": emp.name,
                "role": emp.job,
                "salary": emp.salary,
                "final_score": round(final_score, 2),
                "raw_rating": emp.rating
            })
        except Exception as e:
            print(f"Skipping employee due to error: {e}")
            continue

    # 4. Ranking & Allocation
    processed_staff.sort(key=lambda x: x['final_score'], reverse=True)
    
    HQ_CAPACITY = 20
    hq_team = processed_staff[:HQ_CAPACITY]
    remote_team = processed_staff[HQ_CAPACITY:]
    
    # 5. Risk Analysis
    total_staff = len(processed_staff)
    avg_score = sum(p['final_score'] for p in processed_staff) / total_staff if total_staff > 0 else 0
        
    flagged_risks = []
    for p in processed_staff:
        # Risk Condition: Score < Avg AND Salary > 100k
        if p['final_score'] < avg_score and p['salary'] > 100000:
            flagged_risks.append({
                "name": p['name'],
                "role": p['role'],
                "salary": p['salary'],
                "merit_score": p['final_score'],
                "discrepancy": "High Salary / Low Merit"
            })

    return {
        "status": "success",
        "statistics": {
            "total_employees": total_staff,
            "average_merit_score": round(avg_score, 2),
            "hq_count": len(hq_team),
            "remote_count": len(remote_team)
        },
        "top_talent_hq": [
            {"name": p['name'], "score": p['final_score']} for p in hq_team[:3]
        ],
        "allocation_summary": {
            "headquarters_roster": [p['name'] for p in hq_team],
            "remote_roster": [p['name'] for p in remote_team]
        },
        "risk_flags": flagged_risks
    }

# --- PROMPT GENERATOR ---

def generate_architect_prompt(current_data: Dict) -> str:
    few_shot_examples = """
Input:
{
  "status": "success",
  "statistics": { "total_employees": 5, "average_merit_score": 3.2, "hq_count": 3, "remote_count": 2 },
  "top_talent_hq": [ { "name": "Alice Chen", "score": 4.8 }, { "name": "Marcus Thorne", "score": 4.2 }, { "name": "Sarah Jenkins", "score": 3.9 } ],
  "allocation_summary": { "headquarters_roster": [ "Alice Chen", "Marcus Thorne", "Sarah Jenkins" ], "remote_roster": [ "David Lo", "Elena Rigby" ] },
  "risk_flags": [ { "name": "David Lo", "role": "Senior Associate", "salary": 145000.0, "merit_score": 1.8, "discrepancy": "High Salary / Low Merit" } ]
}

Output:
Strategic Restructuring Report

1. Executive Summary
Overview: We have successfully processed 5 employees for the new department. The group average merit score is 3.2.
Allocation: Based on merit rankings, 3 employees have been assigned to Headquarters, while 2 employees will work Remote/Secondary.

2. Leadership & Top Talent
The following high-performers will anchor the HQ team:
Alice Chen (Score: 4.8)
Marcus Thorne (Score: 4.2)
Sarah Jenkins (Score: 3.9)
Insight: These individuals demonstrate the highest alignment of performance, talent, and education.

3. Financial & Performance Risk Audit (High Priority)
We have identified 1 critical inefficiency requiring immediate attention:
David Lo (Senior Associate)
The Discrepancy: Mr. Lo draws a senior-level salary ($145k) but holds a merit score of 1.8, which is significantly below the department average of 3.2.
Strategic Recommendation: Initiate a formal Performance Improvement Plan (PIP) immediately. If metrics do not improve within 60 days, assess for redundancy to recover budget.

4. Final Roster Visualization
Headquarters Team:
Alice Chen
Marcus Thorne
Sarah Jenkins
"""
    current_json = json.dumps(current_data)
    
    return f"""
You are the Chief Organizational Architect for a prestigious legal firm.
Generate a Strategic Restructuring Report based on the input JSON.
Follow the format of the examples below EXACTLY.

{few_shot_examples}

---

Input:
{current_json}

Output:
"""

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "message": "HR Strategic Restructuring Agent API",
        "version": "3.0.0",
        "endpoints": {
            "/generate-strategy": "POST - Generate restructuring report"
        }
    }

@app.post("/generate-strategy", response_model=RestructuringResponse)
async def generate_strategy(request: RestructuringRequest):
    try:
        if not request.employees:
            raise HTTPException(status_code=400, detail="No employee data provided")

        # 1. Run the Python Node (The Math)
        algo_results = execute_scoring_algorithm(request.employees)

        # 2. Build the Prompt
        prompt = generate_architect_prompt(algo_results)

        # 3. Call WatsonX
        ai_report = call_watsonx(prompt)

        if not ai_report:
            ai_report = "Error: AI generation failed. Please check API Key."

        return {
            "status": "success",
            "statistics": algo_results["statistics"],
            "strategic_report": ai_report,
            "raw_algorithm_output": algo_results
        }

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)