from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import json
import requests
import uvicorn
import re

app = FastAPI()

architect = FastAPI(
    title="HR Strategic Restructuring Agent",
    version="3.0.0",
    description="Calculates weighted scores and generates strategic reports using 3-shot learning."
)
load_dotenv()

# Add CORS middleware to handle cross-origin requests
architect.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# IMPORTANT: Replace these with your actual IBM Cloud Credentials

WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_URL = os.environ.get("WATSONX_URL")
MODEL_ID = os.environ.get("MODEL_ID")

# --- DATA MODELS ---

class EmployeeInput(BaseModel):
    # Using aliases allows the API to accept your CSV headers directly
    # Allow both string and integer for Employee ID
    employee_id: Optional[str | int] = Field(None, alias="Employee ID")
    name: str = Field(..., alias="Name")
    rating: float = Field(..., description="Performance Rating 0.0-5.0")
    # Allow both string and integer for talent
    talent: str | int = Field(..., description="Talent Score (High/Medium/Low or 1-5)")
    studies: str = Field(..., description="Education Level")
    # This alias maps the JSON key "salary" to the Python variable 'salary'
    salary: float = Field(..., alias="salary", description="Current Salary")
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
    response = requests.post(token_url, headers=headers, data=data, timeout=10)
    if response.status_code != 200:
        print(f"Auth Error: {response.text}")
        raise Exception("Failed to authenticate with IBM Cloud")
    return response.json()["access_token"]

def call_watsonx(prompt: str, max_tokens: int = 1500) -> Optional[str]:
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
            return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

# --- ALGORITHMIC LOGIC (THE PYTHON NODE) ---

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
            # Handle numeric input vs text input
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
        except Exception:
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

    # 6. Construct Data Payload
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

# --- PROMPT GENERATOR WITH 3-SHOT EXAMPLES ---

def generate_architect_prompt(current_data: Dict) -> str:
    # We define the examples as a constant string block
    
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

---

Input:
{
  "status": "success",
  "statistics": { "total_employees": 4, "average_merit_score": 2.5, "hq_count": 2, "remote_count": 2 },
  "top_talent_hq": [ { "name": "Priya Patel", "score": 4.5 }, { "name": "John Smith", "score": 3.1 } ],
  "allocation_summary": { "headquarters_roster": [ "Priya Patel", "John Smith" ], "remote_roster": [ "Gary Oldman", "Linda Free" ] },
  "risk_flags": [ 
    { "name": "Gary Oldman", "role": "Partner Track", "salary": 200000.0, "merit_score": 1.5, "discrepancy": "High Salary / Low Merit" },
    { "name": "Linda Free", "role": "Special Counsel", "salary": 180000.0, "merit_score": 1.2, "discrepancy": "High Salary / Low Merit" }
  ]
}

Output:
Strategic Restructuring Report

1. Executive Summary
Overview: The team of 4 is underperforming, with a low average merit score of 2.5.
Allocation: 2 employees qualified for Headquarters; 2 employees are assigned Remote status.

2. Leadership & Top Talent
Priya Patel (Score: 4.5)
John Smith (Score: 3.1)
Insight: Priya is the sole high-performer. John Smith made the HQ cut largely due to lack of competition.

3. Financial & Performance Risk Audit (High Priority)
This department shows severe cost-to-value misalignment.
Gary Oldman (Partner Track)
The Discrepancy: A $200k salary is unjustifiable for a score of 1.5. This represents a major sunk cost.
Strategic Recommendation: Role De-scoping & Salary Review. His current output does not match the Partner Track criteria.
Linda Free (Special Counsel)
The Discrepancy: Earning $180k with the lowest score in the group (1.2).
Strategic Recommendation: Assess for Redundancy. The gap between cost and value is too wide to bridge via training.

4. Final Roster Visualization
Headquarters Team:
Priya Patel
John Smith

---

Input:
{
  "status": "success",
  "statistics": { "total_employees": 3, "average_merit_score": 4.8, "hq_count": 2, "remote_count": 1 },
  "top_talent_hq": [ { "name": "Diana Prince", "score": 5.0 }, { "name": "Clark Kent", "score": 4.9 } ],
  "allocation_summary": { "headquarters_roster": [ "Diana Prince", "Clark Kent" ], "remote_roster": [ "Bruce Wayne" ] },
  "risk_flags": []
}

Output:
Strategic Restructuring Report

1. Executive Summary
Overview: This is an exceptional group. The total staff of 3 has an outstanding average merit score of 4.8.
Allocation: Due to strict capacity limits, 2 employees are in Headquarters and 1 employee is Remote, despite high performance across the board.

2. Leadership & Top Talent
Diana Prince (Score: 5.0)
Clark Kent (Score: 4.9)
Insight: Both individuals achieved near-perfect scores and are critical assets to the firm.

3. Financial & Performance Risk Audit (High Priority)
No Risks Identified.
Observation: All salaries appear commensurate with the high level of merit delivered by this team. No cost-saving actions are required.

4. Final Roster Visualization
Headquarters Team:
Diana Prince
Clark Kent
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

@architect.get("/")
async def root():
    return {
        "message": "HR Strategic Restructuring Agent API",
        "version": "3.0.0",
        "endpoints": {
            "/generate-strategy": "POST - Generate restructuring report"
        }
    }

@architect.post("/generate-strategy", response_model=RestructuringResponse)
async def generate_strategy(request: RestructuringRequest):
    try:
        if not request.employees:
            raise HTTPException(status_code=400, detail="No employee data provided")

        # 1. Run the Python Node (The Math)
        algo_results = execute_scoring_algorithm(request.employees)

        # 2. Build the Prompt (The 3 Examples + Current Data)
        prompt = generate_architect_prompt(algo_results)

        # 3. Call WatsonX
        ai_report = call_watsonx(prompt)

        if not ai_report:
            # Fallback if AI fails, return the data at least
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

auditor = FastAPI()

SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY")
SERPAPI_URL = "https://serpapi.com/search"

class InvoiceItem(BaseModel):
    service_name: str
    description: Optional[str] = "Software Subscription"
    internal_cost: float 

class HarmonizationRequest(BaseModel):
    company_a_invoices: List[InvoiceItem]
    company_b_invoices: List[InvoiceItem]

def get_watsonx_token():
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY}
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            return response.json().get("access_token")
    except:
        pass
    return None

def call_watsonx(prompt: str, max_tokens: int = 500) -> str:
    token = get_watsonx_token()
    if not token: 
        return None
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    body = {
        "model_id": MODEL_ID,
        "input": prompt,
        "parameters": {"decoding_method": "greedy", "max_new_tokens": max_tokens, "repetition_penalty": 1.1},
        "project_id": WATSONX_PROJECT_ID
    }
    try:
        res = requests.post(WATSONX_URL, headers=headers, json=body, timeout=30)
        if res.status_code == 200:
            return res.json()["results"][0]["generated_text"].strip()
    except:
        pass
    return None

def search_web_price(service_name: str) -> float:
    try:
        params = {'q': f"{service_name} pricing cost per user per month business plan", 'api_key': SEARCH_API_KEY, 'engine': 'google', 'num': 5}
        res = requests.get(SERPAPI_URL, params=params, timeout=15)
        
        if res.status_code != 200:
            return search_price_fallback(service_name)
        
        data = res.json()
        if 'error' in data or "organic_results" not in data or len(data.get("organic_results", [])) == 0:
            return search_price_fallback(service_name)
        
        snippets = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in data.get('organic_results', [])[:5]]
        combined_text = " ".join(snippets)
        
        price = extract_price_from_text(combined_text)
        if price > 0:
            return price
        
        price = extract_price_ai(service_name, combined_text)
        if price > 0:
            return price
        
        return search_price_fallback(service_name)
    except:
        return search_price_fallback(service_name)

def search_price_fallback(service_name: str) -> float:
    prompt = f"What is the typical monthly per-user pricing for {service_name} business plan in 2024? Return ONLY a number. Example: 15.00"
    res = call_watsonx(prompt, max_tokens=20)
    
    if res:
        res = res.strip().replace('$', '').replace(',', '')
        match = re.search(r'(\d+(?:\.\d{1,2})?)', res)
        if match:
            return float(match.group(1))
    
    pricing_db = {"slack": 12.50, "microsoft teams": 5.00, "zoom": 15.00, "jira": 8.15, "asana": 13.49, 
                  "monday": 12.00, "figma": 15.00, "notion": 10.00, "hubspot": 50.00, "salesforce": 75.00}
    
    name_lower = service_name.lower()
    for key, price in pricing_db.items():
        if key in name_lower:
            return price
    
    return 0.0

def extract_price_from_text(text: str) -> float:
    patterns = [r'\$(\d+(?:\.\d{2})?)\s*(?:per|/)\s*(?:user|seat)', r'\$(\d+(?:\.\d{2})?)\s*(?:per|/)\s*month']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0

def extract_price_ai(service_name: str, text: str) -> float:
    prompt = f"Extract monthly per-user pricing for {service_name} from: \"{text[:600]}\" Return ONLY a number."
    res = call_watsonx(prompt, max_tokens=20)
    if res:
        res = res.strip().replace('$', '').replace(',', '')
        match = re.search(r'(\d+(?:\.\d{1,2})?)', res)
        if match: 
            return float(match.group(1))
    return 0.0

def detect_redundancies_ai(invoices: List[InvoiceItem]) -> List[dict]:
    list_text = "\n".join([f"- {inv.service_name} ({inv.description})" for inv in invoices])
    prompt = f"Analyze these tools and find redundancies:\n{list_text}\n\nReturn JSON array: [{{\"category\": \"Chat\", \"tools\": [\"Slack\", \"Teams\"]}}]\nReturn ONLY JSON. If no conflicts, return []."
    
    res = call_watsonx(prompt, max_tokens=600)
    conflicts = []
    
    if res:
        try:
            clean = res.replace("```json", "").replace("```", "").strip()
            conflicts = json.loads(clean)
        except:
            pass
            
    if not conflicts:
        conflicts = fallback_redundancy_check(invoices)
        
    return conflicts

def fallback_redundancy_check(invoices: List[InvoiceItem]) -> List[dict]:
    groups = {
        "Communication": ["slack", "teams", "discord", "zoom", "meet"],
        "Project Management": ["jira", "asana", "monday", "trello", "linear"],
        "CRM": ["hubspot", "salesforce", "zoho"],
        "Design": ["figma", "adobe xd", "sketch"],
        "Cloud": ["aws", "azure", "google cloud"]
    }
    
    found_groups = {}
    for inv in invoices:
        name_lower = inv.service_name.lower()
        for category, keywords in groups.items():
            if any(k in name_lower for k in keywords):
                if category not in found_groups:
                    found_groups[category] = []
                found_groups[category].append(inv.service_name)
                break
        
    return [{"category": cat, "tools": tools} for cat, tools in found_groups.items() if len(tools) > 1]

def fuzzy_find_tool(target_name: str, tool_map: dict) -> str:
    if target_name in tool_map: 
        return target_name
    for name in tool_map.keys():
        if target_name.lower() == name.lower() or target_name.lower() in name.lower() or name.lower() in target_name.lower():
            return name
    return None

@auditor.post("/harmonize-tech-stack")
async def harmonize_tech_stack(request: HarmonizationRequest):
    all_invoices = request.company_a_invoices + request.company_b_invoices
    tool_map = {inv.service_name: inv for inv in all_invoices}
    
    conflicts = detect_redundancies_ai(all_invoices)
    recommendations = []
    processed_tools = set()

    for group in conflicts:
        category = group.get("category", "General")
        raw_tool_names = group.get("tools", [])
        
        valid_tools = [fuzzy_find_tool(name, tool_map) for name in raw_tool_names]
        valid_tools = list(set([t for t in valid_tools if t]))
        
        if len(valid_tools) > 1:
            for t in valid_tools: 
                processed_tools.add(t)
            
            name_a, name_b = valid_tools[0], valid_tools[1]
            item_a, item_b = tool_map[name_a], tool_map[name_b]
            
            price_a = search_web_price(name_a)
            price_b = search_web_price(name_b)
            
            final_a = price_a if price_a > 0 else item_a.internal_cost
            final_b = price_b if price_b > 0 else item_b.internal_cost
            
            if final_a < final_b:
                winner, loser = name_a, name_b
                savings = (final_b - final_a) * 100 * 12
            else:
                winner, loser = name_b, name_a
                savings = (final_a - final_b) * 100 * 12
            
            recommendations.append({
                "category": category,
                "action": "MIGRATE",
                "tool_to_keep": winner,
                "tool_to_drop": loser,
                "estimated_annual_savings": round(savings, 2),
                "reasoning": f"{category} overlap. {winner} (${min(final_a, final_b)}/mo) cheaper than {loser} (${max(final_a, final_b)}/mo)."
            })

    for name in tool_map.keys():
        if name not in processed_tools:
            recommendations.append({
                "category": "Unique",
                "action": "KEEP",
                "tool_to_keep": name,
                "tool_to_drop": "None",
                "estimated_annual_savings": 0.0,
                "reasoning": "No conflict found."
            })

    return {"status": "success", "conflicts_detected": len(conflicts), "recommendations": recommendations}

@auditor.get("/")
async def root():
    return {"service": "Tech Stack Harmonization Agent", "status": "active", "version": "3.0.0"}

analyst = FastAPI()

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


@analyst.post("/analyze-pnl", response_model=PnLAnalysisResponse)
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


@analyst.get("/")
async def root():
    return {
        "service": "P&L Strategic Advisor Agent",
        "status": "active",
        "version": "5.0.0 - Pure JSON",
        "endpoint": "/analyze-pnl"
    }

curator = FastAPI()

class Employee(BaseModel):
    employee_id: str
    employee_name: str
    seniority_level: str
    position: str

class ProcessEmployeesRequest(BaseModel):
    employees: List[Employee]

class SkillPredictionRequest(BaseModel):
    employee_name: Optional[str] = None
    position: str
    seniority_level: str

COURSE_CATALOG_FILE = os.path.join("Dev Extra", "course_catalog.json")

def load_course_catalog():
    try:
        with open(COURSE_CATALOG_FILE, 'r') as f:
            catalog = json.load(f)
            print(f" Loaded {len(catalog)} courses from {COURSE_CATALOG_FILE}")
            return catalog
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        return []

COURSE_CATALOG = load_course_catalog()

def get_watsonx_token():
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY}
    response = requests.post(token_url, headers=headers, data=data, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to get token: {response.text}")
    return response.json()["access_token"]

def call_watsonx(prompt: str, max_tokens: int = 512) -> str:
    try:
        token = get_watsonx_token()
        headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        body = {
            "model_id": MODEL_ID,
            "input": prompt,
            "parameters": {"decoding_method": "greedy", "max_new_tokens": max_tokens, "temperature": 0.3},
            "project_id": WATSONX_PROJECT_ID
        }
        
        response = requests.post(WATSONX_URL, headers=headers, json=body, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["results"][0]["generated_text"].strip()
        else:
            return None
    except Exception as e:
        return None

def predict_skills_with_ai(position: str, seniority_level: str) -> List[str]:
    prompt = f"""Predict 8-12 skills for: {position} ({seniority_level})

Return ONLY a JSON array like: ["Python", "Docker", "Leadership"]

Your response:"""

    response = call_watsonx(prompt, max_tokens=512)
    
    if response:
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            skills = json.loads(cleaned)
            if isinstance(skills, list) and len(skills) > 0:
                return skills
        except:
            pass
    
    return fallback_skills(position, seniority_level)

def recommend_courses_with_ai(skills: List[str], position: str, seniority_level: str) -> List[dict]:
    
    courses_text = "\n".join([
        f"{i+1}. {c['course_name']} - {', '.join(c['skills_covered'])}"
        for i, c in enumerate(COURSE_CATALOG)
    ])
    
    prompt = f"""Recommend top 5 courses for:
Position: {position}
Skills: {', '.join(skills)}

Courses:
{courses_text}
You ARE absolutely needed to recommend at least one course to every person!
Return JSON array:
[{{"course_name": "exact name", "priority": "high/medium/low", "reason": "why"}}]

Your response:"""

    response = call_watsonx(prompt, max_tokens=800)
    if response:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            
            recs = json.loads(cleaned)
            
            if isinstance(recs, list) and len(recs) > 0:
                enriched = []
                for rec in recs: 
                    course_name = rec.get('course_name', '').strip()
                    
                    course = next((c for c in COURSE_CATALOG if c['course_name'].lower() == course_name.lower()), None)
                    
                    if not course:
                        course = next((c for c in COURSE_CATALOG if c['course_name'].lower() in course_name.lower() or course_name.lower() in c['course_name'].lower()), None)
                        
                    if course:
                        enriched.append({
                            "course_name": course["course_name"],
                            "provider": course["provider"],
                            "duration_hours": course["duration_hours"],
                            "cost": course["cost"],
                            "priority": rec.get("priority", "medium"),
                            "matched_skills": [s for s in skills if s in course["skills_covered"]],
                            "reason": rec.get("reason", "Recommended for your role")
                        })
                    else:
                        enriched.append({
                            "course_name": course_name,
                            "provider": "AI Recommendation",
                            "duration_hours": 0,
                            "cost": 0,
                            "priority": rec.get("priority", "medium"),
                            "matched_skills": [], 
                            "reason": rec.get("reason", "Recommended by AI")
                        })
                if len(enriched) > 0:
                    return enriched
    return fallback_courses(skills)

def fallback_skills(position: str, seniority_level: str) -> List[str]:
    base = ["Communication", "Problem Solving", "Teamwork"]
    if "software" in position.lower() or "developer" in position.lower():
        base.extend(["Python", "JavaScript", "Git", "Docker"])
    elif "devops" in position.lower():
        base.extend(["Docker", "Kubernetes", "CI/CD", "AWS"])
    elif "data" in position.lower():
        base.extend(["Python", "SQL", "Machine Learning"])
    elif "product" in position.lower():
        base.extend(["Roadmapping", "Analytics", "Agile"])
    
    if "senior" in seniority_level.lower():
        base.extend(["Leadership", "Mentoring"])
    
    return list(set(base))

def fallback_courses(skills: List[str]) -> List[dict]:
    recommendations = []
    for course in COURSE_CATALOG:
        matched = [s for s in skills if s in course["skills_covered"]]
        if matched:
            recommendations.append({
                "course_name": course["course_name"],
                "provider": course["provider"],
                "duration_hours": course["duration_hours"],
                "cost": course["cost"],
                "priority": "high" if len(matched) >= 2 else "medium",
                "matched_skills": matched,
                "reason": f"Matches {', '.join(matched[:2])}"
            })
    recommendations.sort(key=lambda x: (0 if x["priority"]=="high" else 1, -len(x["matched_skills"])))
    return recommendations[:5]

@curator.post("/process-employees")
async def process_employees(request: ProcessEmployeesRequest):
    try:
        results = []
        for emp in request.employees:
            skills = predict_skills_with_ai(emp.position, emp.seniority_level)
            courses = recommend_courses_with_ai(skills, emp.position, emp.seniority_level)
            
            results.append({
                "employee_id": emp.employee_id,
                "employee_name": emp.employee_name,
                "position": emp.position,
                "seniority_level": emp.seniority_level,
                "predicted_skills": skills,
                "recommended_courses": courses
            })
        
        return {"total_employees": len(results), "employees": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@curator.post("/predict-skills")
async def predict_skills(request: SkillPredictionRequest):
    skills = predict_skills_with_ai(request.position, request.seniority_level)
    return {
        "employee_name": request.employee_name,
        "position": request.position,
        "seniority_level": request.seniority_level,
        "predicted_skills": skills,
        "prediction_method": "WatsonX AI (Cloud API)"
    }

@curator.get("/")
async def root():
    return {
        "service": "M&A Employee Onboarding System",
        "version": "6.0.0 - WatsonX API",
        "method": "Cloud AI (Fast!)",
        "status": "healthy"
    }

app.mount("/architect", architect)
app.mount("/auditor", auditor)
app.mount("/analyst", analyst)
app.mount("/curator", curator)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)