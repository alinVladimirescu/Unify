from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import requests
import re
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

WATSONX_API_KEY = "apikeyplaceholder"
WATSONX_PROJECT_ID ="projectidplaceholder"
WATSONX_URL =   "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
MODEL_ID = "ibm/granite-3-8b-instruct"

SEARCH_API_KEY = "apikeyplaceholder"
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

@app.post("/harmonize-tech-stack")
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

@app.get("/")
async def root():
    return {"service": "Tech Stack Harmonization Agent", "status": "active", "version": "3.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)