from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import json
import requests
import os

app = FastAPI()
load_dotenv()

WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_URL = os.environ.get("WATSONX_URL")
MODEL_ID = os.environ.get("MODEL_ID")

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

@app.post("/process-employees")
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

@app.post("/predict-skills")
async def predict_skills(request: SkillPredictionRequest):
    skills = predict_skills_with_ai(request.position, request.seniority_level)
    return {
        "employee_name": request.employee_name,
        "position": request.position,
        "seniority_level": request.seniority_level,
        "predicted_skills": skills,
        "prediction_method": "WatsonX AI (Cloud API)"
    }

@app.get("/")
async def root():
    return {
        "service": "M&A Employee Onboarding System",
        "version": "6.0.0 - WatsonX API",
        "method": "Cloud AI (Fast!)",
        "status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)