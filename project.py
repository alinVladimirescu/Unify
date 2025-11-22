from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import json

app = FastAPI()

WATSONX_API_KEY = "API_KEY_PLACEHOLDER"
WATSONX_PROJECT_ID = "PROJECT_ID_PLACEHOLDER"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_MODEL = "ibm/granite-13b-chat-v2"

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

class CourseRecommendationRequest(BaseModel):
    skills: List[str]
    position: Optional[str] = None
    seniority_level: Optional[str] = None

COURSE_CATALOG = [
    {"course_name": "Kubernetes Mastery", "provider": "Udemy", "duration_hours": 16, "cost": 399, "skills_covered": ["Kubernetes", "Docker"]},
    {"course_name": "Python for Data Science", "provider": "Coursera", "duration_hours": 24, "cost": 299, "skills_covered": ["Python", "Machine Learning"]},
    {"course_name": "AWS Solutions Architect", "provider": "A Cloud Guru", "duration_hours": 40, "cost": 499, "skills_covered": ["AWS", "Cloud"]},
    {"course_name": "Leadership Essentials", "provider": "LinkedIn", "duration_hours": 12, "cost": 199, "skills_covered": ["Leadership"]},
    {"course_name": "CI/CD Pipelines", "provider": "Pluralsight", "duration_hours": 8, "cost": 149, "skills_covered": ["CI/CD", "DevOps"]},
    {"course_name": "JavaScript Advanced", "provider": "Udemy", "duration_hours": 20, "cost": 349, "skills_covered": ["JavaScript", "React"]},
    {"course_name": "Terraform IaC", "provider": "HashiCorp", "duration_hours": 10, "cost": 249, "skills_covered": ["Terraform", "Cloud"]},
]

def get_watsonx_token():
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY}
    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to get token")
    return response.json()["access_token"]

def predict_skills_with_watsonx(position: str, seniority_level: str) -> List[str]:
    prompt = f"""Predict 8-12 skills for: {position} ({seniority_level})

Return ONLY a JSON array like: ["Python", "Docker", "Leadership"]

Your response:"""

    try:
        token = get_watsonx_token()
        headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        body = {
            "model_id": WATSONX_MODEL,
            "input": prompt,
            "parameters": {"decoding_method": "greedy", "max_new_tokens": 200, "temperature": 0.3},
            "project_id": WATSONX_PROJECT_ID
        }
        
        response = requests.post(WATSONX_URL, headers=headers, json=body)
        if response.status_code == 200:
            result = response.json()
            text = result["results"][0]["generated_text"].strip().replace("```json", "").replace("```", "").strip()
            skills = json.loads(text)
            if isinstance(skills, list) and len(skills) > 0:
                return skills
    except Exception as e:
        print(f"WatsonX Error: {e}")
    
    
    return fallback_skills(position, seniority_level)

def fallback_skills(position: str, seniority_level: str) -> List[str]:
    base = ["Communication", "Problem Solving"]
    if "software" in position.lower() or "developer" in position.lower():
        base.extend(["Python", "JavaScript", "Git", "Docker"])
    elif "devops" in position.lower():
        base.extend(["Docker", "Kubernetes", "CI/CD", "AWS"])
    elif "data" in position.lower():
        base.extend(["Python", "SQL", "Machine Learning"])
    elif "product" in position.lower():
        base.extend(["Roadmapping", "Analytics", "Agile"])
    
    if "senior" in seniority_level.lower():
        base.append("Leadership")
    
    return list(set(base))

def recommend_courses(skills: List[str]) -> List[dict]:
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
                "matched_skills": matched
            })
    recommendations.sort(key=lambda x: (0 if x["priority"]=="high" else 1, -len(x["matched_skills"])))
    return recommendations[:5]

@app.post("/process-employees")
async def process_employees(request: ProcessEmployeesRequest):
    try:
        print(f"\n📥 Processing {len(request.employees)} employees")
        
        results = []
        for emp in request.employees:
            print(f"   {emp.employee_name} - {emp.position}")
            
            skills = predict_skills_with_watsonx(emp.position, emp.seniority_level)
            courses = recommend_courses(skills)
            
            results.append({
                "employee_id": emp.employee_id,
                "employee_name": emp.employee_name,
                "position": emp.position,
                "seniority_level": emp.seniority_level,
                "predicted_skills": skills,
                "recommended_courses": courses
            })
        
        print(f"Done: {len(results)} employees\n")
        return {"total_employees": len(results), "employees": results}
    
    except Exception as e:
        print(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-skills")
async def predict_skills(request: SkillPredictionRequest):
    skills = predict_skills_with_watsonx(request.position, request.seniority_level)
    return {
        "employee_name": request.employee_name,
        "position": request.position,
        "seniority_level": request.seniority_level,
        "predicted_skills": skills,
        "prediction_method": "IBM WatsonX.ai"
    }

@app.post("/recommend-courses")
async def recommend_courses_endpoint(request: CourseRecommendationRequest):
    courses = recommend_courses(request.skills)
    return {"total_matches": len(courses), "courses": courses}

@app.get("/courses")
async def get_courses():
    return {"total_courses": len(COURSE_CATALOG), "courses": COURSE_CATALOG}

@app.get("/")
async def root():
    return {"service": "Course Recommender", "version": "3.0.0", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)