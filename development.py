from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import json
import requests
import os
import traceback

app = FastAPI(title="Employee Course Recommender", version="7.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WATSONX_API_KEY = "apikeyplaceholder"
WATSONX_PROJECT_ID = "projectidplaceholder"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
MODEL_ID = "ibm/granite-3-8b-instruct"

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
            print(f"✅ Loaded {len(catalog)} courses from {COURSE_CATALOG_FILE}")
            return catalog
    except FileNotFoundError:
        print(f"⚠️ Course catalog not found at {COURSE_CATALOG_FILE}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing course catalog: {e}")
        return []

COURSE_CATALOG = load_course_catalog()

def get_watsonx_token():
    """Get IAM token for Watsonx"""
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey", 
        "apikey": WATSONX_API_KEY
    }
    
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        print(f"❌ Failed to get IAM token: {e}")
        raise

def call_watsonx(prompt: str, max_tokens: int = 512) -> str:
    """Call Watsonx.ai for text generation"""
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
                "temperature": 0.3,
                "stop_sequences": []
            },
            "project_id": WATSONX_PROJECT_ID
        }
        
        response = requests.post(WATSONX_URL, headers=headers, json=body, timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            return result["results"][0]["generated_text"].strip()
        else:
            print(f"❌ Watsonx error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Watsonx call failed: {e}")
        return None

def predict_skills_with_ai(position: str, seniority_level: str) -> List[str]:
    """Predict skills using AI"""
    prompt = f"""Predict 8-12 skills for: {position} ({seniority_level})

Return ONLY a JSON array like: ["Python", "Docker", "Leadership"]

Your response:"""

    try:
        response = call_watsonx(prompt, max_tokens=512)
        
        if response:
            # Clean the response
            cleaned = response.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON array - find first [ and matching ]
            start = cleaned.find("[")
            if start != -1:
                # Find the matching closing bracket
                bracket_count = 0
                end = start
                for i in range(start, len(cleaned)):
                    if cleaned[i] == '[':
                        bracket_count += 1
                    elif cleaned[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    cleaned = cleaned[start:end]
            
            skills = json.loads(cleaned)
            
            if isinstance(skills, list) and len(skills) > 0:
                print(f"✅ Predicted {len(skills)} skills for {position}")
                return skills
                
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error in skills prediction: {e}")
    except Exception as e:
        print(f"⚠️ Error predicting skills: {e}")
    
    # Fallback
    print(f"📋 Using fallback skills for {position}")
    return fallback_skills(position, seniority_level)

def recommend_courses_with_ai(skills: List[str], position: str, seniority_level: str) -> List[dict]:
    """Recommend courses using AI"""
    
    if not COURSE_CATALOG:
        print("⚠️ No course catalog available, using fallback")
        return fallback_courses(skills)
    
    # Prepare course catalog text
    courses_text = "\n".join([
        f"{i+1}. {c['course_name']} - {', '.join(c['skills_covered'])}"
        for i, c in enumerate(COURSE_CATALOG[:30])  # Limit to avoid token limits
    ])
    
    prompt = f"""Recommend top 5 courses for:
Position: {position}
Seniority: {seniority_level}
Skills: {', '.join(skills)}

Available Courses:
{courses_text}

You MUST recommend at least one course to every person!

Return JSON array with this exact format:
[{{"course_name": "exact name from list", "priority": "high", "reason": "why"}}]

Your response:"""

    try:
        response = call_watsonx(prompt, max_tokens=1000)
        
        if response:
            # Clean the response
            cleaned = response.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON array - find first [ and first ] after it
            start = cleaned.find("[")
            if start != -1:
                # Find the matching closing bracket
                bracket_count = 0
                end = start
                for i in range(start, len(cleaned)):
                    if cleaned[i] == '[':
                        bracket_count += 1
                    elif cleaned[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    cleaned = cleaned[start:end]
            
            recs = json.loads(cleaned)
            
            if isinstance(recs, list) and len(recs) > 0:
                enriched = []
                
                for rec in recs:
                    course_name = rec.get('course_name', '').strip()
                    
                    # Find matching course in catalog
                    course = next(
                        (c for c in COURSE_CATALOG if c['course_name'].lower() == course_name.lower()), 
                        None
                    )
                    
                    # Try fuzzy match if exact match fails
                    if not course:
                        course = next(
                            (c for c in COURSE_CATALOG 
                             if c['course_name'].lower() in course_name.lower() 
                             or course_name.lower() in c['course_name'].lower()), 
                            None
                        )
                    
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
                        # Include AI recommendation even if not in catalog
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
                    print(f"✅ Recommended {len(enriched)} courses")
                    return enriched
                    
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error in course recommendations: {e}")
        print(f"Raw response: {response[:200] if response else 'None'}...")
    except Exception as e:
        print(f"⚠️ Error recommending courses: {e}")
        traceback.print_exc()
    
    # Fallback
    print("📋 Using fallback course recommendations")
    return fallback_courses(skills)

def fallback_skills(position: str, seniority_level: str) -> List[str]:
    """Fallback skill prediction"""
    base = ["Communication", "Problem Solving", "Teamwork"]
    
    position_lower = position.lower()
    
    if "software" in position_lower or "developer" in position_lower:
        base.extend(["Python", "JavaScript", "Git", "Docker", "REST APIs"])
    elif "devops" in position_lower:
        base.extend(["Docker", "Kubernetes", "CI/CD", "AWS", "Terraform"])
    elif "data" in position_lower:
        base.extend(["Python", "SQL", "Machine Learning", "Statistics"])
    elif "product" in position_lower:
        base.extend(["Roadmapping", "Analytics", "Agile", "User Research"])
    elif "design" in position_lower:
        base.extend(["Figma", "UX Research", "Prototyping"])
    else:
        base.extend(["Project Management", "Time Management"])
    
    if "senior" in seniority_level.lower() or "lead" in seniority_level.lower():
        base.extend(["Leadership", "Mentoring", "Strategic Thinking"])
    
    return list(set(base))

def fallback_courses(skills: List[str]) -> List[dict]:
    """Fallback course recommendations based on skill matching"""
    if not COURSE_CATALOG:
        return [{
            "course_name": "General Skills Development",
            "provider": "Internal",
            "duration_hours": 10,
            "cost": 0,
            "priority": "medium",
            "matched_skills": skills[:3],
            "reason": "No course catalog available"
        }]
    
    recommendations = []
    
    for course in COURSE_CATALOG:
        matched = [s for s in skills if s.lower() in [sc.lower() for sc in course.get("skills_covered", [])]]
        
        if matched:
            recommendations.append({
                "course_name": course["course_name"],
                "provider": course.get("provider", "Unknown"),
                "duration_hours": course.get("duration_hours", 0),
                "cost": course.get("cost", 0),
                "priority": "high" if len(matched) >= 2 else "medium",
                "matched_skills": matched,
                "reason": f"Matches {', '.join(matched[:2])}"
            })
    
    # Sort by priority and number of matched skills
    recommendations.sort(key=lambda x: (0 if x["priority"]=="high" else 1, -len(x["matched_skills"])))
    
    return recommendations[:5]

@app.post("/process-employees")
async def process_employees(request: ProcessEmployeesRequest):
    """Process multiple employees and recommend courses"""
    try:
        print(f"📊 Processing {len(request.employees)} employees")
        
        results = []
        
        for i, emp in enumerate(request.employees):
            print(f"👤 Processing employee {i+1}/{len(request.employees)}: {emp.employee_name}")
            
            try:
                # Predict skills
                skills = predict_skills_with_ai(emp.position, emp.seniority_level)
                
                # Recommend courses
                courses = recommend_courses_with_ai(skills, emp.position, emp.seniority_level)
                
                results.append({
                    "employee_id": emp.employee_id,
                    "employee_name": emp.employee_name,
                    "position": emp.position,
                    "seniority_level": emp.seniority_level,
                    "predicted_skills": skills,
                    "recommended_courses": courses
                })
                
                print(f"✅ Completed {emp.employee_name}: {len(skills)} skills, {len(courses)} courses")
                
            except Exception as emp_error:
                print(f"❌ Error processing {emp.employee_name}: {emp_error}")
                traceback.print_exc()
                
                # Add error result
                results.append({
                    "employee_id": emp.employee_id,
                    "employee_name": emp.employee_name,
                    "position": emp.position,
                    "seniority_level": emp.seniority_level,
                    "predicted_skills": [],
                    "recommended_courses": [],
                    "error": str(emp_error)
                })
        
        print(f"✅ Completed processing {len(results)} employees")
        
        return {
            "total_employees": len(results),
            "employees": results,
            "status": "success"
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR in process_employees: {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {error_msg}")

@app.post("/predict-skills")
async def predict_skills(request: SkillPredictionRequest):
    """Predict skills for one employee"""
    try:
        skills = predict_skills_with_ai(request.position, request.seniority_level)
        return {
            "employee_name": request.employee_name,
            "position": request.position,
            "seniority_level": request.seniority_level,
            "predicted_skills": skills,
            "prediction_method": "WatsonX AI (Cloud API)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/courses")
async def get_courses():
    """Get all available courses"""
    return {
        "total_courses": len(COURSE_CATALOG),
        "courses": COURSE_CATALOG
    }

@app.get("/health")
async def health_check():
    """Health check with diagnostics"""
    try:
        token = get_watsonx_token()
        return {
            "status": "healthy",
            "watsonx_auth": "success",
            "courses_loaded": len(COURSE_CATALOG),
            "model": MODEL_ID
        }
    except Exception as e:
        return {
            "status": "degraded",
            "watsonx_auth": "failed",
            "courses_loaded": len(COURSE_CATALOG),
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "M&A Employee Onboarding System",
        "version": "7.0.0 - Enhanced Error Handling",
        "method": "WatsonX Cloud AI",
        "status": "healthy",
        "endpoints": {
            "process_employees": "/process-employees",
            "predict_skills": "/predict-skills",
            "courses": "/courses",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Employee Course Recommender API...")
    print(f"📚 Loaded {len(COURSE_CATALOG)} courses")
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=True)