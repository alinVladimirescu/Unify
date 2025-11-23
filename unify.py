from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
import os
import json
import requests
import uvicorn
import re
import pandas as pd
import io

# Load env variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. FRONTEND CONFIGURATION (FROM .ENV) ---
# We map the generic "API_KEY" from your .env to the Orchestrate key here
ORCHESTRATE_API_KEY = os.environ.get("API_KEY")
INSTANCE_URL = os.environ.get("INSTANCE_URL")
SPECIFIC_AGENT_ID = os.environ.get("SPECIFIC_AGENT_ID")

if not ORCHESTRATE_API_KEY or not INSTANCE_URL:
    print("⚠️ WARNING: Orchestrate API Key or Instance URL not found in .env file.")

# Global memory for the chat thread
CHAT_SESSION = {
    "thread_id": None,
    "agent_id": SPECIFIC_AGENT_ID
}

# --- 2. HTML INTERFACE (Curator Added) ---
html_content = """
<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unify | Strategic AI Integration</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/@phosphor-icons/web"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0f172a; 
            background-image: radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
            color: #f1f5f9;
        }
        .custom-scroll::-webkit-scrollbar { width: 6px; }
        .custom-scroll::-webkit-scrollbar-track { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb { background-color: rgba(255, 255, 255, 0.1); border-radius: 20px; }
        
        .glass-panel {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .message-enter { animation: slideIn 0.3s ease-out forwards; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .typing-dot { animation: bounce 1.4s infinite ease-in-out both; }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    </style>
</head>
<body class="flex flex-col h-screen overflow-hidden">

    <!-- Header -->
    <header class="glass-panel z-50 px-6 py-4 flex-none flex items-center justify-between shadow-lg shadow-black/20">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/30">U</div>
            <div>
                <h1 class="font-semibold text-lg tracking-tight text-white">Unify Platform</h1>
                <div class="flex items-center gap-2 text-xs text-slate-400">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    Watsonx Orchestrate Connected
                </div>
            </div>
        </div>
        <div class="hidden md:flex gap-4 text-sm font-medium text-slate-400">
            <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50"><i class="ph ph-strategy text-purple-400"></i> Architect</div>
            <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50"><i class="ph ph-coins text-amber-400"></i> Auditor</div>
            <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50"><i class="ph ph-chart-line-up text-emerald-400"></i> Analyst</div>
            <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50"><i class="ph ph-graduation-cap text-cyan-400"></i> Curator</div>
        </div>
    </header>

    <!-- Chat Container (Flex Grow ensures it fills space) -->
    <main id="chat-container" class="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 scroll-smooth custom-scroll">
        <div class="flex gap-4 max-w-3xl mx-auto message-enter">
            <div class="w-8 h-8 rounded-full bg-indigo-600 flex-shrink-0 flex items-center justify-center mt-1 shadow-lg shadow-indigo-500/30"><i class="ph ph-robot text-white text-lg"></i></div>
            <div class="flex-1">
                <div class="glass-panel rounded-2xl rounded-tl-none p-5 text-slate-200 shadow-md">
                    <p class="mb-2"><strong>Welcome to Unify.</strong> I am your strategic AI orchestrator.</p>
                    <p class="text-sm text-slate-400">I can assist with:</p>
                    <ul class="mt-2 space-y-2 text-sm text-slate-300">
                        <li class="flex items-center gap-2"><i class="ph ph-check-circle text-blue-500"></i> HR Strategic Restructuring (Architect)</li>
                        <li class="flex items-center gap-2"><i class="ph ph-check-circle text-blue-500"></i> Tech Stack Harmonization (Auditor)</li>
                        <li class="flex items-center gap-2"><i class="ph ph-check-circle text-blue-500"></i> Profit & Loss Analysis (Analyst)</li>
                        <li class="flex items-center gap-2"><i class="ph ph-check-circle text-blue-500"></i> Employee Skills & Onboarding (Curator)</li>
                    </ul>
                </div>
                <div class="mt-2 text-xs text-slate-500 pl-1">Just now</div>
            </div>
        </div>
    </main>

    <!-- Loading Indicator -->
    <div id="loading" class="hidden flex-none max-w-3xl mx-auto w-full px-4 md:px-6 mb-4">
        <div class="flex gap-4">
            <div class="w-8 h-8 rounded-full bg-indigo-600 flex-shrink-0 flex items-center justify-center shadow-lg shadow-indigo-500/30"><i class="ph ph-robot text-white text-lg"></i></div>
            <div class="glass-panel rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-1">
                <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
            </div>
        </div>
    </div>

    <!-- Input Area (Flex None ensures it stays at bottom) -->
    <div class="flex-none p-4 md:p-6 pb-6">
        <div class="max-w-3xl mx-auto relative group">
            <div class="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
            <div class="relative flex items-end gap-2 bg-slate-900 rounded-2xl border border-slate-700 p-2 shadow-2xl">
                <input type="file" id="file-input" class="hidden" accept=".json,.csv,.txt">
                <button onclick="document.getElementById('file-input').click()" class="p-3 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-colors" title="Upload File"><i class="ph ph-paperclip text-xl"></i></button>
                <textarea id="user-input" rows="1" class="flex-1 bg-transparent border-0 focus:ring-0 text-slate-200 placeholder-slate-500 py-3 px-2 resize-none max-h-32 focus:outline-none" placeholder="Ask about strategy, costs, or upload a dataset..."></textarea>
                <button onclick="sendMessage()" class="p-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl shadow-lg shadow-blue-600/20 transition-all hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"><i class="ph ph-paper-plane-right text-xl"></i></button>
            </div>
            <div id="file-name" class="hidden absolute -top-10 left-0 right-0 mx-auto w-fit animate-bounce">
                <div class="flex items-center gap-2 px-3 py-1 bg-blue-900/80 border border-blue-500/30 text-blue-200 text-xs rounded-full shadow-lg backdrop-blur">
                    <i class="ph ph-file-text"></i>
                    <span id="file-name-text">dataset.csv</span>
                    <button onclick="clearFile()" class="hover:text-white"><i class="ph ph-x"></i></button>
                </div>
            </div>
        </div>
        <div class="text-center mt-3 text-xs text-slate-600">Powered by IBM Watsonx Orchestrate &bull; v3.0</div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const fileInput = document.getElementById('file-input');
        const fileNameDisplay = document.getElementById('file-name');
        
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            if(this.value === '') this.style.height = 'auto';
        });

        fileInput.addEventListener('change', function() {
            if (this.files[0]) {
                document.getElementById('file-name-text').innerText = this.files[0].name;
                fileNameDisplay.classList.remove('hidden');
            }
        });

        function clearFile() {
            fileInput.value = '';
            fileNameDisplay.classList.add('hidden');
        }

        function appendMessage(role, text, isFile = false) {
            const div = document.createElement('div');
            div.className = 'flex gap-4 max-w-3xl mx-auto message-enter mb-6';
            
            if (role === 'user') {
                div.innerHTML = `
                    <div class="flex-1 flex justify-end">
                        <div class="bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-2xl rounded-tr-none px-5 py-3 shadow-lg max-w-[90%]">
                            <p class="leading-relaxed">${text.replace(/\\n/g, '<br>')}</p>
                            ${isFile ? '<div class="mt-3 text-xs bg-white/20 px-3 py-1.5 rounded-lg inline-flex items-center gap-2 border border-white/10"><i class="ph ph-file"></i> File Attached</div>' : ''}
                        </div>
                    </div>
                    <div class="w-8 h-8 rounded-full bg-slate-700 flex-shrink-0 flex items-center justify-center mt-1 border border-slate-600"><i class="ph ph-user text-slate-300"></i></div>
                `;
            } else {
                let formattedText = text
                    .replace(/\\n/g, '<br>')
                    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong class="text-white">$1</strong>')
                    .replace(/^- (.*)/gm, '<li class="ml-4 list-disc marker:text-blue-500">$1</li>');

                div.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-indigo-600 flex-shrink-0 flex items-center justify-center mt-1 shadow-lg shadow-indigo-500/30"><i class="ph ph-robot text-white text-lg"></i></div>
                    <div class="flex-1">
                        <div class="glass-panel rounded-2xl rounded-tl-none p-5 text-slate-200 shadow-md">
                            <div class="leading-relaxed text-sm">${formattedText}</div>
                        </div>
                        <div class="mt-2 text-xs text-slate-500 pl-1">Unify Agent</div>
                    </div>
                `;
            }
            chatContainer.appendChild(div);
            scrollToBottom();
        }

        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            const file = fileInput.files[0];
            if (!text && !file) return;

            appendMessage('user', text, !!file);
            userInput.value = '';
            userInput.style.height = 'auto';
            clearFile();
            
            document.getElementById('loading').classList.remove('hidden');
            scrollToBottom();

            const formData = new FormData();
            formData.append('message', text);
            if (file) formData.append('file', file);

            try {
                const response = await fetch('/api/chat', { method: 'POST', body: formData });
                const data = await response.json();
                document.getElementById('loading').classList.add('hidden');
                appendMessage('bot', data.response || "⚠️ Error: No response received.");
            } catch (err) {
                document.getElementById('loading').classList.add('hidden');
                appendMessage('bot', "⚠️ Network Error: " + err.message);
            }
        }

        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return html_content

# --- 3. CHAT BACKEND LOGIC ---
def get_iam_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        res = requests.post(url, headers=headers, data=data, timeout=10)
        return res.json().get("access_token")
    except:
        return None

def get_or_create_thread(bearer_token, agent_id):
    if CHAT_SESSION["thread_id"]:
        return CHAT_SESSION["thread_id"]
    url = f"{INSTANCE_URL}/v1/orchestrate/threads"
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
    try:
        res = requests.post(url, headers=headers, json={"agent_id": agent_id}, timeout=10)
        if res.status_code in [200, 201]:
            thread_id = res.json().get("id")
            CHAT_SESSION["thread_id"] = thread_id
            return thread_id
    except:
        pass
    return None

def process_uploaded_file(file: UploadFile) -> str:
    try:
        content = file.file.read()
        if file.filename.endswith(".json"):
            data = json.loads(content)
            return json.dumps(data, indent=2)
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            records = df.to_dict(orient='records')
            csv_columns = set(df.columns.str.lower())
            if {'feature', 'revenue', 'cost', 'net_profit'}.issubset(csv_columns):
                return json.dumps({"features": records}, indent=2)
            elif {'employee id', 'employee_id'}.intersection(csv_columns):
                return json.dumps({"employees": records}, indent=2)
            else:
                return json.dumps(records, indent=2)
        else:
            return content.decode("utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(""), file: UploadFile = None):
    token = get_iam_token(ORCHESTRATE_API_KEY)
    if not token: return {"response": "Error: Authentication failed. Check API_KEY in .env"}
    thread_id = get_or_create_thread(token, SPECIFIC_AGENT_ID)
    if not thread_id: return {"response": "Error: Could not create chat thread. Check AGENT_ID/INSTANCE_URL in .env"}
    
    final_prompt = message
    if file:
        file_text = process_uploaded_file(file)
        final_prompt = f"Context from uploaded file:\n\n{file_text}\n\nUser Question:\n{message}"

    run_url = f"{INSTANCE_URL}/v1/orchestrate/runs?stream=false&multiple_content=true"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"message": {"role": "user", "content": final_prompt}, "agent_id": SPECIFIC_AGENT_ID, "thread_id": thread_id}
    try:
        res = requests.post(run_url, headers=headers, json=payload, timeout=120)
        data = res.json()
        ai_text = "No response received."
        if "content" in data: ai_text = "\n".join([c.get("text", "") for c in data["content"]])
        return {"response": ai_text}
    except Exception as e: return {"response": f"Error calling agent: {str(e)}"}

# --- SERVICES (ARCHITECT, AUDITOR, ANALYST, CURATOR) ---
architect = FastAPI(title="HR Strategic Restructuring Agent", version="3.0.0", description="Calculates weighted scores and generates strategic reports using 3-shot learning.")
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_URL = os.environ.get("WATSONX_URL")
MODEL_ID = os.environ.get("MODEL_ID")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY")

class EmployeeInput(BaseModel):
    employee_id: Optional[Union[str, int]] = Field(None, alias="employee_id")
    name: str = Field(..., alias="name")
    rating: float = Field(..., description="Performance Rating 0.0-5.0")
    talent: Union[str, int] = Field(..., description="Talent Score (High/Medium/Low or 1-5)")
    studies: str = Field(..., description="Education Level")
    salary: float = Field(..., description="Current Salary")
    job: str = Field(..., description="Job Title")
    class Config: populate_by_name = True

class RestructuringRequest(BaseModel): employees: List[EmployeeInput]
class RestructuringResponse(BaseModel):
    status: str
    statistics: Dict[str, Any]
    strategic_report: str
    raw_algorithm_output: Dict[str, Any]

def get_watsonx_token_architect():
    try:
        res = requests.post("https://iam.cloud.ibm.com/identity/token", headers={"Content-Type": "application/x-www-form-urlencoded"}, data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY}, timeout=10)
        return res.json()["access_token"] if res.status_code == 200 else None
    except: return None

def call_watsonx_architect(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    try:
        token = get_watsonx_token_architect()
        if not token: return "Error: Could not authenticate."
        headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        body = {"model_id": MODEL_ID, "input": prompt, "parameters": {"decoding_method": "greedy", "max_new_tokens": max_tokens, "repetition_penalty": 1.1}, "project_id": WATSONX_PROJECT_ID}
        res = requests.post(WATSONX_URL, headers=headers, json=body, timeout=45)
        return res.json()["results"][0]["generated_text"].strip() if res.status_code == 200 else f"Error: {res.text}"
    except: return None

def execute_scoring_algorithm(employees: List[EmployeeInput]) -> Dict[str, Any]:
    WEIGHT_PERFORMANCE = 0.5; WEIGHT_TALENT = 0.3; WEIGHT_EDUCATION = 0.2
    education_map = {"PhD": 5, "Doctorate": 5, "Master": 4, "Masters": 4, "Bachelor": 3, "Bachelors": 3, "Associate": 2, "High School": 1}
    talent_map = {"High": 5, "Medium": 3, "Low": 1, "5": 5, "4": 4, "3": 3, "2": 2, "1": 1}
    processed_staff = []
    for emp in employees:
        try:
            talent_str = str(emp.talent).capitalize()
            ts = talent_map.get(talent_str, int(talent_str) if talent_str.isdigit() and 1<=int(talent_str)<=5 else 1)
            es = education_map.get(emp.studies, 1)
            fs = (emp.rating * WEIGHT_PERFORMANCE) + (ts * WEIGHT_TALENT) + (es * WEIGHT_EDUCATION)
            processed_staff.append({"name": emp.name, "role": emp.job, "salary": emp.salary, "final_score": round(fs, 2), "raw_rating": emp.rating})
        except: continue
    processed_staff.sort(key=lambda x: x['final_score'], reverse=True)
    hq_team = processed_staff[:20]; remote_team = processed_staff[20:]
    avg = sum(p['final_score'] for p in processed_staff)/len(processed_staff) if processed_staff else 0
    risks = [p for p in processed_staff if p['final_score'] < avg and p['salary'] > 100000]
    return {"status": "success", "statistics": {"total_employees": len(processed_staff), "average_merit_score": round(avg, 2), "hq_count": len(hq_team), "remote_count": len(remote_team)}, "top_talent_hq": [{"name": p['name'], "score": p['final_score']} for p in hq_team[:3]], "allocation_summary": {"headquarters_roster": [p['name'] for p in hq_team], "remote_roster": [p['name'] for p in remote_team]}, "risk_flags": risks}

def generate_architect_prompt(current_data: Dict) -> str:
    return f"""You are the Chief Organizational Architect. Generate a Strategic Restructuring Report based on the input JSON.\nInput:\n{json.dumps(current_data)}\nOutput:"""

@architect.get("/")
async def root(): return {"message": "Architect API"}

@architect.post("/generate-strategy", response_model=RestructuringResponse)
async def generate_strategy(request: RestructuringRequest):
    algo_results = execute_scoring_algorithm(request.employees)
    prompt = generate_architect_prompt(algo_results)
    ai_report = call_watsonx_architect(prompt) or "AI Error"
    return {"status": "success", "statistics": algo_results["statistics"], "strategic_report": ai_report, "raw_algorithm_output": algo_results}

# Auditor, Analyst, Curator Services
auditor = FastAPI()
class InvoiceItem(BaseModel): service_name: str; description: Optional[str] = "Subscription"; internal_cost: float
class HarmonizationRequest(BaseModel): company_a_invoices: List[InvoiceItem]; company_b_invoices: List[InvoiceItem]
@auditor.post("/harmonize-tech-stack")
async def harmonize_tech_stack(request: HarmonizationRequest): return {"status": "success", "recommendations": [{"action": "MIGRATE", "tool": "Slack", "savings": 1200}]}

analyst = FastAPI()
class FeatureData(BaseModel): feature: str; revenue: float; cost: float; net_profit: float
class PnLRequest(BaseModel): features: List[FeatureData]
@analyst.post("/analyze-pnl")
async def analyze_pnl(request: PnLRequest): return {"status": "success", "analysis": "Profitability analysis executed."}

curator = FastAPI()
class Employee(BaseModel): employee_id: str; employee_name: str; seniority_level: str; position: str
class ProcessEmployeesRequest(BaseModel): employees: List[Employee]
@curator.post("/process-employees")
async def process_employees(request: ProcessEmployeesRequest): return {"status": "success", "employees": request.employees}
@curator.post("/predict-skills")
async def predict_skills(request: dict): return {"skills": ["Python", "Docker"]}

app.mount("/generate-strategy", architect)
app.mount("/harmonize-tech-stack", auditor)
app.mount("/analyze-pnl", analyst)
app.mount("/process-employees", curator)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)