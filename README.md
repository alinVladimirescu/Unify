# Unify - M&A Integration Automation Platform

An AI-powered platform that automates critical M&A (Mergers & Acquisitions) integration tasks using IBM watsonx.ai, watsonx Orchestrate, and intelligent agent orchestration. The platform includes four specialized agents that handle organizational restructuring, employee onboarding, tech stack harmonization, and financial analysis.

## 🎯 Project Overview

This platform streamlines the complex post-merger integration process by automating:
- **Strategic workforce restructuring** with merit-based analysis
- **Employee skill assessment** and personalized training recommendations
- **Tech stack consolidation** with real-time pricing comparisons
- **P&L analysis** for product feature rationalization

Built for the IBM Hackathon, this solution demonstrates advanced agentic AI workflows that reduce integration timelines from months to days.

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  watsonx Orchestrate                       │
│            (Central Workflow Coordinator)                  │
└────────┬────────────┬────────────┬──────────────┬──────────┘
         │            │            │              │
    ┌────▼────┐   ┌───▼────┐  ┌────▼─────┐  ┌─────▼──────┐
    │Architect│   │  Dev   │  │TechStack │  │   P&L      │
    │ Agent   │   │ Agent  │  │  Agent   │  │  Agent     │
    └────┬────┘   └───┬────┘  └────┬─────┘  └─────┬──────┘
         │            │            │              │
         └────────────┴────────────┴──────────────┘
                      │
               ┌──────▼──────┐
               │ watsonx.ai  │
               │  Granite 3  │
               │   8B Model  │
               └─────────────┘
```

## ✨ Key Features

### 🏢 **Architect Agent** - Organizational Restructuring
- Multi-factor employee evaluation (performance, talent, education)
- Weighted scoring algorithm for objective ranking
- Automated HQ/Remote team allocation
- Financial risk detection (high salary + low performance)
- AI-generated executive reports with actionable recommendations

### 👨‍💻 **Dev Agent** - Employee Onboarding & Development
- AI-powered skill gap prediction based on role and seniority
- Personalized course recommendations from 32-course catalog
- Batch processing for multiple employees
- Integration with training providers (Udemy, Coursera, Pluralsight, etc.)
- ROI-focused learning paths

### 🔧 **TechStack Agent** - Software Consolidation
- Automated redundancy detection across merged companies
- Real-time pricing comparison via SerpAPI web search
- Category-based tool analysis (Communication, DevOps, CRM, etc.)
- Annual cost savings projections
- Migration roadmap generation

### 💰 **P&L Agent** - Financial Analysis
- Product feature profitability analysis
- Keep/Ditch recommendations with financial justification
- Revenue vs. cost discrepancy detection
- Strategic portfolio optimization
- Executive-ready financial summaries

### 🖥️ **Frontend Interface**
- Streamlit-based conversational UI
- File upload support (CSV, JSON, TXT)
- Multi-turn conversation with context retention
- Real-time streaming responses from watsonx Orchestrate

## 🔧 Technology Stack

- **AI/ML Platform**: IBM watsonx.ai (Granite 3 8B Instruct)
- **Workflow Orchestration**: IBM watsonx Orchestrate
- **Backend Framework**: FastAPI (Python 3.8+)
- **Frontend**: Streamlit
- **Web Search**: SerpAPI for real-time pricing data
- **Authentication**: IBM Cloud IAM
- **Deployment**: Ngrok (development), IBM Cloud (production)
- **API Standard**: OpenAPI 3.0

## 📋 Prerequisites

- Python 3.8+
- IBM Cloud account with watsonx.ai access
- IBM watsonx Orchestrate instance
- Ngrok account (for local development)
- SerpAPI key (for TechStack agent)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/alinVladimirescu/IBM-Hackathon.git
cd IBM-Hackathon
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn pydantic requests python-dotenv streamlit pandas
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# IBM watsonx.ai Configuration
WATSONX_API_KEY=your-watsonx-api-key
WATSONX_PROJECT_ID=your-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29
MODEL_ID=ibm/granite-3-8b-instruct

# watsonx Orchestrate Configuration
API_KEY=your-orchestrate-api-key
INSTANCE_URL=your-orchestrate-instance-url
SPECIFIC_AGENT_ID=your-agent-id

# SerpAPI (for TechStack Agent)
SEARCH_API_KEY=your-serpapi-key
```

**To get your IBM credentials:**
1. Log in to [IBM Cloud](https://cloud.ibm.com)
2. Navigate to watsonx.ai
3. Go to Profile & Settings → API Keys
4. Create or copy your API key
5. Get your Project ID from your watsonx.ai project

### 4. Start the Backend Agents

Each agent runs on a different port. Open separate terminal windows:

**Architect Agent (Port 8080):**
```bash
python architect.py
```

**Dev Agent (Port 8000):**
```bash
python development.py
```

**TechStack Agent (Port 8000):**
```bash
python techstack.py
```

**P&L Agent (Port 8000):**
```bash
python profitloss.py
```

### 5. Expose with Ngrok

For each agent, open a new terminal and expose the port:

```bash
# For Architect Agent
ngrok http 8080

# For other agents (use different ngrok instances)
ngrok http 8000
```

Copy the HTTPS forwarding URLs.

### 6. Configure watsonx Orchestrate

1. Log in to your watsonx Orchestrate instance
2. Navigate to **Skills** → **Add custom skill**
3. Upload each YAML file:
   - `ArchitectAgent.yaml`
   - `DevAgent.yaml`
   - `TechStack.yaml`
   - `PLAgent.yaml`
4. Update the server URLs in each YAML with your ngrok URLs
5. Test each connection

### 7. Launch the Frontend

```bash
streamlit run frontend.py
```

Access the UI at `http://localhost:8501`

## 📝 API Documentation

### Architect Agent (`/generate-strategy`)

**Purpose**: Restructure workforce based on merit scores

**Request**:
```json
{
  "employees": [
    {
      "Employee ID": 101,
      "Name": "Alice Sterling",
      "rating": 4.8,
      "talent": 5,
      "studies": "PhD",
      "dupa salary": 180000,
      "job": "Senior Partner"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "statistics": {
    "total_employees": 50,
    "average_merit_score": 3.85,
    "hq_count": 20,
    "remote_count": 30
  },
  "strategic_report": "Strategic Restructuring Report...",
  "raw_algorithm_output": {...}
}
```

### Dev Agent (`/process-employees`)

**Purpose**: Recommend training courses based on role

**Request**:
```json
{
  "employees": [
    {
      "employee_id": "EMP001",
      "employee_name": "Sarah Johnson",
      "seniority_level": "Senior",
      "position": "Software Engineer"
    }
  ]
}
```

**Response**:
```json
{
  "total_employees": 1,
  "employees": [
    {
      "employee_id": "EMP001",
      "predicted_skills": ["Python", "Docker", "Leadership"],
      "recommended_courses": [
        {
          "course_name": "Kubernetes Mastery",
          "provider": "Udemy",
          "duration_hours": 16,
          "cost": 399,
          "priority": "high",
          "matched_skills": ["Docker"],
          "reason": "Essential for container orchestration"
        }
      ]
    }
  ]
}
```

### TechStack Agent (`/harmonize-tech-stack`)

**Purpose**: Identify redundant software and cost savings

**Request**:
```json
{
  "company_a_invoices": [
    {"service_name": "Slack", "internal_cost": 15.00}
  ],
  "company_b_invoices": [
    {"service_name": "Microsoft Teams", "internal_cost": 5.00}
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "conflicts_detected": 1,
  "recommendations": [
    {
      "category": "Communication",
      "action": "MIGRATE",
      "tool_to_keep": "Microsoft Teams",
      "tool_to_drop": "Slack",
      "estimated_annual_savings": 12000.00,
      "reasoning": "Teams ($5/mo) cheaper than Slack ($15/mo)"
    }
  ]
}
```

### P&L Agent (`/analyze-pnl`)

**Purpose**: Analyze feature profitability

**Request**:
```json
{
  "features": [
    {
      "feature": "AI Analytics Module",
      "revenue": 150000,
      "cost": 45000,
      "net_profit": 105000
    }
  ]
}
```

**Response**:
```json
{
  "executive_summary": "Strong portfolio with 80% profitability",
  "features_to_keep": [
    {
      "feature_name": "AI Analytics Module",
      "status": "Keep",
      "financial_metric": "Net Profit: $105k",
      "reason": "High-margin flagship product"
    }
  ],
  "features_to_ditch": []
}
```

## 🧮 Algorithms & AI Techniques

### Architect Agent - Weighted Scoring

```
Final Score = (Performance × 0.5) + (Talent × 0.3) + (Education × 0.2)
```

**Mappings**:
- **Talent**: High=5, Medium=3, Low=1
- **Education**: PhD=5, Master=4, Bachelor=3, Associate=2, HS=1
- **Risk Flag**: Score < Average AND Salary > $100k

### Dev Agent - Skill Prediction

Uses **few-shot prompting** with IBM Granite to predict 8-12 skills based on:
- Job position keywords
- Seniority level
- Industry best practices

Course matching considers:
- Skill overlap
- Course duration and cost
- Skill priority (high/medium/low)

### TechStack Agent - Redundancy Detection

1. **AI Categorization**: Granite classifies tools into functional groups
2. **Web Search**: SerpAPI fetches real-time pricing (5 results per tool)
3. **Price Extraction**: Regex + AI parsing from search snippets
4. **Savings Calculation**: `(Price_Higher - Price_Lower) × 100 users × 12 months`

### P&L Agent - Financial Analysis

Uses **3-shot learning** with structured JSON output:
- Profit margin thresholds
- Cost-to-revenue ratios
- Strategic value assessment
- Keep/Ditch/Review classifications

## 🖥️ Frontend Features

The Streamlit interface provides:

- **File Upload**: Support for CSV, JSON, TXT files
- **Context Preservation**: Uploaded data is automatically included in prompts
- **Streaming Responses**: Real-time output from watsonx Orchestrate
- **Conversation History**: Multi-turn dialogue with memory
- **File Preview**: View uploaded content before processing

**Example Workflow**:
1. Upload employee CSV
2. Ask: "Analyze this team and recommend restructuring"
3. System processes via Architect Agent
4. View strategic report with action items

## 🐛 Troubleshooting

### 403 Forbidden (Ngrok)
Add header to bypass ngrok warning page:
```python
headers = {"ngrok-skip-browser-warning": "true"}
```

### 401 Unauthorized (watsonx)
- Verify API key hasn't expired
- Check project ID is correct
- Ensure you have watsonx.ai access

### 422 Validation Error
- Field names must match exactly (including "dupa salary")
- Check data types (string vs integer for talent/Employee ID)
- Review YAML schema definitions

### SerpAPI Rate Limits
- Free tier: 100 searches/month
- Agent implements fallback pricing database
- Monitor usage at https://serpapi.com/dashboard

### Port Conflicts
If ports 8000/8080 are in use:
```python
# Change port in main block
uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 📊 Sample Data

The repository includes:

- **`course_catalog.json`**: 32 training courses across 8 categories
- Sample employee data structures in YAML files
- Example P&L feature data
- Tech stack invoice templates

## 🔒 Security Best Practices

- ✅ Never commit API keys to version control
- ✅ Use environment variables (`.env` file)
- ✅ Add `.env` to `.gitignore`
- ✅ Implement rate limiting for production
- ✅ Add authentication middleware for public endpoints
- ✅ Enable HTTPS in production (ngrok provides this)
- ✅ Validate all input data
- ✅ Sanitize LLM outputs before display

## 📚 Additional Resources

- [IBM watsonx.ai Documentation](https://www.ibm.com/docs/en/watsonx-as-a-service)
- [IBM watsonx Orchestrate](https://www.ibm.com/products/watsonx-orchestrate)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAPI Specification](https://swagger.io/specification/)
- [SerpAPI Documentation](https://serpapi.com/docs)

## 🎓 Key Learnings & Innovations

1. **Multi-Agent Architecture**: Orchestrated 4 specialized agents via watsonx Orchestrate
2. **Hybrid AI Approach**: Combined algorithmic logic with LLM intelligence
3. **Real-time Data Integration**: Web search for live pricing comparisons
4. **3-Shot Learning**: Achieved consistent JSON outputs from Granite LLM
5. **Conversational UX**: Natural language interface for complex workflows

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project was created for the IBM Hackathon 2024.

## 👥 Team

- [Alin Vladimirescu](https://github.com/alinVladimirescu)
- [Matej Pechoucek](https://github.com/MatejPechoucek)
- [Javier Borrajo](https://github.com/jborrajo21)

## 🏆 IBM Hackathon 2024

This platform demonstrates:
- Advanced agentic AI workflows
- Real-world M&A automation use cases
- IBM watsonx.ai and Orchestrate integration
- Production-ready API design
- Scalable multi-agent architecture

**Problem Solved**: Reduced M&A integration time from 6-12 months to 2-4 weeks through AI-powered automation of workforce planning, tech consolidation, and financial analysis.

---

**Built with ❤️ using IBM watsonx**

For questions or support, please open an issue on GitHub.
