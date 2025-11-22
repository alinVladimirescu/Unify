# Watsonx Orchestrate Code Block
# ---------------------------------------------------------
# INPUT:  employee_data_str (String/Text from previous node)
# OUTPUT: result_str (String/Text for next node)
# ---------------------------------------------------------

# 1. CONFIGURATION
# We use standard libraries (json) which are pre-loaded in the environment.
# We avoid 'import', 'class', 'type()', and 'str.format()' as per constraints.

WEIGHT_PERFORMANCE = 0.5
WEIGHT_TALENT = 0.3
WEIGHT_EDUCATION = 0.2

education_map = {
    "PhD": 5, "Master": 4, "Bachelor": 3, "High School": 1,
    "Doctorate": 5, "Masters": 4, "Bachelors": 3
}
talent_map = {"High": 5, "Medium": 3, "Low": 1}

# Initialize output safe default
result_str = ""

# 2. PROCESSING LOGIC
try:
    # Parse JSON string input
    # 'employee_data_str' is provided by the upstream workflow node
    if not employee_data_str:
        employees = []
    else:
        employees = json.loads(employee_data_str)

    processed_staff = []

    # 3. CALCULATIONS
    for emp in employees:
        try:
            # Safe extraction with defaults
            rating_raw = emp.get('rating', 0)
            salary_raw = emp.get('dupa salary', 0)
            
            # Force conversion to float
            rating = float(rating_raw)
            salary = float(salary_raw)
            
            talent_txt = emp.get('talent', 'Low')
            edu_txt = emp.get('studii', 'Bachelor')
            
            # Map text to score
            talent_score = talent_map.get(talent_txt, 1)
            edu_score = education_map.get(edu_txt, 1)
            
            # The Weighted Formula
            final_score = (rating * WEIGHT_PERFORMANCE) + \
                          (talent_score * WEIGHT_TALENT) + \
                          (edu_score * WEIGHT_EDUCATION)
            
            # Build processed record
            processed_staff.append({
                "name": emp.get('Name', 'Unknown'),
                "role": emp.get('jobul', 'Unknown'),
                "salary": salary,
                "final_score": round(final_score, 2),
                "raw_rating": rating
            })
        except Exception:
            # Skip malformed rows silently or handle as needed
            continue

    # 4. RANKING & ALLOCATION
    # Sort descending by score
    processed_staff.sort(key=lambda x: x['final_score'], reverse=True)
    
    HQ_CAPACITY = 20
    hq_team = processed_staff[:HQ_CAPACITY]
    remote_team = processed_staff[HQ_CAPACITY:]
    
    # 5. RISK ANALYSIS
    total_staff = len(processed_staff)
    if total_staff > 0:
        avg_score = sum(p['final_score'] for p in processed_staff) / total_staff
    else:
        avg_score = 0.0
        
    flagged_risks = []
    
    for p in processed_staff:
        # Condition: Below Average Score AND Salary > 100k
        if p['final_score'] < avg_score and p['salary'] > 100000:
            flagged_risks.append({
                "name": p['name'],
                "role": p['role'],
                "salary": p['salary'],
                "merit_score": p['final_score'],
                "discrepancy": "High Salary / Low Merit"
            })

    # 6. CONSTRUCT OUTPUT
    output_payload = {
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
    
    # Serialize back to string for the next node
    result_str = json.dumps(output_payload)

except Exception as e:
    # Error Handling
    # We construct a JSON error string manually if json fails, or use json.dumps
    error_response = {"status": "error", "message": "Processing failed in Python block."}
    result_str = json.dumps(error_response)