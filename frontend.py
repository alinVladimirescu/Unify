import streamlit as st
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
INSTANCE_URL = os.getenv("INSTANCE_URL")
SPECIFIC_AGENT_ID = os.getenv("AGENT_ID")

st.set_page_config(page_title="Unify", layout="centered")
st.title("Unify")
st.markdown("### M&A Integration Orchestrator")

with st.sidebar:
    
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Attach files (Text, CSV, JSON)", 
        type=["txt", "csv", "json"],
        accept_multiple_files=True
    )
    
    all_files_content = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_content = ""
                
                if uploaded_file.type == "application/json":
                    data = json.load(uploaded_file)
                    file_content = json.dumps(data, indent=2)
                    st.success(f"✅ {uploaded_file.name} - JSON loaded!")
                    
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    records = df.to_dict(orient='records')
                    
                    # Check if CSV contains P&L feature columns
                    pl_columns = {'feature', 'revenue', 'cost', 'net_profit'}
                    csv_columns = set(df.columns.str.lower())
                    
                    # Check if CSV contains employee-related columns
                    employee_indicators = {'employee id', 'employee_id'}
                    has_employee_field = bool(employee_indicators.intersection(csv_columns))
                    
                    if pl_columns.issubset(csv_columns):
                        formatted_data = {
                            "features": records
                        }
                        file_content = json.dumps(formatted_data, indent=2)
                        print(formatted_data)
                        st.success(f"✅ {uploaded_file.name} - P&L JSON format!")
                        
                    elif has_employee_field:
                        formatted_data = {
                            "employees": records
                        }
                        file_content = json.dumps(formatted_data, indent=2)
                        print(formatted_data)
                        st.success(f"✅ {uploaded_file.name} - Employee JSON format!")
                        
                    else:
                        print(records)
                        file_content = json.dumps(records, indent=2)
                        st.success(f"✅ {uploaded_file.name} - JSON format!")
                        
                else:
                    stringio = uploaded_file.getvalue().decode("utf-8")
                    file_content = stringio
                    st.success(f"✅ {uploaded_file.name} - Text file loaded!")
                
                all_files_content.append({
                    "filename": uploaded_file.name,
                    "content": file_content
                })
                
            except Exception as e:
                st.error(f"❌ Error reading {uploaded_file.name}: {e}")
        
        # Preview all uploaded files
        if all_files_content:
            with st.expander(f"Preview All Files ({len(all_files_content)} uploaded)"):
                for file_data in all_files_content:
                    st.write(f"**{file_data['filename']}:**")
                    preview = file_data['content']
                    st.code(preview[:300] + "..." if len(preview) > 300 else preview)
                    st.divider()


def get_iam_token(api_key):
    """Step 1: Get Bearer token from IAM"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        st.error(f"❌ Authentication failed: {e}")
        return None


def get_agent_and_thread(bearer_token):
    """Step 2: Use specific agent ID and create/get thread ID"""
    if SPECIFIC_AGENT_ID:
        agent_id = SPECIFIC_AGENT_ID
    else:
        # Fallback: Get first available agent
        agents_url = f"{INSTANCE_URL}/v1/orchestrate/agents"
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(agents_url, headers=headers, timeout=10)
            response.raise_for_status()
            agents_data = response.json()
            
            if agents_data and len(agents_data) > 0:
                agent_id = agents_data[0].get("id")
            else:
                st.error("❌ No agents found")
                return None, None
        except Exception as e:
            st.error(f"❌ Failed to get agent: {e}")
            return None, None
    
    # Get or create thread for this agent
    thread_id = get_or_create_thread(bearer_token, agent_id)
    
    return agent_id, thread_id


def get_or_create_thread(bearer_token, agent_id):
    """Get existing thread or create new one for the agent"""
    # Check if we have a thread_id in session state for this agent
    if "thread_id" in st.session_state and st.session_state.get("current_agent_id") == agent_id:
        return st.session_state.thread_id
    
    # Create a new thread
    threads_url = f"{INSTANCE_URL}/v1/orchestrate/threads"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "agent_id": agent_id
    }
    
    try:
        response = requests.post(threads_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        thread_data = response.json()
        
        thread_id = thread_data.get("id") or thread_data.get("thread_id")
        if thread_id:
            # Store in session state
            st.session_state.thread_id = thread_id
            st.session_state.current_agent_id = agent_id
            return thread_id
        else:
            return None
    except Exception as e:
        return None


def call_orchestrate_run(bearer_token, agent_id, thread_id, user_message):
    """Step 3: Call the orchestrate run endpoint"""
    url = f"{INSTANCE_URL}/v1/orchestrate/runs?stream=true&stream_timeout=120000&multiple_content=true"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "IAM-API_KEY": API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "message": {
            "role": "user",
            "content": user_message
        },
        "agent_id": agent_id
    }
    
    # Only add thread_id if we have one
    if thread_id:
        payload["thread_id"] = thread_id
    
    try:
        with st.spinner("Waiting for response..."):
            response = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
            
            if response.status_code != 200 and response.status_code != 201:
                return f"❌ Error {response.status_code}: {response.text}"
            
            # Handle streaming response - collect text from delta events
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        
                        # Only process message.delta events
                        if chunk_data.get('event') == 'message.delta':
                            data = chunk_data.get('data', {})
                            delta = data.get('delta', {})
                            content = delta.get('content', [])
                            
                            # Extract text from content array
                            if content and len(content) > 0:
                                text = content[0].get('text', '')
                                full_response += text
                                
                    except json.JSONDecodeError:
                        continue
            
            # Return the accumulated response
            if full_response:
                return full_response
            else:
                return "❌ No response content received"
            
    except Exception as e:
        return f"❌ Error calling orchestrate: {e}"

def call_watsonx_orchestrate(messages):
    """Main function that orchestrates all three steps"""
    if not API_KEY or not INSTANCE_URL:
        return "Error: Configuration missing. Please check API Key and Instance URL."

    last_user_message = messages[-1]["content"]
    
    # Step 1: Get Bearer token
    bearer_token = get_iam_token(API_KEY)
    if not bearer_token:
        return "Error: Could not generate bearer token."
    
    # Step 2: Get Agent ID and Thread ID
    agent_id, thread_id = get_agent_and_thread(bearer_token)
    if not agent_id:
        return "Error: Could not retrieve agent ID."
    
    # Step 3: Call the orchestrate run
    response = call_orchestrate_run(bearer_token, agent_id, thread_id, last_user_message)
    return response


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload files or ask me a question."}
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "current_agent_id" not in st.session_state:
    st.session_state.current_agent_id = None

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    
    final_prompt = prompt
    
    # Combine all uploaded files into the prompt (content only, no filenames)
    if all_files_content:
        files_text = "\n\n".join([
            file_data['content']
            for file_data in all_files_content
        ])
        final_prompt = f"Context from uploaded files:\n\n{files_text}\n\nUser Question:\n{prompt}"
        st.toast(f"{len(all_files_content)} file(s) added to message!", icon="📎")

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    api_messages = st.session_state.messages.copy()
    api_messages[-1]["content"] = final_prompt

    response_text = call_watsonx_orchestrate(api_messages)
    
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})