from fastapi import FastAPI, Query
import requests
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.get("/")
def read_root():
    return {"message": "🚀 FastAPI is working!"}



@app.post("/summarise/{text}")
def summariser(text: str, style: str = Query("default", description="Summary style")):

   
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    json_prompt = f"""
   you are a helpful assistant who outputs the summary of the given text in the following style {style} . the query is {text}
    """

    payload = {
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": [{"role": "user", "content": json_prompt}],
        "temperature": 0.7
    }

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            content = response.json()["choices"][0]["message"]["content"]
            return {"reponse":content}
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": content}
    else:
        return {"error": f"API error {response.status_code}", "raw": response.text}



