from fastapi import FastAPI, Query
import requests
import os
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"


chat_history = []

@app.get("/")
def read_root():
    return {"message": "🚀 FastAPI chatbot running with memory!"}


@app.post("/chat/")
def chat_with_bot(
    user_message: str,
    style: str = Query("default", description="Response style (e.g., formal, casual, funny)")
):


    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    chat_history.append({"role": "user", "content": user_message})

    system_prompt = {
        "role": "system",
        "content": f"You are a helpful chatbot. Always reply in this style: {style}."
    }
    messages = [system_prompt] + chat_history

    payload = {
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            content = response.json()["choices"][0]["message"]["content"]

            chat_history.append({"role": "assistant", "content": content})

            return {"response": content, "history": chat_history}
        except Exception as e:
            return {"error": f"Failed to parse response: {str(e)}", "raw": response.json()}
    else:
        return {"error": f"API error {response.status_code}", "raw": response.text}
