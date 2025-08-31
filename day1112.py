from fastapi import FastAPI, Query
import requests
import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/m2m100_418M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

class Props(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class EmailDraftRequest(BaseModel):
    note: str
    tone:str
    recipient: str

class SummaryRequest(BaseModel):
    text: str

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.get("/")
def read_root():
    return {"message": "🚀 FastAPI is working!"}

@app.post("/email-draft")
def emaildrafter(text: EmailDraftRequest):

   
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    json_prompt = f"""
   you are a helpful assistant who drafts emails in {text.tone} tone , which are being recieved by {text.recipient} out of the given note  . the note is {text.note}
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

MODEL_ID = "sshleifer/distilbart-cnn-12-6"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

@app.post("/summarize")
def summarize(text: SummaryRequest):
    payload = {"inputs": text.text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        try:
            content = response.json()
            return {"reponse":content}
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": content}
    else:
        return {"error": f"API error {response.status_code}", "raw": response.text}

@app.post("/translate")
def translate(props: Props):
    tokenizer.src_lang = props.src_lang
    encoded = tokenizer(props.text, return_tensors="pt").to(model.device)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(props.tgt_lang)
    )

    return {"Translated":tokenizer.decode(generated_tokens[0], skip_special_tokens=True)}
