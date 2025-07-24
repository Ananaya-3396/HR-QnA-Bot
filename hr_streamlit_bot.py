import streamlit as st
import torch
import pandas as pd
from rapidfuzz import process, fuzz
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Device
device = torch.device('cpu')

# Model config
BASE_MODEL = "google/gemma-2b-it"
ADAPTER_DIR = "gemma_finetuned/checkpoint-205"

#Hardcode your HF token here (make sure it is valid)
HF_TOKEN = "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"

@st.cache_resource(show_spinner=True)
def load_model():
    if not HF_TOKEN:
        raise ValueError("Hugging Face token not found.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.float32
    ).to(device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, torch_dtype=torch.float32).to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Load Q&A CSV
@st.cache_data(show_spinner=True)
def load_hr_qa():
    df = pd.read_csv("HR questions.csv", encoding="cp1252")
    questions = df['input'].tolist()
    answers = df['output'].tolist()
    return questions, answers

questions, answers = load_hr_qa()

def get_best_match(user_q, questions, answers, threshold=70):
    match, score, idx = process.extractOne(user_q, questions, scorer=fuzz.token_set_ratio)
    if score >= threshold:
        return answers[idx], match, score
    return None, match, score

# UI
st.title("HR Q&A Assistant")
st.write("Ask me anything about my professional experience, and my bot will answer just like I would.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Enter your question please:")

if st.button("Get Answer") and user_input:
    answer, matched_q, score = get_best_match(user_input, questions, answers)
    if answer:
        st.session_state['chat_history'].append((user_input, answer))
    else:
        st.session_state['chat_history'].append((user_input, "Response cannot be generated. Please try with a different question."))

# Display current answer
if st.session_state['chat_history']:
    latest_q, latest_a = st.session_state['chat_history'][-1]
    st.markdown(f"**Answer:** {latest_a}")

# Chat history
if st.session_state['chat_history']:
    st.sidebar.subheader("Chat History")
    for q, a in reversed(st.session_state['chat_history']):
        st.sidebar.markdown(f"**Question:** {q}")
        st.sidebar.markdown(f"**Answer:** {a}")
