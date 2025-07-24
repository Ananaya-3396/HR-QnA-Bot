import streamlit as st
import torch
import os
import pandas as pd
from rapidfuzz import process, fuzz

# Import transformers differently
import transformers
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from peft.peft_model import PeftModel

# Set device to CPU (since model was trained on CPU)
device = torch.device('cpu')

# Paths
BASE_MODEL = "google/gemma-2b-it"
ADAPTER_DIR = "gemma_finetuned/checkpoint-205"

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, torch_dtype=torch.float32).to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Load HR Q&A data
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

# Streamlit UI
st.title("HR Q&A Assistant")
st.write("Ask me anything about my professional experience, and my bot will answer just like I would.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Enter your question please:")

if st.button("Get Answer") and user_input:
    # Try to find a close match in the HR Q&A data
    answer, matched_q, score = get_best_match(user_input, questions, answers)
    if answer:
        st.session_state['chat_history'].append((user_input, answer))
    else:
        st.session_state['chat_history'].append((user_input, "Response cannot be generated. Please try with a different question."))

# Display the latest Q&A in the main area
if st.session_state['chat_history']:
    latest_q, latest_a = st.session_state['chat_history'][-1]
    # st.subheader("Latest Answer")
    # st.markdown(f"**You:** {latest_q}")
    st.markdown(f"**Answer:** {latest_a}")

# Display chat history
if st.session_state['chat_history']:
    st.sidebar.subheader("Chat History")
    for q, a in reversed(st.session_state['chat_history']):
        st.sidebar.markdown(f"**Question:** {q}")
        st.sidebar.markdown(f"**Answer:** {a}")