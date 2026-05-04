from __future__ import annotations

import streamlit as st
from langchain_community.chat_models import ChatOllama

from banglamed_rag.config import settings
from banglamed_rag.rag_chain import BanglaMedRAG

st.set_page_config(page_title="বাংলা মেড-রাগ", page_icon="🩺", layout="wide")

st.title("বাংলা মেড-রাগ")
st.write("বাংলা ভাষায় স্বাস্থ্য সংক্রান্ত প্রশ্নের জন্য স্থানীয় রাগ সিস্টেম।")

sample_questions = [
    "ডেঙ্গুর লক্ষণ কী কী?",
    "ডায়াবেটিস রোগীরা কী খাবেন?",
    "গর্ভাবস্থায় আয়রন কতটা জরুরি?",
    "জ্বরে কখন ডাক্তার দেখানো উচিত?",
    "শিশুর ডিহাইড্রেশন হলে কী করবেন?",
]

with st.sidebar:
    st.header("সেটিংস")
    model_name = st.selectbox("মডেল", ["phi3", "llama3.2:3b"], index=0)
    temperature = st.slider("তাপমাত্রা", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
    sample = st.selectbox("নমুনা প্রশ্ন", [""] + sample_questions)

question = st.text_input("আপনার প্রশ্ন লিখুন", value=sample)


@st.cache_resource
def load_rag(model: str, temp: float) -> BanglaMedRAG:
    llm = ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=temp)
    return BanglaMedRAG(llm=llm)


if st.button("অনুসন্ধান করুন"):
    if not question.strip():
        st.warning("একটি প্রশ্ন লিখুন।")
    else:
        with st.spinner("উত্তর তৈরি হচ্ছে..."):
            rag = load_rag(model_name, temperature)
            result = rag.ask(question, top_k=top_k)
            st.subheader("উত্তর")
            st.write(result["answer"])
            st.subheader("সূত্র")
            for source in result["sources"]:
                with st.expander(source.get("title") or "সূত্র"):
                    st.write(source.get("snippet", ""))
