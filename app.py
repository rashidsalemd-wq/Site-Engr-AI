import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

# إعداد الواجهة
st.set_page_config(page_title="مساعد مهندس الموقع", page_icon="⚡")
st.title("🤖 مساعد معايير شركة الكهرباء")

# الحصول على التوكن بشكل آمن
hf_token = st.secrets["HF_TOKEN"]

# رفع الملف
uploaded_file = st.file_uploader("ارفع ملف معايير الكهرباء (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("جاري قراءة المعايير..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        
        # تحويل النصوص لبيانات رقمية للبحث
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(pages, embeddings)
        
    query = st.text_input("اسأل عن أي معيار فني (مثلاً: مسافة الحفر، أنواع الكوابل):")
    
    if query:
        # البحث في الملف
        docs = vectorstore.similarity_search(query, k=2)
        context = " ".join([d.page_content for d in docs])
        
        # استشارة الذكاء الاصطناعي
        llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)
        prompt = f"بناءً على هذا النص من المعايير: {context}\n\nالسؤال: {query}\nالإجابة بالعربية وبدقة:"
        
        response = llm.invoke(prompt)
        st.success("الإجابة وفقاً للمعايير:")
        st.write(response)
