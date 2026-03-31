import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

# 1. إعدادات الصفحة والواجهة
st.set_page_config(page_title="مساعد شركة الكهرباء", page_icon="⚡")
st.title("🤖 مساعد معايير شركة الكهرباء الذكي")
st.info("هذا المساعد يقرأ مباشرة من ملف المعايير المرفوع في النظام.")

# 2. جلب المفتاح السري (Token) من Streamlit Secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except:
    st.error("لم يتم العثور على مفتاح HF_TOKEN في إعدادات Secrets!")
    st.stop()

# 3. تحديد اسم الملف المرفوع في GitHub
# ملاحظة: تأكد أن الاسم هنا يطابق تماماً اسم الملف الذي رفعته في GitHub
FILE_NAME = "standards.pdf" 

if os.path.exists(FILE_NAME):
    # استخدام "الذاكرة المؤقتة" لكي لا يعيد تحميل الملف مع كل سؤال
    @st.cache_resource
    def load_data():
        loader = PyPDFLoader(FILE_NAME)
        pages = loader.load_and_split()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(pages, embeddings)

    vectorstore = load_data()
    st.success("✅ تم تحميل المعايير بنجاح. أنا جاهز لأسئلتك!")

    # 4. منطقة الدردشة
    query = st.text_input("اسأل عن أي تفصيل فني في المعايير:")

    if query:
        with st.spinner("جاري البحث في المعايير..."):
            # البحث عن المعلومات ذات الصلة
            docs = vectorstore.similarity_search(query, k=3)
            context = " ".join([d.page_content for d in docs])
            
            # إرسال السؤال للذكاء الاصطناعي (Llama 3)
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
                token=hf_token,
                temperature=0.1 # لضمان إجابة دقيقة وغير خيالية
            )
            
            prompt = f"بناءً على نص المعايير التالي: {context}\n\nالسؤال: {query}\nالإجابة بالعربية وبدقة تقنية:"
            response = llm.invoke(prompt)
            
            st.markdown("### الإجابة:")
            st.write(response)
            
            with st.expander("شاهد المراجع المستند عليها من الملف:"):
                for i, doc in enumerate(docs):
                    st.write(f"**مرجع {i+1}:** {doc.page_content[:300]}...")
else:
    st.error(f"لم يتم العثور على ملف باسم `{FILE_NAME}` في المستودع. يرجى رفعه أو التأكد من الاسم.")
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
