
import streamlit as st

st.set_page_config(
    page_title="ExamineDoc.ai",
    page_icon="🔍",
    layout="wide"
)

st.title("ExamineDoc.ai")
st.subheader("Clarity Through Examination.")

st.write(
    "AI-assisted document examination, "
    "comparison, and reporting."
)

st.divider()

st.header("Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose a PDF or image to examine",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.success(
        f"Document uploaded: {uploaded_file.name}"
    )
    st.info(
        "Document examination features are coming next."
    )

st.divider()
st.caption(
    "ExamineDoc.ai identifies potential inconsistencies. "
    "It does not determine document authenticity "
    "or provide legal advice."
)
