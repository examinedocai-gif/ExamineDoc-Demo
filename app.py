
import streamlit as st
import fitz
from collections import Counter

st.set_page_config(
    page_title="ExamineDoc.ai",
    page_icon="🔍",
    layout="wide"
)

st.title("ExamineDoc.ai")
st.subheader("Clarity Through Examination.")
st.write("AI-assisted document examination and reporting.")

st.divider()
st.header("Document Examination")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Document uploaded: {uploaded_file.name}")

    try:
        pdf = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        if pdf.needs_pass:
            st.warning("This PDF is password protected.")
        else:
            st.subheader("Examination Results")
            st.metric("Total Pages", len(pdf))

            all_fonts = []
            findings = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                font_names = []
                font_sizes = []

                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        for span in line["spans"]:
                            if not span["text"].strip():
                                continue

                            font_names.append(span["font"])
                            font_sizes.append(
                                round(span["size"], 1)
                            )

                all_fonts.extend(font_names)

                with st.expander(f"Page {page_number}"):
                    st.write("Fonts found:", sorted(set(font_names)))
                    st.write(
                        "Font sizes:",
                        sorted(set(font_sizes))
                    )

                    st.text_area(
                        "Extracted Text",
                        text,
                        height=180,
                        key=f"page_{page_number}"
                    )

                if len(set(font_names)) > 1:
                    findings.append(
                        f"Page {page_number}: "
                        "Multiple font styles detected."
                    )

            st.divider()
            st.subheader("Document Font Summary")

            if all_fonts:
                font_counts = Counter(all_fonts)
                st.dataframe(
                    [
                        {"Font": name, "Text spans": count}
                        for name, count in font_counts.items()
                    ],
                    hide_index=True
                )
            else:
                st.info(
                    "No embedded text fonts detected. "
                    "This may be a scanned document."
                )

            st.divider()
            st.subheader("Preliminary Findings")

            if findings:
                for finding in findings:
                    st.warning(finding)
            else:
                st.success(
                    "No multiple-font flags detected."
                )

            st.caption(
                "Font differences are common in legitimate "
                "documents and do not establish alterations. "
                "Scanned PDFs require additional image analysis. "
                "This tool does not determine authenticity "
                "or provide legal advice."
            )

        pdf.close()

    except Exception as error:
        st.error(
            "The document could not be examined. "
            "Please check that it is a valid PDF."
        )
