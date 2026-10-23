
import streamlit as st
import fitz
from collections import Counter
from io import BytesIO
from PIL import Image
import hashlib

st.set_page_config(
    page_title="ExamineDoc.ai",
    page_icon="🔍",
    layout="wide"
)

st.title("ExamineDoc.ai")
st.subheader("Clarity Through Examination.")
st.write("Visual document examination and analysis.")

st.warning(
    "Development demonstration only. "
    "Do not upload confidential or sensitive documents."
)

st.divider()
st.header("Examine a Document")

uploaded_file = st.file_uploader(
    "Upload a PDF for examination",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf:

            if pdf.needs_pass:
                st.error("Password-protected PDFs are not supported.")
            elif len(pdf) == 0:
                st.error("This PDF contains no pages.")
            else:
                st.success("Document loaded successfully.")

                st.subheader("Document Information")
                st.write("Filename:", uploaded_file.name)
                st.write("Pages:", len(pdf))
                st.caption(
                    "SHA-256 fingerprint: " + file_hash
                )

                page_number = st.selectbox(
                    "Select a page to examine",
                    range(1, len(pdf) + 1)
                )

                page = pdf[page_number - 1]

                spans = []
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span["text"].strip():
                                spans.append(span)

                font_counts = Counter(
                    span["font"] for span in spans
                )

                st.subheader("Font Inspection")

                if font_counts:
                    st.dataframe(
                        [
                            {
                                "Font": name,
                                "Text spans": count
                            }
                            for name, count
                            in font_counts.items()
                        ],
                        hide_index=True
                    )

                    selected_font = st.selectbox(
                        "Highlight text using this font",
                        ["None"] + sorted(font_counts)
                    )
                else:
                    selected_font = "None"
                    st.info(
                        "No extractable text found. "
                        "This may be a scanned PDF."
                    )

                st.subheader("Visual Page Examination")

                zoom = 1.5
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                ).convert("RGB")

                if selected_font != "None":
                    overlay_page = fitz.open()
                    try:
                        new_page = overlay_page.new_page(
                            width=page.rect.width,
                            height=page.rect.height
                        )

                        for span in spans:
                            if span["font"] == selected_font:
                                rect = fitz.Rect(span["bbox"])
                                new_page.draw_rect(
                                    rect,
                                    color=(1, 0, 0),
                                    width=1.5
                                )

                        overlay = new_page.get_pixmap(
                            matrix=fitz.Matrix(zoom, zoom),
                            alpha=True
                        )

                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(image)

                        for span in spans:
                            if span["font"] == selected_font:
                                box = span["bbox"]
                                draw.rectangle(
                                    [
                                        box[0] * zoom,
                                        box[1] * zoom,
                                        box[2] * zoom,
                                        box[3] * zoom
                                    ],
                                    outline="red",
                                    width=3
                                )
                    finally:
                        overlay_page.close()

                st.image(
                    image,
                    caption=f"Page {page_number}",
                    use_container_width=True
                )

                st.divider()
                st.subheader("Extracted Page Text")
                st.text_area(
                    "Text extracted from selected page",
                    page.get_text("text"),
                    height=220
                )

                st.info(
                    "Red rectangles identify text using "
                    "the selected font. They do not "
                    "indicate fraud or alteration."
                )

    except Exception:
        st.error(
            "Unable to examine this PDF. "
            "Confirm that it is a valid document."
        )

st.divider()
st.caption(
    "ExamineDoc.ai is an experimental examination "
    "tool, not a certified forensic examiner. "
    "Its observations are not legal conclusions."
)
