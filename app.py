
import streamlit as st
import fitz
import hashlib
from collections import Counter
from io import BytesIO
from PIL import Image, ImageDraw
from statistics import median

st.set_page_config(
    page_title="ExamineDoc.ai",
    page_icon="🔍",
    layout="wide"
)

st.title("ExamineDoc.ai")
st.subheader("Clarity Through Examination.")
st.write("Visual document inspection and review.")

st.warning(
    "Experimental public demo. "
    "Do not upload personal or confidential documents."
)

uploaded = st.file_uploader(
    "Upload a test PDF",
    type=["pdf"]
)

if uploaded:
    data = uploaded.getvalue()

    if len(data) > 20 * 1024 * 1024:
        st.error("Maximum file size is 20 MB.")
        st.stop()

    try:
        pdf = fitz.open(stream=data, filetype="pdf")

        if pdf.needs_pass:
            st.error("Password-protected PDFs are unsupported.")
            pdf.close()
            st.stop()

        if len(pdf) == 0:
            st.error("PDF has no pages.")
            pdf.close()
            st.stop()

        st.success("Document loaded.")
        st.write("Pages:", len(pdf))
        st.caption(
            "SHA-256: " + hashlib.sha256(data).hexdigest()
        )

        number = st.selectbox(
            "Select page",
            range(1, len(pdf) + 1)
        )
        page = pdf[number - 1]

        spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        spans.append(span)

        fonts = Counter(s["font"] for s in spans)

        st.subheader("Font Information")
        st.dataframe(
            [
                {"Font": font, "Text spans": count}
                for font, count in fonts.items()
            ],
            hide_index=True
        )

        ocr_detected = any(
            "GlyphLessFont" in s["font"] for s in spans
        )

        if ocr_detected:
            st.info(
                "An OCR-style text layer was detected. "
                "Text geometry may not match the original "
                "printed font or spacing."
            )

        st.subheader("Position and Size Review")

        sizes = [
            s["size"] for s in spans if s["size"] > 0
        ]

        typical_size = median(sizes) if sizes else None

        flagged = []

        if typical_size and not ocr_detected:
            for s in spans:
                ratio = s["size"] / typical_size

                if ratio < 0.7 or ratio > 1.5:
                    flagged.append({
                        "bbox": s["bbox"],
                        "reason": "Unusual text size",
                        "text": s["text"]
                    })

        if ocr_detected:
            st.warning(
                "Automatic font-size flags are disabled "
                "because OCR text dimensions may not "
                "represent the scanned document."
            )

        st.write(
            "Review candidates found:",
            len(flagged)
        )

        zoom = 1.5
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            alpha=False
        )

        image = Image.open(
            BytesIO(pix.tobytes("png"))
        ).convert("RGB")

        draw = ImageDraw.Draw(image)

        for item in flagged:
            box = item["bbox"]
            draw.rectangle(
                (
                    box[0] * zoom,
                    box[1] * zoom,
                    box[2] * zoom,
                    box[3] * zoom
                ),
                outline="red",
                width=3
            )

        st.subheader("Page Preview")
        st.image(
            image,
            use_container_width=True
        )

        if flagged:
            st.subheader("Review Candidates")
            st.dataframe(
                [
                    {
                        "Text": x["text"],
                        "Reason": x["reason"]
                    }
                    for x in flagged
                ],
                hide_index=True
            )
        else:
            st.info(
                "No candidates identified by the "
                "current text-size rule. This does not "
                "mean the document is authentic or free "
                "of irregularities."
            )

        pdf.close()

    except Exception:
        st.error(
            "This PDF could not be processed. "
            "Use a valid test document."
        )

st.divider()
st.caption(
    "ExamineDoc.ai flags preliminary review candidates. "
    "It cannot establish fraud, forgery, or authenticity."
)
