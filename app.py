
import streamlit as st
import fitz
import cv2
import numpy as np
import hashlib
from PIL import Image

st.set_page_config(
    page_title="ExamineDoc.ai",
    page_icon="🔍",
    layout="wide"
)

st.title("ExamineDoc.ai")
st.subheader("Clarity Through Examination.")
st.write("Scanned document visual examination.")

st.warning(
    "Public development demo. Use fictional or "
    "redacted documents only."
)

uploaded = st.file_uploader(
    "Upload a test PDF",
    type=["pdf"]
)

if uploaded is not None:
    data = uploaded.getvalue()

    if len(data) > 20 * 1024 * 1024:
        st.error("Maximum file size is 20 MB.")
        st.stop()

    try:
        with fitz.open(
            stream=data,
            filetype="pdf"
        ) as pdf:

            if pdf.needs_pass:
                st.error("Password-protected PDFs are unsupported.")
                st.stop()

            if len(pdf) == 0:
                st.error("PDF contains no pages.")
                st.stop()

            st.success("Document loaded.")

            st.write("Total pages:", len(pdf))
            st.caption(
                "SHA-256: " +
                hashlib.sha256(data).hexdigest()
            )

            page_number = st.selectbox(
                "Select page",
                range(1, len(pdf) + 1)
            )

            page = pdf[page_number - 1]

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                colorspace=fitz.csRGB,
                alpha=False
            )

            image = np.array(
                Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples
                )
            )

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY_INV +
                cv2.THRESH_OTSU
            )

            st.subheader("Scanned Image Analysis")

            min_width_pct = st.slider(
                "Minimum text-region width (% of page)",
                min_value=5,
                max_value=60,
                value=15,
                step=5
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (25, 2)
            )

            connected = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel
            )

            contours, _ = cv2.findContours(
                connected,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            page_height, page_width = gray.shape

            minimum_width = (
                page_width * min_width_pct / 100
            )

            regions = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if (
                    w >= minimum_width
                    and h >= 5
                    and h <= 100
                ):
                    regions.append((x, y, w, h))

            regions.sort(key=lambda r: (r[1], r[0]))

            annotated = image.copy()

            for x, y, w, h in regions:
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

            st.metric(
                "Detected candidate regions",
                len(regions)
            )

            st.subheader("Visual Examination")

            st.image(
                annotated,
                caption=(
                    f"Page {page_number}: "
                    "Candidate text regions in red"
                ),
                use_container_width=True
            )

            st.subheader("Region Measurements")

            if regions:
                results = []

                for index, (x, y, w, h) in enumerate(
                    regions, start=1
                ):
                    results.append({
                        "Region": index,
                        "Left (px)": x,
                        "Top (px)": y,
                        "Width (px)": w,
                        "Height (px)": h
                    })

                st.dataframe(
                    results,
                    hide_index=True
                )

            else:
                st.info(
                    "No regions matched the current "
                    "detection settings."
                )

            st.info(
                "Red rectangles indicate image regions "
                "identified by the detection algorithm. "
                "They are not evidence of document alteration."
            )

            st.caption(
                "Measurements use pixels at the current "
                "rendering resolution. Regions may include "
                "headings, marks, redactions or joined text."
            )

    except Exception:
        st.error(
            "Unable to process this PDF. "
            "Check that the document is valid."
        )

st.divider()

st.caption(
    "ExamineDoc.ai is an experimental document "
    "review tool. It does not establish authenticity, "
    "forgery, fraud or legal conclusions."
)
