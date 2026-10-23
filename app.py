
import streamlit as st
import fitz
import cv2
import numpy as np
import hashlib
from PIL import Image
from io import BytesIO
from datetime import datetime, timezone

st.set_page_config(
    page_title="ExamineDoc.ai",
    page_icon="🔍",
    layout="wide"
)

st.title("ExamineDoc.ai")
st.subheader("Clarity Through Examination.")
st.write("Scanned document examination and reporting.")

st.warning(
    "Public development demo. Use fictional or "
    "fully redacted test documents only."
)

uploaded = st.file_uploader(
    "Upload a test PDF",
    type=["pdf"]
)

if uploaded is not None:
    data = uploaded.getvalue()

    if len(data) > 20 * 1024 * 1024:
        st.error("Maximum PDF size is 20 MB.")
        st.stop()

    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                st.error("Password-protected PDFs are unsupported.")
                st.stop()

            if len(pdf) == 0:
                st.error("PDF has no pages.")
                st.stop()

            document_hash = hashlib.sha256(
                data
            ).hexdigest()

            st.success("Document loaded.")
            st.write("Total pages:", len(pdf))
            st.caption("SHA-256: " + document_hash)

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

            minimum_percent = st.slider(
                "Minimum region width (% of page)",
                5, 60, 15, 5
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

            height, width = gray.shape
            regions = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(
                    contour
                )

                if (
                    w >= width * minimum_percent / 100
                    and 5 <= h <= 100
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
                caption=f"Page {page_number}",
                use_container_width=True
            )

            measurements = [
                {
                    "Region": index,
                    "Left": x,
                    "Top": y,
                    "Width": w,
                    "Height": h
                }
                for index, (x, y, w, h)
                in enumerate(regions, start=1)
            ]

            st.subheader("Region Measurements")

            if measurements:
                st.dataframe(
                    measurements,
                    hide_index=True
                )
            else:
                st.info(
                    "No regions matched the settings."
                )

            st.divider()
            st.subheader("Download Examination Report")

            report = fitz.open()

            try:
                report_page = report.new_page(
                    width=595,
                    height=842
                )

                lines = [
                    "ExamineDoc.ai - Examination Report",
                    "Clarity Through Examination",
                    "",
                    "Generated (UTC): " +
                    datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Filename: " + uploaded.name,
                    "Pages in source: " + str(len(pdf)),
                    "Page examined: " + str(page_number),
                    "SHA-256:",
                    document_hash,
                    "",
                    "Candidate regions: " + str(len(regions)),
                    "Detection threshold: " +
                    str(minimum_percent) + "% page width",
                    "",
                    "Region measurements (pixels):"
                ]

                for item in measurements[:12]:
                    lines.append(
                        "Region {Region}: x={Left}, y={Top}, "
                        "w={Width}, h={Height}".format(
                            **item
                        )
                    )

                if len(measurements) > 12:
                    lines.append(
                        "Additional regions omitted "
                        "from this summary."
                    )

                lines.extend([
                    "",
                    "LIMITATIONS:",
                    "Regions indicate detected image shapes.",
                    "They do not establish alteration, fraud,",
                    "forgery, or document authenticity."
                ])

                text = "\n".join(lines)

                report_page.insert_textbox(
                    fitz.Rect(40, 40, 555, 570),
                    text,
                    fontsize=10,
                    fontname="helv",
                    lineheight=1.3
                )

                png_buffer = BytesIO()
                Image.fromarray(annotated).save(
                    png_buffer,
                    format="PNG"
                )

                report_page.insert_image(
                    fitz.Rect(45, 585, 550, 800),
                    stream=png_buffer.getvalue(),
                    keep_proportion=True
                )

                report_bytes = report.tobytes()

            finally:
                report.close()

            st.download_button(
                "Download PDF Examination Report",
                data=report_bytes,
                file_name="ExamineDoc_Examination_Report.pdf",
                mime="application/pdf"
            )

            st.caption(
                "The report contains preliminary image "
                "measurements, not a forensic conclusion. "
                "Only the first 12 regions are listed "
                "in the summary."
            )

    except Exception:
        st.error(
            "The PDF could not be processed. "
            "Use a valid test document."
        )

st.divider()
st.caption(
    "ExamineDoc.ai is experimental software. "
    "It does not determine authenticity or "
    "provide legal advice."
)
