
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
st.write("Document examination and evidence reporting.")

st.warning(
    "Public development demo. Upload only fictional "
    "or fully redacted test documents."
)

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 20


def safe_text(value):
    return str(value).encode(
        "ascii", errors="replace"
    ).decode("ascii")


def analyze_page(page, min_percent):
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
        image, cv2.COLOR_RGB2GRAY
    )

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (25, 2)
    )

    connected = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = gray.shape
    regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if (
            w >= width * min_percent / 100
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

    return annotated, regions


def write_lines(page, lines, top=50):
    y = top

    for line in lines:
        if y > 795:
            break

        page.insert_text(
            (45, y),
            safe_text(line)[:95],
            fontsize=10,
            fontname="helv"
        )
        y += 18


def build_report(data, filename, min_percent):
    source_hash = hashlib.sha256(data).hexdigest()
    report = fitz.open()

    with fitz.open(
        stream=data, filetype="pdf"
    ) as source:

        if source.needs_pass:
            raise ValueError("Password-protected PDF")

        if not 1 <= len(source) <= MAX_PAGES:
            raise ValueError("Unsupported page count")

        # Cover page
        cover = report.new_page(
            width=595, height=842
        )

        cover.insert_text(
            (45, 65),
            "ExamineDoc.ai",
            fontsize=25,
            fontname="hebo"
        )

        cover.insert_text(
            (45, 95),
            "Clarity Through Examination",
            fontsize=13
        )

        cover.insert_text(
            (45, 150),
            "DOCUMENT EXAMINATION REPORT",
            fontsize=16,
            fontname="hebo"
        )

        created = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")

        cover_lines = [
            "",
            "Report type: Automated image-region review",
            "Generated: " + created,
            "Filename: " + filename[:65],
            "Source pages: " + str(len(source)),
            "Rendering scale: 2x",
            "Minimum region width: " +
            str(min_percent) + "%",
            "",
            "Source SHA-256:",
            source_hash[:32],
            source_hash[32:],
            "",
            "METHOD:",
            "Grayscale conversion and Otsu thresholding.",
            "Horizontal morphological closing.",
            "Contour-based image region detection.",
            "",
            "IMPORTANT LIMITATIONS:",
            "Detected regions are not confirmed anomalies.",
            "Scanning artifacts can affect results.",
            "This report does not establish document",
            "authenticity, forgery, alteration or fraud.",
            "It is not a certified forensic opinion."
        ]

        write_lines(cover, cover_lines, top=185)

        total_regions = 0

        for page_index, source_page in enumerate(
            source, start=1
        ):
            annotated, regions = analyze_page(
                source_page, min_percent
            )

            total_regions += len(regions)

            # Full-page image
            image_page = report.new_page(
                width=595, height=842
            )

            image_page.insert_text(
                (40, 35),
                f"PAGE {page_index} - ANNOTATED IMAGE",
                fontsize=14,
                fontname="hebo"
            )

            buffer = BytesIO()
            Image.fromarray(annotated).save(
                buffer, format="PNG"
            )

            image_page.insert_image(
                fitz.Rect(35, 55, 560, 780),
                stream=buffer.getvalue(),
                keep_proportion=True
            )

            image_page.insert_text(
                (40, 815),
                f"Detected regions: {len(regions)}",
                fontsize=9
            )

            # Measurement pages
            rows_per_page = 30

            if regions:
                for start in range(
                    0, len(regions), rows_per_page
                ):
                    measurement_page = report.new_page(
                        width=595, height=842
                    )

                    measurement_page.insert_text(
                        (40, 45),
                        f"PAGE {page_index} - REGION MEASUREMENTS",
                        fontsize=13,
                        fontname="hebo"
                    )

                    lines = [
                        "Coordinates are rendered-image pixels.",
                        "Scale: 2x; positions measured from top-left.",
                        "",
                        "ID       Left       Top       Width       Height"
                    ]

                    for index in range(
                        start,
                        min(start + rows_per_page, len(regions))
                    ):
                        x, y, w, h = regions[index]

                        lines.append(
                            f"{index + 1:<8} "
                            f"{x:<10} {y:<10} "
                            f"{w:<11} {h}"
                        )

                    lines.extend([
                        "",
                        "Measurements describe image regions.",
                        "They do not establish document alteration."
                    ])

                    write_lines(
                        measurement_page, lines, top=78
                    )

            else:
                measurement_page = report.new_page(
                    width=595, height=842
                )
                write_lines(
                    measurement_page,
                    [
                        f"Page {page_index}: Region measurements",
                        "",
                        "No regions met the detection threshold.",
                        "This does not establish authenticity."
                    ]
                )

        # Summary appended at end
        summary = report.new_page(
            width=595, height=842
        )

        write_lines(
            summary,
            [
                "EXAMINATION SUMMARY",
                "",
                f"Pages examined: {len(source)}",
                f"Total detected regions: {total_regions}",
                "",
                "This document records automated",
                "image-region measurements.",
                "",
                "Human review is required to interpret",
                "the significance of any observation.",
                "",
                "Original source PDF not modified.",
                "Source fingerprint recorded on cover.",
                "",
                "End of report."
            ]
        )

    try:
        return report.tobytes(
            garbage=4, deflate=True
        )
    finally:
        report.close()


uploaded = st.file_uploader(
    "Upload a test PDF",
    type=["pdf"]
)

if uploaded is not None:
    data = uploaded.getvalue()

    if len(data) > MAX_BYTES:
        st.error("Maximum file size is 20 MB.")
        st.stop()

    min_percent = st.slider(
        "Minimum region width (% of page)",
        5, 60, 15, 5
    )

    try:
        with fitz.open(
            stream=data, filetype="pdf"
        ) as pdf:

            if pdf.needs_pass:
                st.error("Password-protected PDFs are unsupported.")
                st.stop()

            if not 1 <= len(pdf) <= MAX_PAGES:
                st.error("Use a PDF containing 1 to 20 pages.")
                st.stop()

            st.success("Document loaded successfully.")

            st.metric("Pages", len(pdf))

            st.caption(
                "SHA-256: " +
                hashlib.sha256(data).hexdigest()
            )

            number = st.selectbox(
                "Select page to preview",
                range(1, len(pdf) + 1)
            )

            annotated, regions = analyze_page(
                pdf[number - 1], min_percent
            )

            st.metric(
                "Detected regions on selected page",
                len(regions)
            )

            st.subheader("Visual Examination")

            st.image(
                annotated,
                use_container_width=True
            )

            st.subheader("Region Measurements")

            rows = [
                {
                    "Region": i,
                    "Left (px)": x,
                    "Top (px)": y,
                    "Width (px)": w,
                    "Height (px)": h
                }
                for i, (x, y, w, h) in enumerate(
                    regions, start=1
                )
            ]

            if rows:
                st.dataframe(rows, hide_index=True)
            else:
                st.info(
                    "No regions met the detection threshold."
                )

        st.divider()
        st.subheader("Professional PDF Report")

        st.write(
            "Generate a report covering every page, "
            "with full-page annotated images and "
            "complete region measurements."
        )

        if st.button("Generate Examination Report"):
            with st.spinner("Examining document pages..."):
                report_bytes = build_report(
                    data, uploaded.name, min_percent
                )

            st.download_button(
                "Download Complete PDF Report",
                data=report_bytes,
                file_name="ExamineDoc_Full_Report.pdf",
                mime="application/pdf"
            )

    except Exception:
        st.error(
            "Unable to process the PDF. "
            "Check that it is a valid test document."
        )

st.divider()

st.caption(
    "ExamineDoc.ai is an experimental document "
    "review tool. Detected regions do not establish "
    "fraud, forgery or authenticity."
)
