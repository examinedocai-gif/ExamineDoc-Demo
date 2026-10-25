
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

import re

def extract_names_and_dates(extracted_pages):
    """
    Collect possible names and dates from PDF text.
    Results require visual verification and are
    not automatically considered inconsistencies.
    """

    results = []

    date_pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE
    )

    name_pattern = re.compile(
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+"
        r"(?:\s+[A-Z][a-z]+)?\b"
    )

    for page in extracted_pages:
        page_number = page["page"]
        text = page["text"]

        for match in date_pattern.finditer(text):
            results.append({
                "Page": page_number,
                "Type": "Date",
                "Extracted value": match.group(),
                "Review status": "Unverified"
            })

        for match in name_pattern.finditer(text):
            results.append({
                "Page": page_number,
                "Type": "Possible name",
                "Extracted value": match.group(),
                "Review status": "Unverified"
            })

    return results




def extract_document_text(pdf):
    """
    Extract text from each PDF page while
    preserving page references.
    """

    results = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text", sort=True)

        results.append({
            "page": page_number,
            "text": text,
            "characters": len(text)
        })

    return results

def review_ocr_quality(page):
    """
    Identify indicators that extracted PDF text
    may require visual verification.

    This is not an OCR accuracy measurement
    or a document authenticity test.
    """

    text = page.get_text("text", sort=True)
    spans = []

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span["text"].strip():
                    spans.append(span)

    has_ocr_font = any(
        "GlyphLessFont" in span["font"]
        for span in spans
    )

    nonempty_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    short_lines = [
        line for line in nonempty_lines
        if len(line) <= 2
    ]

    short_line_ratio = (
        len(short_lines) / len(nonempty_lines)
        if nonempty_lines else 0
    )

    indicators = []

    if has_ocr_font:
        indicators.append(
            "OCR-style font detected. Embedded text "
            "may not reflect original typography."
        )

    if len(nonempty_lines) >= 10 and short_line_ratio > 0.20:
        indicators.append(
            "Many very short extracted lines detected. "
            "Text ordering or recognition may need review."
        )

    if not text.strip():
        indicators.append(
            "No extractable text found. Image-only "
            "pages may require OCR."
        )

    return {
        "ocr_style_font": has_ocr_font,
        "nonempty_lines": len(nonempty_lines),
        "short_line_ratio": short_line_ratio,
        "indicators": indicators
    }




def detect_document_type(page):
    """
    Classify a PDF page by its text and image structure.
    This is a preliminary technical classification.
    """

    blocks = page.get_text("dict")["blocks"]

    has_images = any(
        block.get("type") == 1
        for block in blocks
    )

    spans = []

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    spans.append(span)

    has_text = bool(spans)

    has_ocr = any(
        "GlyphLessFont" in span.get("font", "")
        for span in spans
    )

    if has_ocr:
        doc_type = "Scanned/OCR-style PDF"
    elif has_images and has_text:
        doc_type = "Mixed image and text PDF"
    elif has_images:
        doc_type = "Image-only PDF"
    elif has_text:
        doc_type = "Digital text PDF"
    else:
        doc_type = "Unclassified PDF page"

    return {
        "type": doc_type,
        "has_images": has_images,
        "has_text": has_text,
        "ocr_style_font": has_ocr
    }



def measure_scanned_typography(page, min_percent=15):
    """
    Measure visible characteristics of detected
    regions in a scanned document.

    These are image measurements, not evidence
    of altered fonts or document tampering.
    """

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
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
    minimum_width = page_width * min_percent / 100

    measurements = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if not (
            w >= minimum_width
            and 10 <= h <= 45
        ):
            continue

        region_gray = gray[y:y+h, x:x+w]
        region_binary = binary[y:y+h, x:x+w]

        dark_pixel_ratio = float(
            np.count_nonzero(region_binary)
            / region_binary.size
        )

        sharpness = float(
            cv2.Laplacian(
                region_gray,
                cv2.CV_64F
            ).var()
        )

        measurements.append({
            "Left (px)": x,
            "Top (px)": y,
            "Width (px)": w,
            "Height (px)": h,
            "Dark pixel ratio": round(
                dark_pixel_ratio, 4
            ),
            "Sharpness measure": round(
                sharpness, 2
            )
        })

    measurements.sort(
        key=lambda item: (
            item["Top (px)"],
            item["Left (px)"]
        )
    )

    return measurements



def compare_scanned_typography(measurements):
    """
    Compare similarly sized text regions.

    Return measurement outliers for technical review.
    Results are not evidence of document alteration.
    """

    results = []

    # Compare only reasonably long text regions.
    # This reduces comparisons with short headings.
    comparable = [
        item for item in measurements
        if item["Width (px)"] >= 500
        and 20 <= item["Height (px)"] <= 40
    ]

    # Avoid drawing conclusions from tiny samples.
    if len(comparable) < 8:
        return results

    # Group similar line heights together.
    groups = {}

    for item in comparable:
        height = item["Height (px)"]

        if height <= 25:
            group = "20-25 px"
        elif height <= 30:
            group = "26-30 px"
        else:
            group = "31-40 px"

        groups.setdefault(group, []).append(item)

    for group_name, items in groups.items():

        if len(items) < 6:
            continue

        for metric in [
            "Dark pixel ratio",
            "Sharpness measure"
        ]:
            values = np.array(
                [item[metric] for item in items],
                dtype=float
            )

            median_value = float(np.median(values))
            deviations = np.abs(values - median_value)
            mad = float(np.median(deviations))

            # A zero MAD cannot support this rule.
            if mad <= 1e-9:
                continue

            for item in items:
                difference = abs(
                    item[metric] - median_value
                )

                robust_score = (
                    0.6745 * difference / mad
                )

                if robust_score > 3.5:
                    results.append({
                        "Group": group_name,
                        "Measurement": metric,
                        "Observed value": item[metric],
                        "Group median": round(
                            median_value, 4
                        ),
                        "Robust score": round(
                            robust_score, 2
                        ),
                        "Left (px)": item["Left (px)"],
                        "Top (px)": item["Top (px)"],
                        "Width (px)": item["Width (px)"],
                        "Height (px)": item["Height (px)"],
                        "Status": "Technical review only"
                    })

    return results


def measure_word_regions(page):
    """
    Detect candidate word-sized image regions
    in a scanned PDF.

    These measurements do not determine whether
    text was changed or altered.
    """

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
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Connect nearby printed characters.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 2)
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

    measurements = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Approximate word-sized regions.
        if not (
            15 <= w <= 250
            and 8 <= h <= 45
        ):
            continue

        region_gray = gray[y:y+h, x:x+w]
        region_binary = binary[y:y+h, x:x+w]

        if region_gray.size == 0:
            continue

        darkness = float(
            np.count_nonzero(region_binary)
            / region_binary.size
        )

        sharpness = float(
            cv2.Laplacian(
                region_gray,
                cv2.CV_64F
            ).var()
        )

        measurements.append({
            "Left (px)": x,
            "Top (px)": y,
            "Width (px)": w,
            "Height (px)": h,
            "Dark pixel ratio": round(darkness, 4),
            "Sharpness": round(sharpness, 2)
        })

    measurements.sort(
        key=lambda item: (
            item["Top (px)"],
            item["Left (px)"]
        )
    )

    return measurements



def find_repeated_words(pdf, minimum_length=4):
    """
    Locate repeated words in a PDF text layer.

    Returns word positions and page references.
    OCR text and coordinates remain unverified.
    """

    from collections import defaultdict
    import re

    occurrences = defaultdict(list)

    for page_number, page in enumerate(pdf, start=1):
        words = page.get_text("words", sort=True)

        for word in words:
            x0, y0, x1, y1, text = word[:5]

            # Remove punctuation around words.
            cleaned = re.sub(
                r"^[^\w]+|[^\w]+$",
                "",
                text
            ).casefold()

            if (
                len(cleaned) < minimum_length
                or not any(c.isalpha() for c in cleaned)
            ):
                continue

            occurrences[cleaned].append({
                "Page": page_number,
                "Left": round(x0, 2),
                "Top": round(y0, 2),
                "Right": round(x1, 2),
                "Bottom": round(y1, 2),
                "Original text": text
            })

    return {
        word: locations
        for word, locations in occurrences.items()
        if len(locations) >= 2
    }


def crop_word_image(pdf, location, zoom=3):
    """
    Crop a word's printed appearance from a PDF.
    OCR coordinates may be inaccurate.
    """

    page = pdf[location["Page"] - 1]

    rect = fitz.Rect(
        location["Left"],
        location["Top"],
        location["Right"],
        location["Bottom"]
    )

    rect = fitz.Rect(
        rect.x0 - 3,
        rect.y0 - 3,
        rect.x1 + 3,
        rect.y1 + 3
    )

    rect = rect & page.rect

    if rect.is_empty or rect.is_infinite:
        return None

    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        clip=rect,
        colorspace=fitz.csRGB,
        alpha=False
    )

    if pix.width == 0 or pix.height == 0:
        return None

    return Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )


def measure_cropped_word(word_image):
    """
    Measure image characteristics of a cropped word.

    These are descriptive measurements, not
    font identification or tampering findings.
    """

    if word_image is None:
        return None

    gray = np.array(
        word_image.convert("L")
    )

    # Apply a small blur to reduce pixel noise.
    smoothed = cv2.GaussianBlur(
        gray, (3, 3), 0
    )

    _, binary = cv2.threshold(
        smoothed,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    dark_ratio = float(
        np.count_nonzero(binary) / binary.size
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    return {
        "Crop width (px)": word_image.width,
        "Crop height (px)": word_image.height,
        "Dark pixel ratio": round(dark_ratio, 4),
        "Sharpness measure": round(sharpness, 2)
    }


def normalize_word_image(word_image):
    """
    Prepare a cropped word for visual comparison.

    Removes isolated horizontal rules where possible.
    Preserves the original input image.
    This is experimental image processing.
    """

    if word_image is None:
        return None

    gray = np.array(word_image.convert("L"))

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Identify long horizontal structures.
    # These may be underlines or parts of letters.
    line_width = max(20, int(binary.shape[1] * 0.75))

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (line_width, 1)
    )

    possible_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Remove only lines located in the lower
    # portion of the image.
    lower_start = int(binary.shape[0] * 0.75)

   
    # Identify separate horizontal components near the bottom.
    removal_mask = np.zeros_like(binary)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    for component in range(1, n_labels):
        x, y, w, h, area = stats[component]

        if (
            y >= lower_start
            and w >= binary.shape[1] * 0.65
            and h <= max(4, binary.shape[0] * 0.12)
            and w >= 4 * h
        ):
            removal_mask[labels == component] = 255

    cleaned = cv2.subtract(binary, removal_mask)
 

    # Find remaining dark pixels.
    points = cv2.findNonZero(cleaned)

    if points is None:
        return word_image.copy()

    x, y, w, h = cv2.boundingRect(points)

    # Preserve a small border.
    padding = 3

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(cleaned.shape[1], x + w + padding)
    y1 = min(cleaned.shape[0], y + h + padding)

    cropped = cleaned[y0:y1, x0:x1]

    # Convert back to black text on white.
    normalized = 255 - cropped

    return Image.fromarray(normalized).convert("RGB")



def diagnose_horizontal_lines(word_image):
    """
    Display candidate horizontal components in red.

    Diagnostic visualization only.
    Does not modify the original image.
    """

    if word_image is None:
        return None

    gray = np.array(word_image.convert("L"))

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    height, width = binary.shape
    lower_start = int(height * 0.75)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    candidate_mask = np.zeros_like(binary)

    for component in range(1, n_labels):
        x, y, w, h, area = stats[component]

        if (
            y >= lower_start
            and w >= width * 0.65
            and h <= max(4, height * 0.12)
            and w >= 4 * h
        ):
            candidate_mask[labels == component] = 255

    diagnostic = np.array(
        word_image.convert("RGB")
    ).copy()

    # Highlight detected candidate pixels in red.
    diagnostic[candidate_mask > 0] = [255, 0, 0]

    return Image.fromarray(diagnostic)



def analyze_horizontal_ink(word_image):
    """
    Measure horizontal dark-pixel coverage
    in a cropped word image.

    Diagnostic only. Does not change the image
    or establish document alteration.
    """

    if word_image is None:
        return []

    gray = np.array(word_image.convert("L"))

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    height, width = binary.shape

    results = []

    for y in range(height):
        dark_pixels = int(
            np.count_nonzero(binary[y, :])
        )

        coverage = dark_pixels / width

        results.append({
            "Row (px)": y,
            "Dark pixels": dark_pixels,
            "Coverage (%)": round(coverage * 100, 2),
            "Possible horizontal rule": (
                coverage >= 0.65
            )
        })

    return results







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




def review_layout(regions, page_width):
    """
    Identify statistical layout outliers.
    These are review candidates, not evidence
    of alteration or forgery.
    """

    results = []

    # Only compare likely text-line regions.
    # Exclude unusually tall shapes and short marks.
    lines = [
        (x, y, w, h)
        for x, y, w, h in regions
        if 10 <= h <= 45
        and w >= page_width * 0.30
    ]

    if len(lines) < 6:
        return results

    heights = np.array(
        [r[3] for r in lines],
        dtype=float
    )

    median_height = float(np.median(heights))

    # Flag substantially unusual line heights.
    for x, y, w, h in lines:
        if (
            median_height > 0
            and (
                h < median_height * 0.70
                or h > median_height * 1.40
            )
        ):
            results.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "reason": "Unusual text-region height"
            })

    

    # Compare vertical gaps between adjacent regions.
    ordered = sorted(lines, key=lambda r: r[1])

    gaps = []
    for previous, current in zip(
        ordered, ordered[1:]
    ):
        gap = current[1] - (
            previous[1] + previous[3]
        )

        if 0 <= gap <= median_height * 3:
            gaps.append(gap)

    if len(gaps) >= 5:
        median_gap = float(np.median(gaps))

        for previous, current in zip(
            ordered, ordered[1:]
        ):
            gap = current[1] - (
                previous[1] + previous[3]
            )

            if (
                median_gap > 2
                and gap > median_gap * 2.5
                and gap <= median_height * 3
            ):
                x, y, w, h = current

                results.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "reason": "Unusual vertical spacing"
                })

    return results



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

            # Identify layout review candidates
            page_width = annotated.shape[1]

            layout_findings = review_layout(
                regions, page_width
            )

            # Add orange rectangles to the report image
            for finding in layout_findings:
                x = int(finding["x"])
                y = int(finding["y"])
                w = int(finding["width"])
                h = int(finding["height"])

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (255, 165, 0),
                    4
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
   
            st.subheader("Document Type Detection")

            document_types = []

            for page_index, current_page in enumerate(
                pdf, start=1
            ):
                result = detect_document_type(
                    current_page
                )

                document_types.append({
                    "Page": page_index,
                    "Detected type": result["type"],
                    "Text found": result["has_text"],
                    "Image blocks found": result["has_images"],
                    "OCR-style font": result["ocr_style_font"]
                })

            st.dataframe(
                document_types,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Page classification is preliminary. "
                "A PDF may contain different text and image "
                "structures that require further examination."
            )

            st.subheader("OCR Quality Review")

            quality_page = st.selectbox(
                "Select page for OCR quality review",
                range(1, len(pdf) + 1),
                key="ocr_quality_page"
            )

            quality = review_ocr_quality(
                pdf[quality_page - 1]
            )

            if quality["indicators"]:
                st.info(
                    "Text extraction may require "
                    "visual verification."
                )

                for indicator in quality["indicators"]:
                    st.write("• " + indicator)
            else:
                st.success(
                    "No text-quality indicators were "
                    "triggered by the current checks."
                )

            st.caption(
                "These checks do not measure OCR "
                "accuracy or determine whether the "
                "document has been altered."
            )


            st.subheader("Document Text Review")

            extracted_pages = extract_document_text(pdf)
        
            st.subheader("Names and Dates Review")

            extracted_items = extract_names_and_dates(
                extracted_pages
            )

            if extracted_items:
                st.dataframe(
                    extracted_items,
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "These are possible names and dates "
                    "identified in extracted PDF text. "
                    "They have not been verified against "
                    "the document image."
                )
            else:
                st.info(
                    "No names or dates matched the "
                    "current extraction patterns."
                )

            text_page_number = st.selectbox(
                "Choose a page to review its text",
                range(1, len(pdf) + 1),
                key="text_review_page"
            )

            page_result = extracted_pages[
                text_page_number - 1
            ]

            st.caption(
                f"Page {text_page_number} — "
                f"{page_result['characters']} "
                "extracted characters"
            )

            st.text_area(
                "Extracted text",
                value=page_result["text"],
                height=250,
                key=f"extracted_text_{text_page_number}"
            )

            if not page_result["text"].strip():
                st.info(
                    "No extractable text was found. "
                    "This page may require OCR."
                )

            st.caption(
                "Extracted text may contain OCR errors. "
                "Always compare it with the page image."
            )

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

            page_width = annotated.shape[1]

            layout_findings = review_layout(
                regions, page_width
            )

            # Highlight layout review candidates in orange
            highlighted = annotated.copy()

            for finding in layout_findings:
                x = int(finding["x"])
                y = int(finding["y"])
                w = int(finding["width"])
                h = int(finding["height"])

                cv2.rectangle(
                    highlighted,
                    (x, y),
                    (x + w, y + h),
                    (255, 165, 0),
                    4
                )

            st.subheader("Layout Inconsistency Review")

            st.metric(
                "Potential layout differences",
                len(layout_findings)
            )

            if layout_findings:
                st.dataframe(
                    [
                        {
                            "Reason": item["reason"],
                            "Left (px)": item["x"],
                            "Top (px)": item["y"],
                            "Width (px)": item["width"],
                            "Height (px)": item["height"]
                        }
                        for item in layout_findings
                    ],
                    hide_index=True
                )
            else:
                st.info(
                    "No layout differences met the "
                    "current review thresholds."
                )

            st.metric(
                "Detected regions on selected page",
                len(regions)
            )


            
            st.subheader("Repeated-Word Review")

            repeated_words = find_repeated_words(pdf)

            if repeated_words:
                selected_word = st.selectbox(
                    "Choose a repeated word",
                    sorted(repeated_words.keys()),
                    key="repeated_word_selection"
                )

                locations = repeated_words[selected_word]

                st.write(
                    f"'{selected_word}' appears "
                    f"{len(locations)} times in the PDF text layer."
                )

                st.dataframe(
                    locations,
                    hide_index=True,
                    use_container_width=True
                )
           
                st.subheader("Visual Word Comparison")

                comparison_locations = locations[:2]

                columns = st.columns(2)

                for index, location in enumerate(
                    comparison_locations
                ):
                    with columns[index]:
                        st.write(
                            f"Occurrence {index + 1} "
                            f"— Page {location['Page']}"
                        )

                        word_image = crop_word_image(
                            pdf,
                            location
                        )

                        if word_image is not None:
                            st.image(
                                word_image,
                                use_container_width=True
                            )


                            normalized_image = normalize_word_image(
                                word_image
                            )

                            if normalized_image is not None:
                                st.caption("Experimental normalized image")
                                st.image(
                                    normalized_image,
                                    use_container_width=True
                                )

                            diagnostic_image = diagnose_horizontal_lines(
                                word_image
                            )

                            if diagnostic_image is not None:
                                with st.expander(
                                    "Show horizontal-line diagnostic"
                                ):
                                    st.image(
                                        diagnostic_image,
                                        use_container_width=True
                                    )

                                    st.caption(
                                        "Red pixels indicate candidate "
                                        "horizontal-line components. "
                                        "They do not indicate alteration "
                                        "or tampering."
                                    )


                                    ink_results = analyze_horizontal_ink(
                                        word_image
                                    )

                                    if ink_results:
                                        st.write("Horizontal Ink Coverage")

                                        st.dataframe(
                                            ink_results,
                                            hide_index=True,
                                            use_container_width=True
                                        )

                                        high_coverage = [
                                            row for row in ink_results
                                            if row["Possible horizontal rule"]
                                        ]

                                        st.metric(
                                            "Rows with high ink coverage",
                                            len(high_coverage)
                                        )

                                    st.caption(
                                        "High ink coverage can indicate "
                                        "an underline, printed lettering, "
                                        "or another horizontal mark. "
                                        "It does not establish alteration."
                                    )

                            
                            word_metrics = measure_cropped_word(
                                word_image
                            )

                            if word_metrics:
                                st.write("Image Measurements")
                                st.dataframe(
                                    [
                                        {
                                            "Measurement": key,
                                            "Value": value
                                        }
                                        for key, value in word_metrics.items()
                                    ],
                                    hide_index=True,
                                    use_container_width=True
                                )
 
                        
                        else:
                            st.warning(
                                "Unable to extract word image."
                            )

                st.caption(
                    "Images are cropped using unverified "
                    "PDF/OCR word coordinates. Visual "
                    "differences do not establish tampering."
                )

 


                
                st.caption(
                    "Word locations are based on PDF text "
                    "or OCR and require visual verification. "
                    "Repeated words are not evidence of alteration."
                )
            else:
                st.info(
                    "No repeated words were found using "
                    "the current extraction settings."
                )


            
            st.subheader("Word-Level Image Measurements")

            word_results = measure_word_regions(
                pdf[number - 1]
            )

            st.metric(
                "Candidate word-sized regions",
                len(word_results)
            )

            if word_results:
                st.dataframe(
                    word_results,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info(
                    "No word-sized image regions matched "
                    "the current detection settings."
                )

            st.caption(
                "These are candidate word-sized shapes, "
                "not verified words or suspicious alterations. "
                "Measurements can be affected by scanning, "
                "punctuation, and connected characters."
            )


            
            st.subheader("Scanned Typography Measurements")

            typography_results = measure_scanned_typography(
                pdf[number - 1],
                min_percent
            )

    
            comparison_results = compare_scanned_typography(
                typography_results
            )

            st.subheader("Typography Comparison Review")

            if comparison_results:
                st.write(
                    f"{len(comparison_results)} statistical "
                    "measurement outliers identified."
                )

                st.dataframe(
                    comparison_results,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info(
                    "No statistical measurement outliers "
                    "were identified within eligible "
                    "comparison groups."
                )

            st.caption(
                "These are experimental statistical "
                "comparisons, not confirmed font changes, "
                "document alterations, or forensic findings."
            )
 
            
            if typography_results:
                st.dataframe(
                    typography_results,
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "These measurements describe image "
                    "characteristics of detected regions. "
                    "Differences do not establish font "
                    "alterations or document tampering."
                )
            else:
                st.info(
                    "No suitable text regions were found "
                    "for typography measurements."
                )
 
            st.subheader("Visual Examination")

            view_mode = st.radio(
                "Choose document view",
                [
                    "Original Document",
                    "Technical Measurements",
                    "Review Findings"
                ],
                horizontal=True
            )

            
            if view_mode == "Original Document":
                original_pix = pdf[number - 1].get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    colorspace=fitz.csRGB,
                    alpha=False
                )

                display_image = np.array(
                    Image.frombytes(
                        "RGB",
                        (original_pix.width, original_pix.height),
                        original_pix.samples
                    )
                )


            elif view_mode == "Technical Measurements":
                display_image = annotated

          
            else:
                
                clean_pix = pdf[number - 1].get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    colorspace=fitz.csRGB,
                    alpha=False
                )

                display_image = np.array(
                    Image.frombytes(
                        "RGB",
                        (clean_pix.width, clean_pix.height),
                        clean_pix.samples
                    )
                )

                for finding in layout_findings:
                    x = int(finding["x"])
                    y = int(finding["y"])
                    w = int(finding["width"])
                    h = int(finding["height"])

                    cv2.rectangle(
                        display_image,
                        (x, y),
                        (x + w, y + h),
                        (255, 165, 0),
                        4
                    )
 

                if not layout_findings:
                    st.info(
                        "No layout review candidates were "
                        "identified by the current rules."
                    )

            st.image(
                display_image,
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
