import streamlit as st
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    layout="wide"
)

st.title("🔷 Shape & Contour Analyzer")
st.markdown(
    "Upload an image to detect geometric shapes, count objects, "
    "and display their area and perimeter."
)

# --------------------------------------------------
# Color definitions (BGR format)
# --------------------------------------------------
LIGHT_BLUE = (255, 255, 0)   # Cyan / Light Blue
RED = (0, 0, 255)

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

# --------------------------------------------------
# Shape detection function
# --------------------------------------------------
def detect_shape(cnt, approx):
    sides = len(approx)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return "Unknown"

    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    if sides == 3:
        return "Triangle"

    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"

    elif circularity > 0.80:
        return "Circle"

    else:
        return "Polygon"

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = image.copy()

    # -----------------------------
    # Pre-processing
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # -----------------------------
    # Contour detection
    # -----------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    object_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small noise
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        shape = detect_shape(cnt, approx)

        x, y, w, h = cv2.boundingRect(approx)

        # Draw bounding box (red)
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            RED,
            2
        )

        # Draw contour (light blue)
        cv2.drawContours(
            image,
            [approx],
            -1,
            LIGHT_BLUE,
            2
        )

        # Label
        label = f"{shape} {int(area)}"
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            LIGHT_BLUE,
            2
        )

        results.append({
            "Object ID": object_id,
            "Shape Type": shape,
            "Area (px)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2)
        })

        object_id += 1

    # --------------------------------------------------
    # Display images
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Input Image")
        st.image(original, channels="BGR", width=700)

    with col2:
        st.subheader(" Processed Image with Detected Shapes")
        st.image(image, channels="BGR", width=700)

    # --------------------------------------------------
    # Results table
    # --------------------------------------------------
    st.subheader(" Analysis Results")

    if results:
        df = pd.DataFrame(results)
        st.write(f"**Total Objects Detected:** {len(df)}")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No valid shapes detected.")

