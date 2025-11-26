import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from utils import compute_symmetry_score

st.set_page_config(page_title="Symmetry Map From Image", layout="wide")
st.title("Symmetry Map From Image")

st.markdown(
    "This page takes a square image (e.g. 64×64), treats each pixel as an intensity value, "
    "and for every interior pixel it builds a 3×3 patch centered on that pixel. "
    "For each 3×3 patch it computes the **symmetry score** and uses that score as the new "
    "intensity value. Border pixels (first/last row and column) are ignored, so the output "
    "image has size `(H−2)×(W−2)` (e.g. 62×62 for a 64×64 input)."
)

uploaded = st.file_uploader(
    "Upload a square grayscale image (PNG/JPG/JPEG). If the image is RGB it will be converted to grayscale.",
    type=["png", "jpg", "jpeg"]
)

run_btn = st.button("Compute symmetry map")

if uploaded is not None and run_btn:
    img = Image.open(uploaded).convert("L")
    arr = np.array(img).astype(float)

    H, W = arr.shape
    if H < 3 or W < 3:
        st.error("Image must be at least 3×3.")
    else:
        out_h, out_w = H - 2, W - 2
        out = np.zeros((out_h, out_w), dtype=float)

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                patch = arr[i - 1:i + 2, j - 1:j + 2]
                s = compute_symmetry_score(patch)
                out[i - 1, j - 1] = s

        out = np.clip(out, 0.0, 1.0)
        out_uint8 = (out * 255).astype(np.uint8)
        out_img = Image.fromarray(out_uint8, mode="L")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Original image ({H}×{W})")
            st.image(img, use_column_width=True)
        with c2:
            st.subheader(f"Symmetry map ({out_h}×{out_w})")
            st.image(out_img, use_column_width=True)

        buf = BytesIO()
        out_img.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            "Download symmetry map (PNG)",
            buf.getvalue(),
            file_name="symmetry_map.png",
            mime="image/png"
        )

elif run_btn and uploaded is None:
    st.error("Please upload an image first.")
