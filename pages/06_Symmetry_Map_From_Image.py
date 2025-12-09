import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import ticker
from utils import compute_symmetry_score

st.set_page_config(page_title="Symmetry Map From Image", layout="wide")
st.title("Symmetry Map From Image")

st.markdown(
    "This page takes an image, converts it to grayscale, and for every interior pixel "
    "builds a 3×3 patch centered on that pixel. For each patch it computes the "
    "symmetry score and uses that score as the new intensity value. "
    "Border pixels are ignored, so the output image has size (H−2)×(W−2)."
)

if "sym_data" not in st.session_state:
    st.session_state["sym_data"] = None

T = st.sidebar.slider(
    "Symmetry threshold T (0 = show all, 1 = highlight only very symmetric regions)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)

uploaded = st.file_uploader(
    "Upload an image (PNG/JPG/JPEG). If the image is RGB it will be converted to grayscale.",
    type=["png", "jpg", "jpeg"]
)

run_btn = st.button("Compute symmetry map")

MAX_SIDE = 256

if uploaded is not None and run_btn:
    img_raw = Image.open(uploaded).convert("L")
    orig_w, orig_h = img_raw.size

    if max(orig_w, orig_h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(orig_w, orig_h))
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img_raw.resize((new_w, new_h), Image.BILINEAR)
    else:
        img = img_raw

    arr = np.array(img).astype(float)
    H, W = arr.shape

    if H < 3 or W < 3:
        st.error("Image must be at least 3×3 after resizing.")
        st.session_state["sym_data"] = None
    else:
        with st.spinner("Computing symmetry map..."):
            out_h, out_w = H - 2, W - 2
            out = np.zeros((out_h, out_w), dtype=float)
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    patch = arr[i - 1:i + 2, j - 1:j + 2]
                    s = compute_symmetry_score(patch)
                    out[i - 1, j - 1] = s

        out = np.clip(out, 0.0, 1.0)
        out_img = Image.fromarray((out * 255).astype(np.uint8), mode="L")

        st.session_state["sym_data"] = {
            "orig_img": img_raw,
            "orig_size": (orig_h, orig_w),
            "proc_size": (H, W),
            "out": out,
            "out_img": out_img,
        }

elif uploaded is None and run_btn:
    st.error("Please upload an image first.")
    st.session_state["sym_data"] = None

sym_data = st.session_state["sym_data"]

if sym_data is not None:
    orig_img = sym_data["orig_img"]
    orig_h, orig_w = sym_data["orig_size"]
    H, W = sym_data["proc_size"]
    out = sym_data["out"]
    out_img = sym_data["out_img"]
    out_h, out_w = out.shape

    denom = max(1e-8, 1.0 - T)
    out_adj = np.clip((out - T) / denom, 0.0, 1.0)
    out_adj_img = Image.fromarray((out_adj * 255).astype(np.uint8), mode="L")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader(f"Original ({orig_h}×{orig_w}, processed: {H}×{W})")
        st.image(orig_img, use_container_width=True)
    with c2:
        st.subheader(f"Raw symmetry map ({out_h}×{out_w})")
        st.image(out_img, use_container_width=True)
    with c3:
        st.subheader(f"Adjusted symmetry map (T = {T:.2f})")
        st.image(out_adj_img, use_container_width=True)

    buf = BytesIO()
    out_adj_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        "Download adjusted symmetry map (PNG)",
        buf.getvalue(),
        file_name=f"symmetry_map_T_{T:.2f}.png",
        mime="image/png"
    )

    st.divider()
    st.subheader("Distribution of symmetry scores in the raw symmetry map")

    scores = out.flatten()
    mean_val = float(np.mean(scores))
    median_val = float(np.median(scores))

    fig = plt.figure(figsize=(10, 6))
    bins = 12
    counts, bin_edges, _ = plt.hist(scores, bins=bins, edgecolor="black", alpha=0.7)

    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.3f}")
    plt.axvline(median_val, color="green", linestyle="-", linewidth=2, label=f"Median = {median_val:.3f}")

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]
    plt.xticks(centers, labels, rotation=45, ha="right")

    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if counts.size > 0:
        plt.ylim(0, counts.max() * 1.12)

    plt.xlabel("Symmetry score range")
    plt.ylabel("Frequency")
    plt.title("Symmetry Score Distribution (raw symmetry map)")
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)
