import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from utils import load_model_from_bytes_cached, kernels_to_matrices

st.set_page_config(page_title="Model Layer Inspector", layout="wide")
st.title("Model Layer Inspector")

st.markdown(
    "This page lets you upload a Keras `.h5` model, inspect all its layers, "
    "filter convolution kernels by spatial size (3×3, 5×5, etc.), and export "
    "the kernels of a selected layer to a CSV file for further offline analysis."
)

if "model" not in st.session_state:
    st.session_state["model"] = None
if "layers_df" not in st.session_state:
    st.session_state["layers_df"] = None

sizes = st.multiselect("Select kernel sizes to include", [3,5,7,9,11], default=[3,5,7,9,11])
show_all = st.checkbox("Show all layers", value=False)
uploaded = st.file_uploader("Drag & drop a Keras .h5 model", type=["h5"])
run = st.button("Show layers")

if run:
    if uploaded is None:
        st.error("please upload an .h5 model first")
    elif not sizes:
        st.error("please select at least one kernel size")
    else:
        st.session_state["model_bytes"] = uploaded.getvalue()
        with st.spinner("Loading model..."):
            st.session_state["model"] = load_model_from_bytes_cached(st.session_state["model_bytes"])
        model = st.session_state["model"]
        records = []
        for idx, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights and isinstance(weights[0], np.ndarray) and weights[0].ndim == 4:
                W0 = weights[0]
                if W0.shape[0] <= 11 and W0.shape[1] <= 11:
                    h, w, in_ch, out_ch = W0.shape
                else:
                    out_ch, in_ch, h, w = W0.shape
                num_matrices = int(in_ch) * int(out_ch)
                status = "matched" if (h == w and h in sizes and h not in (1,2)) else "ignored_size"
                records.append({
                    "index": idx,
                    "layer_name": layer.name,
                    "kernel_h": int(h),
                    "kernel_w": int(w),
                    "in_channels": int(in_ch),
                    "out_channels": int(out_ch),
                    "num_matrices": int(num_matrices),
                    "status": status
                })
            else:
                records.append({
                    "index": idx,
                    "layer_name": layer.name,
                    "kernel_h": None,
                    "kernel_w": None,
                    "in_channels": None,
                    "out_channels": None,
                    "num_matrices": 0,
                    "status": "no_matrices"
                })
        st.session_state["layers_df"] = pd.DataFrame(records)

if st.session_state["layers_df"] is not None:
    df = st.session_state["layers_df"]
    matched = df[df["status"] == "matched"].copy()
    if show_all:
        st.subheader(f"All layers ({len(df)})")
        st.dataframe(df, use_container_width=True)
        selectable = df[(df["status"] != "no_matrices") & (df["kernel_h"].notna())]
    else:
        st.subheader(f"Matched convolution layers ({len(matched)})")
        st.dataframe(matched, use_container_width=True)
        selectable = matched

    if not selectable.empty:
        st.divider()
        st.subheader("Download matrices from a layer")
        options = selectable.apply(lambda r: f'#{int(r["index"])} | {r["layer_name"]} | {int(r["kernel_h"])}x{int(r["kernel_w"])} | {int(r["num_matrices"])} mats', axis=1).tolist()
        index_map = selectable["index"].tolist()
        sel = st.selectbox("Select a layer", options, key="sel_layer_option")
        sel_idx = index_map[options.index(sel)]
        download_btn = st.button("Download matrices CSV of selected layer")

        if download_btn:
            if st.session_state["model"] is None:
                st.error("Model not loaded. Click 'Show layers' after uploading a model.")
            else:
                layer = st.session_state["model"].layers[sel_idx]
                w = layer.get_weights()
                if not w:
                    st.error("Selected layer has no weights")
                else:
                    mats, h, w_ = kernels_to_matrices(w[0])
                    if mats is None:
                        st.error("Unsupported kernel tensor shape")
                    else:
                        flat = mats.reshape(mats.shape[0], -1)
                        sio = StringIO()
                        np.savetxt(sio, flat, delimiter=",", fmt="%.8g")
                        csv_bytes = sio.getvalue().encode("utf-8")
                        st.download_button(
                            f"Download CSV: layer{sel_idx:03d}_{layer.name}_{h}x{w_}.csv",
                            csv_bytes,
                            file_name=f"layer{sel_idx:03d}_{layer.name}_{h}x{w_}.csv",
                            mime="text/csv",
                            key=f"dl_btn_{sel_idx}"
                        )
else:
    st.info("Upload a model and click 'Show layers' to proceed")
