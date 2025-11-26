import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from utils import recondition_kernel

st.set_page_config(page_title="Condition Number Reconditioning 3×3", layout="wide")
st.title("Condition Number Reconditioning 3×3")

st.markdown(
    "This page takes a CSV of flattened **3×3 kernels**, computes the "
    "**condition number** for each kernel, checks if it exceeds a chosen "
    "threshold **C**, and applies SVD-based reconditioning if needed. "
    "The exported CSV contains **only** the condition number, a flag indicating "
    "whether reconditioning occurred, and the **final 3×3 matrix** after processing."
)

rec_csv = st.file_uploader("Upload a CSV of flattened 3×3 kernels (each row has 9 values)", type=["csv"])
C_val = st.number_input("Condition number threshold C", min_value=1.0, value=5.0, step=0.5)
rec_btn = st.button("Run reconditioning")

if rec_csv is not None and rec_btn:
    df_in = pd.read_csv(rec_csv, header=None)

    if df_in.shape[1] != 9:
        st.error("CSV must contain exactly 9 columns (each row = flattened 3×3 matrix).")
    else:
        mats = df_in.values.reshape(-1, 3, 3)

        conds = np.array([np.linalg.cond(m) for m in mats])
        needs_rec = conds > C_val

        output_mats = []
        flags = []

        for m, cond_val in zip(mats, conds):
            if cond_val > C_val:
                m_rec, _, _, _, _ = recondition_kernel(m, C_val)
                output_mats.append(m_rec.reshape(-1))
                flags.append("reconditioned")
            else:
                output_mats.append(m.reshape(-1))
                flags.append("unchanged")

        output_mats = np.vstack(output_mats)

        # Build final output dataframe
        rec_cols = [f"val_{i+1}" for i in range(9)]
        df_out = pd.DataFrame(output_mats, columns=rec_cols)
        df_out["condition_number"] = conds
        df_out["status"] = flags

        st.subheader("Preview")
        st.dataframe(df_out.head(20), use_container_width=True)

        csv_buffer = StringIO()
        df_out.to_csv(csv_buffer, index=False)

        st.download_button(
            "Download reconditioned CSV",
            csv_buffer.getvalue().encode("utf-8"),
            file_name="reconditioned_output.csv",
            mime="text/csv"
        )

elif rec_btn and rec_csv is None:
    st.error("Please upload a CSV file first.")
