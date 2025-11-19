import streamlit as st
import numpy as np
import pandas as pd
from utils import compute_symmetry_score, recondition_kernel, kernel_distance

st.set_page_config(page_title="Random and Input Kernel Lab", layout="wide")
st.title("Random and Input Kernel Lab")

st.markdown(
    "This page is a sandbox for individual kernels. In the **Random matrix** tab, "
    "you can generate a random n×n kernel, compute its symmetry score, condition "
    "number, and reconditioned version. In the **Input matrix** tab, you can "
    "paste a single kernel by hand and run the same analysis pipeline on it."
)
tabs = st.tabs(["Random matrix", "Input matrix"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    size = c1.selectbox("Filter size", [3,5,7,9,11], index=0)
    mean = c2.number_input("μ (mean)", value=0.0, step=0.1)
    std = c3.number_input("σ (stddev)", value=1.0, min_value=0.0, step=0.1)
    c4, c5 = st.columns(2)
    seed = c4.number_input("Random seed", value=0, step=1)
    C = c5.number_input("Condition cap C", value=5.0, min_value=1.0, step=0.5)
    run_rand = st.button("Generate and analyze")

    if run_rand:
        if std <= 0:
            st.error("stddev must be > 0")
        else:
            if seed is not None:
                np.random.seed(int(seed))
            K = np.random.normal(loc=mean, scale=std, size=(size, size))
            score_before = compute_symmetry_score(K)
            K_rec, cond_before, cond_after, s_before, s_after = recondition_kernel(K, C)
            score_after = compute_symmetry_score(K_rec)
            dist = kernel_distance(K, K_rec)

            m1, m2 = st.columns(2)
            with m1:
                st.subheader("Input kernel")
                st.dataframe(pd.DataFrame(K), use_container_width=True)
            with m2:
                st.subheader("Output (reconditioned) kernel")
                st.dataframe(pd.DataFrame(K_rec), use_container_width=True)

            st.divider()
            cA, cB, cC = st.columns(3)
            cA.metric("Frobenius distance ‖K−Kʳᵉᶜ‖", f"{dist:.6g}")
            cB.metric("Condition number (before)", f"{cond_before:.6g}")
            cC.metric("Condition number (after)", f"{cond_after:.6g}")

            cD, cE = st.columns(2)
            cD.metric("Symmetry score (before)", f"{score_before:.4f}")
            cE.metric("Symmetry score (after)", f"{score_after:.4f}")

            st.subheader("Singular values")
            s_cols = st.columns(2)
            s_cols[0].write("Before")
            s_cols[0].write(np.array2string(s_before, precision=6))
            s_cols[1].write("After")
            s_cols[1].write(np.array2string(s_after, precision=6))

with tabs[1]:
    c1, c2 = st.columns(2)
    size_in = c1.selectbox("Matrix size", [3,5,7,9,11], index=0, key="in_size")
    C_in = c2.number_input("Condition cap C", value=5.0, min_value=1.0, step=0.5, key="in_C")

    st.write("Paste your matrix values below (space/comma separated rows; one row per line). Example for 3×3:")
    st.code("0.1, 0.2, 0.3\n0.0, -0.4, 0.5\n1.2, 0.7, -0.8", language="text")

    txt = st.text_area("Input matrix", height=160, placeholder="a11, a12, ... a1n\na21, a22, ... a2n\n...\nan1, an2, ... ann")
    run_input = st.button("Analyze input matrix")

    if run_input:
        try:
            rows = [r.strip() for r in txt.strip().splitlines() if r.strip()]
            arr = []
            for r in rows:
                parts = [p.strip() for p in r.replace(",", " ").split()]
                arr.append([float(x) for x in parts])
            A = np.array(arr, dtype=float)
            if A.shape != (size_in, size_in):
                st.error(f"Matrix must be {size_in}×{size_in}, but got {A.shape[0]}×{A.shape[1]}")
            else:
                score_before = compute_symmetry_score(A)
                A_rec, cond_before, cond_after, s_before, s_after = recondition_kernel(A, C_in)
                score_after = compute_symmetry_score(A_rec)
                dist = kernel_distance(A, A_rec)

                m1, m2 = st.columns(2)
                with m1:
                    st.subheader("Input kernel")
                    st.dataframe(pd.DataFrame(A), use_container_width=True)
                with m2:
                    st.subheader("Output (reconditioned) kernel")
                    st.dataframe(pd.DataFrame(A_rec), use_container_width=True)

                st.divider()
                cA, cB, cC = st.columns(3)
                cA.metric("Frobenius distance ‖A−Aʳᵉᶜ‖", f"{dist:.6g}")
                cB.metric("Condition number (before)", f"{cond_before:.6g}")
                cC.metric("Condition number (after)", f"{cond_after:.6g}")

                cD, cE = st.columns(2)
                cD.metric("Symmetry score (before)", f"{score_before:.4f}")
                cE.metric("Symmetry score (after)", f"{score_after:.4f}")

                st.subheader("Singular values")
                s_cols = st.columns(2)
                s_cols[0].write("Before")
                s_cols[0].write(np.array2string(s_before, precision=6))
                s_cols[1].write("After")
                s_cols[1].write(np.array2string(s_after, precision=6))
        except Exception as e:
            st.error(f"Failed to parse matrix: {e}")
