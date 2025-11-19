# H5 Kernel Lab (Streamlit)

Tools for inspecting Keras `.h5` models, extracting conv kernels, analyzing symmetry scores, and SVD-based reconditioning.  
Multi-page Streamlit app:

- **Layer Inspector** — list layers, filter by kernel size, export layer kernels to CSV
- **CSV Analysis** — mean matrix + symmetry score; histogram with mean/median lines
- **Recondition 3×3** — add condition numbers, flag by threshold C, SVD recondition rows
- **Random & Input Matrix** — generate random n×n or paste your own; analyze + recondition

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
