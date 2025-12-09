# ConvNet Kernel Lab (Streamlit)

Tools for inspecting Keras .h5 models, extracting convolution kernels, analyzing symmetry scores, computing condition numbers, and generating symmetry maps from images.
Multi-page Streamlit app:

- **Layer Inspector** — list layers, filter by kernel size, export layer kernels to CSV
- **Symmetry & Distribution Analysis** — Compute mean kernel, symmetry score, and histogram (mean/median shown)
- **Condition Number Reconditioning (3×3)** — compute condition numbers, flag by threshold C, apply SVD reconditioning
- **Random & Input Kernel Lab** — generate random n×n kernels or input your own matrix; analyze and recondition
- **Condition Number Analysis** — compute condition numbers from CSVs or visualize distributions
- **Symmetry Map From Image** — compute pixel-wise symmetry heatmaps using 3×3 patches

## Cone Run locally
```bash
git clone https://github.com/Elhassene/ConvNet-Inspector.git
cd ConvNet-Inspector
pip install -r requirements.txt
streamlit run Home.py

