# Streaming k-Medoids for Mixed-Type Healthcare Data

This repository contains the code, configuration files, and scripts used in the paper:

**Streaming + Coreset k-Medoids for Large-Scale Mixed-Type Healthcare Data**  
*(Submitted to NeurIPS 2025 MusIML Workshop)*

---

## 🔧 Features
- Streaming + coreset k-medoids algorithm with weighted Gower distance  
- Supports numeric, binary, and categorical features  
- Chunk-wise streaming with Hungarian alignment of medoids  
- Coreset refinement for scalability under strict memory limits  
- Feature weighting modes: uniform, manual (e.g., ethnicity emphasis), supervised (benchmark only)  
- Built-in metrics: ARI, NMI, Silhouette, Purity, and cluster-specific Precision/Recall  
- Logging of runtime, peak memory, and learned feature weights  

---
## ▶️ Usage Example

Run the pipeline:

python stream_kmedoids_pipeline.py --config configs/config_uniform.yml

To generate synthetic data (200k patients, modify proportions and numbers inside code if needed):

python generate_dataset.py

📊 Reproducing Figures

All figures in the paper can be reproduced with the scripts in evaluation/:

plot_results.py: runtime vs size, memory vs size, accuracy vs size.

plot_weighting.py: ethnicity weighting comparison heatmaps.

plot_summary.py: phenotype summary (Appendix Figure 1).

💻 Environment

Python 3.10

Dependencies in requirements.txt (numpy, pandas, scikit-learn, pyclustering, tqdm, matplotlib, seaborn).

📄 Citation

If you use this code, please cite:

@inproceedings{Shah2025StreamingKMedoids,
  title     = {Streaming + Coreset k-Medoids for Large-Scale Mixed-Type Healthcare Data},
  author    = {Syed Ahmar Shah},
  booktitle = {NeurIPS 2025 Muslims in ML Workshop},
  year      = {2025}
}
