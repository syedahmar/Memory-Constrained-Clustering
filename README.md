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


💻 Environment

Python 3.10


📄 Citation

If you use this code, please cite:

@inproceedings{Shah2025StreamingKMedoids,
  title     = {Streaming + Coreset k-Medoids for Large-Scale Mixed-Type Healthcare Data},
  author    = {Syed Ahmar Shah, Fatima Almaghrabi, Aziz Sheikh},
  booktitle = {NeurIPS 2025 Muslims in ML Workshop},
  year      = {2025}
}
