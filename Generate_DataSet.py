import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# --------- Helper Functions ---------
def generate_cluster_data(n_samples, cluster_id, cfg):
    np.random.seed(cluster_id + 42)

    # Age & BMI as rounded integers
    age_mean = np.mean(cfg["age_range"])
    age_sd = (cfg["age_range"][1] - cfg["age_range"][0]) / 4
    age = np.random.normal(loc=age_mean, scale=age_sd, size=n_samples)
    age = np.clip(np.round(age, 1), *cfg["age_range"])

     # BMI with a tighter distribution and sensible variation
    bmi_mean = np.mean(cfg["bmi_range"])
    bmi_sd = (cfg["bmi_range"][1] - cfg["bmi_range"][0]) / 4  # ~95% within range
    bmi = np.random.normal(loc=bmi_mean, scale=bmi_sd, size=n_samples)
    bmi = np.clip(np.round(bmi, 1), *cfg["bmi_range"])

    data = {
        "age": age,
        "bmi": bmi,
        "smoking": np.random.binomial(1, cfg["smoking_prob"], size=n_samples),
        "eosinophilia": np.random.binomial(1, cfg["eos_prob"], size=n_samples),
        "neutrophilia": np.random.binomial(1, cfg["neut_prob"], size=n_samples),
        "allergy": np.random.binomial(1, cfg["allergy_prob"], size=n_samples),
        "severity": np.random.binomial(1, cfg["severity_prob"], size=n_samples),
        "drug_obesity": np.random.binomial(1, cfg["obesity_drug_prob"], size=n_samples),
        "drug_hypertension": np.random.binomial(1, cfg["htn_drug_prob"], size=n_samples),
        "drug_dyslipidemia": np.random.binomial(1, cfg["lipid_drug_prob"], size=n_samples),
        "drug_diabetes": np.random.binomial(1, cfg["diabetes_drug_prob"], size=n_samples),
        "ethnicity": np.random.choice(
            list(cfg["ethnicity_dist"].keys()),
            size=n_samples,
            p=list(cfg["ethnicity_dist"].values())
        ),
        "true_cluster": [cluster_id] * n_samples
    }

    return pd.DataFrame(data)

def plot_cluster_sizes(df):
    counts = df["true_cluster"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Cluster Size Distribution (200k Synthetic Patients)", fontsize=14)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Number of Patients", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_ethnicity_by_cluster(df):
    counts = df.groupby(["true_cluster", "ethnicity"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    props.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.title("Ethnicity Distribution per Cluster", fontsize=14)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.legend(title="Ethnicity", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --------- Configurations ---------
PHENOTYPE_NAMES = {
    0: "Early-Onset Mild Allergic",
    1: "Early-Onset Allergic Moderate–Severe",
    2: "Late-Onset Non-Allergic Eosinophilic",
    3: "Late-Onset Non-Allergic Neutrophilic",
    4: "Obese Allergic Non-Eosinophilic",
    5: "Late-Onset Smoking Phenotype",
    6: "Obese Metabolic T2-Low Asthma",
    7: "Adult-Onset Metabolic Syndrome Asthma",
    8: "Black Mixed-Inflammation Severe Asthma",
    9: "South Asian Poorly Controlled Asthma"
}

# Probabilities and ranges per phenotype (from literature/domain knowledge)
CLUSTER_CONFIGS = [
    # 0: Early-Onset Mild Allergic
    {"age_range": (16, 35), "bmi_range": (18, 25), "smoking_prob": 0.05,
     "eos_prob": 0.8, "neut_prob": 0.1, "allergy_prob": 0.9, "severity_prob": 0.2,
     "obesity_drug_prob": 0.01, "htn_drug_prob": 0.05, "lipid_drug_prob": 0.05, "diabetes_drug_prob": 0.02,
     "ethnicity_dist": {"White": 0.85, "Asian": 0.05, "Black": 0.03, "Mixed": 0.02, "Other": 0.03, "Unknown": 0.02}},

    # 1: Early-Onset Allergic Moderate–Severe
    {"age_range": (16, 40), "bmi_range": (20, 28), "smoking_prob": 0.1,
     "eos_prob": 0.85, "neut_prob": 0.15, "allergy_prob": 0.95, "severity_prob": 0.6,
     "obesity_drug_prob": 0.05, "htn_drug_prob": 0.07, "lipid_drug_prob": 0.07, "diabetes_drug_prob": 0.03,
     "ethnicity_dist": {"White": 0.83, "Asian": 0.07, "Black": 0.04, "Mixed": 0.02, "Other": 0.02, "Unknown": 0.02}},

    # 2: Late-Onset Non-Allergic Eosinophilic
    {"age_range": (40, 70), "bmi_range": (20, 28), "smoking_prob": 0.1,
     "eos_prob": 0.75, "neut_prob": 0.1, "allergy_prob": 0.2, "severity_prob": 0.5,
     "obesity_drug_prob": 0.04, "htn_drug_prob": 0.2, "lipid_drug_prob": 0.2, "diabetes_drug_prob": 0.1,
     "ethnicity_dist": {"White": 0.88, "Asian": 0.04, "Black": 0.03, "Mixed": 0.02, "Other": 0.02, "Unknown": 0.01}},

    # 3: Late-Onset Non-Allergic Neutrophilic
    {"age_range": (45, 70), "bmi_range": (22, 30), "smoking_prob": 0.2,
     "eos_prob": 0.15, "neut_prob": 0.75, "allergy_prob": 0.15, "severity_prob": 0.55,
     "obesity_drug_prob": 0.06, "htn_drug_prob": 0.25, "lipid_drug_prob": 0.25, "diabetes_drug_prob": 0.12,
     "ethnicity_dist": {"White": 0.87, "Asian": 0.05, "Black": 0.04, "Mixed": 0.02, "Other": 0.01, "Unknown": 0.01}},

    # 4: Obese Allergic Non-Eosinophilic
    {"age_range": (30, 55), "bmi_range": (30, 45), "smoking_prob": 0.15,
     "eos_prob": 0.2, "neut_prob": 0.6, "allergy_prob": 0.8, "severity_prob": 0.5,
     "obesity_drug_prob": 0.2, "htn_drug_prob": 0.2, "lipid_drug_prob": 0.25, "diabetes_drug_prob": 0.15,
     "ethnicity_dist": {"White": 0.8, "Asian": 0.08, "Black": 0.05, "Mixed": 0.03, "Other": 0.02, "Unknown": 0.02}},

    # 5: Late-Onset Smoking Phenotype
    {"age_range": (45, 70), "bmi_range": (20, 28), "smoking_prob": 0.95,
     "eos_prob": 0.25, "neut_prob": 0.65, "allergy_prob": 0.2, "severity_prob": 0.6,
     "obesity_drug_prob": 0.05, "htn_drug_prob": 0.3, "lipid_drug_prob": 0.3, "diabetes_drug_prob": 0.15,
     "ethnicity_dist": {"White": 0.9, "Asian": 0.03, "Black": 0.04, "Mixed": 0.02, "Other": 0.005, "Unknown": 0.005}},

    # 6: Obese Metabolic T2-Low Asthma
    {"age_range": (35, 65), "bmi_range": (32, 45), "smoking_prob": 0.1,
     "eos_prob": 0.2, "neut_prob": 0.4, "allergy_prob": 0.3, "severity_prob": 0.5,
     "obesity_drug_prob": 0.25, "htn_drug_prob": 0.35, "lipid_drug_prob": 0.4, "diabetes_drug_prob": 0.25,
     "ethnicity_dist": {"White": 0.78, "Asian": 0.1, "Black": 0.06, "Mixed": 0.03, "Other": 0.02, "Unknown": 0.01}},

    # 7: Adult-Onset Metabolic Syndrome Asthma
    {"age_range": (40, 70), "bmi_range": (28, 40), "smoking_prob": 0.15,
     "eos_prob": 0.25, "neut_prob": 0.5, "allergy_prob": 0.25, "severity_prob": 0.55,
     "obesity_drug_prob": 0.2, "htn_drug_prob": 0.4, "lipid_drug_prob": 0.45, "diabetes_drug_prob": 0.3,
     "ethnicity_dist": {"White": 0.82, "Asian": 0.08, "Black": 0.05, "Mixed": 0.03, "Other": 0.01, "Unknown": 0.01}},

    # 8: Black Mixed-Inflammation Severe Asthma
    {"age_range": (25, 55), "bmi_range": (24, 35), "smoking_prob": 0.3,
     "eos_prob": 0.5, "neut_prob": 0.5, "allergy_prob": 0.6, "severity_prob": 0.8,
     "obesity_drug_prob": 0.15, "htn_drug_prob": 0.25, "lipid_drug_prob": 0.25, "diabetes_drug_prob": 0.2,
     "ethnicity_dist": {"White": 0.3, "Asian": 0.05, "Black": 0.55, "Mixed": 0.05, "Other": 0.03, "Unknown": 0.02}},

    # 9: South Asian Poorly Controlled Asthma
    {"age_range": (25, 55), "bmi_range": (25, 35), "smoking_prob": 0.1,
     "eos_prob": 0.4, "neut_prob": 0.4, "allergy_prob": 0.5, "severity_prob": 0.75,
     "obesity_drug_prob": 0.12, "htn_drug_prob": 0.2, "lipid_drug_prob": 0.25, "diabetes_drug_prob": 0.22,
     "ethnicity_dist": {"White": 0.05, "Asian": 0.85, "Black": 0.03, "Mixed": 0.03, "Other": 0.02, "Unknown": 0.02}}
]

# Cluster sizes summing to 200,000
CLUSTER_SIZES = [40000, 20000, 24000, 22000, 26000, 14000, 18000, 18000, 8000, 10000]

# --------- Main Script ---------
if __name__ == "__main__":
    all_data = []
    for cid, cfg in enumerate(CLUSTER_CONFIGS):
        cluster_df = generate_cluster_data(CLUSTER_SIZES[cid], cid, cfg)
        all_data.append(cluster_df)

    df_synthetic = pd.concat(all_data).sample(frac=1, random_state=42).reset_index(drop=True)
    df_synthetic["phenotype_name"] = df_synthetic["true_cluster"].map(PHENOTYPE_NAMES)

    print(df_synthetic.head())
    print(f"\nTotal samples: {len(df_synthetic)}")

    plot_cluster_sizes(df_synthetic)
    plot_ethnicity_by_cluster(df_synthetic)

    df_synthetic.to_csv("synthetic_asthma_200k.csv", index=False)
    print("Dataset saved to synthetic_asthma_10k.csv")
