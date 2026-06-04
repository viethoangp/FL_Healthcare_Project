# 🏥 Privacy-Preserving Federated Learning for Healthcare

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FFD21E)
![Differential Privacy](https://img.shields.io/badge/Security-Differential%20Privacy-success)

## 📖 Overview
This project implements a scalable, distributed Deep Learning framework for medical image classification (Tuberculosis Detection) using **Federated Learning (FL)**. By utilizing the **Flower (`flwr`)** framework, this system allows multiple autonomous hospitals (Clients) to collaboratively train a global ResNet50 model without ever sharing raw, sensitive patient X-ray data with the central server.

While inspired by recent baseline frameworks in healthcare FL, this project introduces **critical algorithmic improvements** to handle Extreme Non-IID (highly imbalanced) data distributions across hospitals, achieving state-of-the-art diagnostic accuracy while maintaining strict patient privacy.

---

## 🌟 Key Contributions & Improvements (Over Baseline)

### 1. Advanced Data Balancing: Feature-Space Borderline-SMOTE
* **Baseline Flaw:** Traditional SMOTE generates noisy, ghostly synthetic images that degrade model convergence when applied directly to pixels.
* **Our Solution:** Implemented **Borderline-SMOTE in the Latent Feature-Space**. The ResNet50 backbone acts as a feature extractor, and SMOTE is applied locally at each client to synthesize 2048-dimensional features exclusively near the decision boundaries. This prevents noise generation and significantly boosts the model's **Precision**.

### 2. Smart Server Aggregation: Dynamic Tau (DynTau)
* **Baseline Flaw:** Using a static divergence threshold ($\tau=0.1$) to switch between FedAvg and FedSGD is ineffective for deep networks (like ResNet50) because the L2-norm divergence scale naturally shrinks over training rounds. This causes the system to blindly penalize clients and get "stuck" in FedSGD, causing massive communication overhead.
* **Our Solution:** Proposed **Dynamic Tau (DynTau)**. The server calculates the *median divergence* of all clients dynamically in every round. This adaptive threshold seamlessly transitions the network between FedAvg (to save bandwidth) and FedSGD (to correct client drift), ensuring optimal convergence even under Extreme Non-IID constraints.

---

## 🏆 Optimal Results

The proposed architecture (**Borderline-SMOTE + DynTau** trained over 35 rounds) demonstrated exceptional performance on the completely unseen independent Test Set, outperforming standard baselines.

### Model Performance Metrics (ResNet50)

| Class | Metric | Proposed Method (TH5) | Note |
| :--- | :--- | :---: | :--- |
| **Overall** | **Accuracy** | **95.68%** | Highly accurate overall classification. |
| **Tuberculosis** | **Precision** | **99.30%** | Exceptional precision (Extremely low False Positives). A TB positive prediction is almost absolutely correct. |
| | **Recall** | **91.50%** | High sensitivity in detecting disease. |
| | **F1-Score** | **95.25%** | Excellent balance between Precision and Recall. |
| **Normal** | **Recall** | **99.43%** | Very high specificity. Safely identifies healthy patients, minimizing unnecessary medical procedures. |

*(All metrics achieved with **Differential Privacy (DP)** enabled, proving the robustness of the privacy-utility tradeoff).*

---

## 🗄️ Datasets

The simulation was trained and tested on the following open-source medical datasets. To prevent initial overfitting caused by the extreme class imbalance in the original dataset (3500 Normal vs 700 TB), we integrated an augmented TB dataset to provide a richer foundation before applying local Borderline-SMOTE.

1. **Original TB Dataset:** [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) (Contains ~3500 Normal and 700 TB images).
2. **Augmented TB Dataset:** [Tuberculosis Augmented Images](https://www.kaggle.com/datasets/lavuluriliketh/tubeculosis-augmented-images) (Contains ~2440 augmented TB images to mitigate severe under-representation).

---

## 🚀 Technologies Used
* **Federated Architecture:** Robust Client-Server communication using the `flower` framework.
* **Deep Learning:** `PyTorch` & `torchvision` (Transfer Learning with pre-trained ResNet50).
* **Data Imbalance Handling:** `imbalanced-learn` (Borderline-SMOTE).
* **Privacy:** `opacus` (Differential Privacy via Gaussian noise injection).
* **Data Visualization:** `matplotlib`, `sklearn` (t-SNE, ROC/AUC, Confusion Matrices).

---

## 📂 Project Structure
```text
FL_Healthcare_Project/
├── config.py              # Central experiment & hyperparameter configuration
├── simulation.py          # Full Flower simulation entrypoint
├── requirements.txt       # Project dependencies
├── src/
│   ├── client.py           # Client training logic & Feature-space SMOTE
│   ├── aggregators.py      # Dynamic Tau (DynTau) logic & FedAvg/FedSGD switching
│   ├── models.py           # ResNet50 model instantiation & Backbone freezing
│   ├── strategy.py         # Custom Flower Strategy implementation
│   ├── data.py             # DataLoader & transform definitions
│   ├── partition.py        # Dirichlet (Extreme Non-IID) client partition logic
│   ├── prepare_data.py     # Preprocessing (Resize to 224x224, Train/Val/Test Split)
│   ├── evaluation.py       # Metrics evaluation utilities
│   ├── visualize.py        # Advanced charting (Bar charts, Line graphs)
│   └── advanced_visualize.py # t-SNE and ROC/AUC visualizations
├── data/                   # Datasets (Raw & Preprocessed splits)
└── results/                # Output models, logs, and generated charts
```

---

## ⚙️ Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/viethoangp/FL_Healthcare_Project.git
cd FL_Healthcare_Project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Dataset
Download the TB Chest X-ray dataset and place it in `data/tb/TB_Chest_Radiography_Database/`. Then run preprocessing to split the data (Train 70%, Val 15%, Test 15%):
```bash
python src/prepare_data.py
```

### 3. Run the Federated Simulation
Run the full simulation (default 10 rounds, or edit `config.py` for 35 rounds):
```bash
python simulation.py --num-rounds 10 --model resnet50
```

### 4. Generate Visualizations (For Reporting)
To generate comprehensive comparison charts (Accuracy bars, Loss convergence, t-SNE, ROC):
```bash
python src/visualize.py --compare results/TH1 results/TH2 results/TH3 results/TH4 results/TH5 --compare-labels "Baseline" "BorderSMOTE" "DynTau" "Proposed(10R)" "Proposed(35R)"
python src/advanced_visualize.py
```

---

## 📚 Citation & References

- **Baseline Paper:** Haripriya, R., Khare, N., Pandey, M., and Biswas, S. *A privacy-enhanced framework for collaborative big data analysis in healthcare using adaptive federated learning aggregation.* Journal of Big Data, 12(1), 113, 2025. DOI: [https://doi.org/10.1186/s40537-025-01169-8](https://doi.org/10.1186/s40537-025-01169-8)

**Frameworks & Libraries:**
- **Flower (Federated Learning):** [https://flower.dev](https://flower.dev)
- **PyTorch:** Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS 2019
