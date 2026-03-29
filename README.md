# 🏥 Privacy-Preserving Federated Learning for Healthcare

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FFD21E)

## 📖 Overview
This project implements a scalable, distributed Deep Learning framework for medical image classification using **Federated Learning (FL)**. 

It is designed as a university capstone project for the **Big Data (IS405)** and **Distributed Databases (IS211)** courses. The architecture is heavily inspired by the baseline paper: *"A privacy-enhanced framework for collaborative Big Data analysis in healthcare using adaptive federated learning aggregation."*

By utilizing the **Flower (`flwr`)** framework, this system allows multiple autonomous nodes (representing hospitals) to collaboratively train a global Convolutional Neural Network (CNN) without ever sharing raw, sensitive patient data with the central server.

## ✨ Novel Contributions (Project Improvements)
While based on the baseline paper, our team proposes three major algorithmic and systemic improvements:
1. **Borderline-SMOTE Integration:** Upgrading from standard SMOTE to Borderline-SMOTE at the local client level. This focuses oversampling on "dangerous" borderline cases (critical in medical imaging) while avoiding noise generation, ensuring strict data privacy without cross-client leakage.
2. **Dynamic Aggregation Threshold ($\tau$):** Replacing the static divergence threshold ($\tau=0.10$) with a dynamic $\tau$. The threshold dynamically adjusts based on the convergence rate ($\Delta$loss/round), optimizing the switch between FedAvg and FedSGD algorithms for better global accuracy.
3. **Enhanced Scalability:** Expanding the federated network simulation from 10 to 15-20 autonomous clients. We address GPU memory constraints by implementing partial participation configurations.

## 🚀 Key Features
* **Federated Architecture:** Robust Client-Server communication using the Flower framework.
* **Deep Learning for Medical Imaging:** Utilizes Transfer Learning with **ResNet** and **VGG16** for high-accuracy image classification.
* **Strict Data Privacy:** Raw X-ray and MRI images remain strictly localized at the hospital level. Only mathematical model weights are transmitted.
* **Cloud Simulation Ready:** Fully compatible with Google Colab's GPU environment for simulating large-scale multi-client networks.

## 🗄️ Datasets
The models are trained and evaluated on three primary medical datasets (fetched via Kaggle API):
1. **NIH Chest X-ray** (Pneumonia detection)
2. **Brain Tumor MRI** (Tumor classification)
3. **APTOS 2019** (Diabetic Retinopathy detection)

## 📂 Project Structure
```text
FL_Healthcare_Project/
│
├── client.py           # Hospital node: Loads local data & trains the CNN model
├── server.py           # Central node: Aggregates weights (FedAvg/FedSGD)
├── requirements.txt    # Project dependencies
├── .gitignore          # Ignored files (venv, raw data)
└── README.md           # Project documentation
```
## ⚙️ Installation & Setup (Local Environment)
1. Clone the repository:

```bash
git clone https://github.com/viethoangp/FL_Healthcare_Project.git
cd FL_Healthcare_Project
```
2. Create a virtual environment:

```bash
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```
## 🏃‍♂️ How to Run a Mini-Test
To verify the federated network on your local machine, open multiple terminal windows.

Terminal 1 (Start the Central Server):
```bash
python server.py
```
Terminal 2 & 3 (Start the Hospital Clients):
```bash
python client.py
```