# Cora Citation Network Classification & GCN Inference System

**Author:** Jayant Sharma  
**Contact:** [jayant.sharma2023@vitstudent.ac.in](mailto:jayant.sharma2023@vitstudent.ac.in)  
**Institution:** Vellore Institute of Technology (VIT)  

---

## Overview

This project implements an end-to-end graph machine learning pipeline to classify academic papers into 7 research topics using the **Cora Citation Network** dataset. The system compares a tabular baseline (TF-IDF + Random Forest) against a deep **Graph Convolutional Network (GCN)**, exports the trained model to **ONNX Runtime** for fast CPU inference, and serves predictions through a **FastAPI** backend with two interactive user interfaces (Vanilla HTML/CSS/JS dashboard and a Streamlit application).

```
   Citation Network (Graph)              GCN Feature Aggregation             FastAPI / ONNX Runtime
┌───────────────────────────────┐     ┌────────────────────────────┐     ┌──────────────────────────────┐
│  Paper A ──────► Paper B      │     │  h_i = σ( Σ W · h_j / c )  │     │  POST /predict               │
│  (1433-dim      (1433-dim     │ ──► │  Layer 1: 1433 -> 32       │ ──► │  POST /predict/cora_node     │
│   word vector)   word vector) │     │  Layer 2: 32 -> 7          │     │  Latency: ~1-3 ms (CPU)      │
└───────────────────────────────┘     └────────────────────────────┘     └──────────────────────────────┘
```

---

## Key Technical Highlights & Engineering Impact

- **End-to-End Graph ML Pipeline:** Developed and evaluated a full-lifecycle semi-supervised graph learning pipeline on the Cora citation benchmark (2,708 scientific publications, 10,556 citation edges, 1,433 vocabulary features).
- **Graph Convolutions vs. Tabular Baseline:** Designed a 2-layer Graph Convolutional Network (GCNConv) using PyTorch Geometric with neighborhood message passing, outperforming a TF-IDF + Random Forest baseline by **+21.10% absolute accuracy** (58.10% → 79.20%, a **+36.32% relative boost**) and **+21.26% Macro-F1** (56.98% → 78.24%).
- **Regularized Model Training & Early Stopping:** Implemented model training with Adam optimizer (`lr=0.01`, `weight_decay=5e-4`), dropout (`p=0.5`), and patience-based early stopping (`patience=20`), reaching 100% training accuracy and a peak validation accuracy of **81.20%** within 35 epochs.
- **Production ONNX Optimization:** Exported the PyG computational graph to ONNX Runtime (Opset 18) with dynamic tensor dimensions (`[num_nodes, 1433]` and `[2, num_edges]`), shrinking model footprint to **~64 KB** and achieving ultra-fast **sub-2ms CPU inference** for the full 2,708-node citation graph without loading PyTorch in production.
- **Asynchronous REST Microservice:** Built a high-performance FastAPI backend with Pydantic validation schemas, providing endpoints for custom graph topology inference (`/predict`), pre-indexed Cora node querying (`/predict/cora_node`), and execution health diagnostics (`/health`, `/info`).
- **Dual Full-Stack Interfaces:** Created two client interfaces: a single-page web dashboard (HTML5/CSS3/Vanilla JS) with an automated 5-iteration ONNX inference benchmark tool, and an interactive Streamlit application for synthetic graph generation and real-time classification.

---

## Dataset: Cora Benchmark

The Cora dataset consists of machine learning publications categorized into 7 scientific disciplines:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Nodes (Papers)** | 2,708 | Academic papers in computer science / machine learning |
| **Edges (Citations)** | 10,556 | Directed citation links (representing 5,429 unique undirected citation pairs) |
| **Features per Node** | 1,433 | Binary bag-of-words dictionary (1 = present, 0 = absent) |
| **Classes** | 7 | Subject categories |
| **Training Nodes** | 140 | 20 labeled papers per class (5.17% of total dataset) |
| **Validation Nodes** | 500 | Labeled papers used for hyperparameter tuning & early stopping |
| **Testing Nodes** | 1,000 | Held-out labeled papers for final model evaluation |
| **Unlabeled Context Nodes** | 1,068 | Propagate relational message-passing features during inference |

### Topic Classes & Distribution
1. `Case_Based` (298 papers)
2. `Genetic_Algorithms` (418 papers)
3. `Neural_Networks` (818 papers)
4. `Probabilistic_Methods` (426 papers)
5. `Reinforcement_Learning` (217 papers)
6. `Rule_Learning` (180 papers)
7. `Theory` (351 papers)

---

## Methodology & Architecture

### 1. Baseline Model (Random Forest)
- **Features:** 1,433 binary word indicators reweighted with `TfidfTransformer`, supplemented by node graph degree (number of connected citation neighbors).
- **Classifier:** `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Limitation:** Treats nodes mostly independently and cannot effectively leverage multi-hop neighborhood citation patterns.

### 2. Graph Convolutional Network (SimpleGCN)
- **Architecture:**
  - `GCNConv(in_channels=1433, out_channels=32)`
  - `ReLU()` activation + `Dropout(p=0.5)`
  - `GCNConv(in_channels=32, out_channels=7)`
- **Mechanism:** Localized first-order approximation of spectral graph convolutions (Kipf & Welling). Aggregates feature representations from directly cited and citing papers:
  $$\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{\tilde{A}}\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)$$
- **Optimization:** Adam optimizer (`lr=0.01`, `weight_decay=5e-4`), Cross-Entropy loss with early stopping (`patience=20`).

### 3. ONNX Model Export & Serving
- The PyTorch Geometric model is exported to an optimized ONNX computational graph (`simple_gcn_cora.onnx`, Opset 18) with dynamic axes for node counts and edge lengths.
- Predictions run through **ONNX Runtime** on CPU (`CPUExecutionProvider`), achieving sub-2ms inference times without loading the full PyTorch stack in production.

---

## Performance Comparison & Metrics

All metrics were evaluated on the held-out test split of 1,000 nodes:

| Evaluation Metric | Baseline (Random Forest) | Two-Layer GCN (SimpleGCN) | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **58.10%** (`0.5810`) | **79.20%** (`0.7920`) | **+21.10%** absolute (+36.32% relative) |
| **Test Macro-F1** | **56.98%** (`0.5698`) | **78.24%** (`0.7824`) | **+21.26%** absolute (+37.31% relative) |
| **Best Validation Accuracy** | — | **81.20%** (`0.8120`) | Converged at epoch 35 |
| **Final Training Accuracy** | 100.00% | 100.00% | Cross-entropy loss: 0.0224 |
| **Model Size** | — | **~64.8 KB** (ONNX) | Highly compact deployment |
| **Inference Latency (Full Graph)** | — | **~1.2 – 2.5 ms** (CPU) | Sub-millisecond per-node |

*Graph convolutions achieve a +21.10% absolute accuracy boost over tabular classification because paper topic distributions correlate heavily with citation topology (homophily).*

---

## Repository Structure

```
CORA-Project/
├── cora_citation_network_classification_(1).ipynb  # Training notebook & experiments
├── main.py                                        # FastAPI inference server & ONNX runtime
├── streamlit_app.py                               # Streamlit interactive dashboard
├── simple_gcn_cora.onnx                           # Exported ONNX GCN model (~64 KB)
├── simple_gcn_cora.onnx.data                      # External tensor weights for ONNX
├── requirements.txt                               # Python dependencies
├── .gitignore                                     # Git ignore definitions
├── README.md                                      # Project documentation
├── static/                                        # Frontend web application
│   ├── index.html                                 # Single-page dashboard UI
│   ├── index.css                                  # Custom dark theme styles
│   └── app.js                                     # API communication & benchmark scripts
└── data/
    └── Planetoid/
        └── Cora/                                  # Cora dataset raw & processed tensors
            ├── processed/
            │   ├── data.pt
            │   ├── pre_filter.pt
            │   └── pre_transform.pt
            └── raw/
                ├── ind.cora.allx
                ├── ind.cora.ally
                ├── ind.cora.graph
                ├── ind.cora.test.index
                ├── ind.cora.tx
                ├── ind.cora.ty
                ├── ind.cora.x
                └── ind.cora.y
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 to 3.12 recommended
- Git

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/jayant733/Cora-project.git
cd Cora-project

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend Server
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to view the interactive web dashboard.
- Interactive API Swagger documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Run the Streamlit Dashboard (Alternative UI)
In a separate terminal (with environment activated):
```bash
streamlit run streamlit_app.py
```
- The Streamlit interface allows you to query individual Cora nodes or generate random synthetic graphs for real-time inference against the local API (`http://127.0.0.1:8000`).

### 4. Running the Jupyter Notebook
```bash
jupyter notebook cora_citation_network_classification_(1).ipynb
```
- Walks through data loading, EDA, citation graph visualization using NetworkX, baseline training, GCN training with early stopping, evaluation metrics, and ONNX model export.

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web dashboard (`static/index.html`) |
| `GET` | `/health` | Healthcheck and ONNX execution providers status |
| `GET` | `/info` | Model metadata, input feature dimension (1433), and class mapping |
| `POST` | `/predict` | Predicts topics for custom node feature vectors and edge matrices |
| `POST` | `/predict/cora_node` | Runs inference for specific node indices from the Cora dataset (0 to 2707) |

### Sample Request: Query Cora Node
```bash
curl -X POST "http://127.0.0.1:8000/predict/cora_node" \
     -H "Content-Type: application/json" \
     -d '{"node_indices": [0, 10, 42]}'
```

### Sample Response:
```json
{
  "num_nodes": 2708,
  "num_edges": 10556,
  "predictions": [
    {
      "node_index": 0,
      "predicted_class_id": 2,
      "predicted_class_name": "Neural_Networks",
      "probabilites": [0.03, 0.02, 0.81, 0.05, 0.03, 0.02, 0.04],
      "logits": [0.42, 0.15, 3.84, 0.89, 0.35, 0.11, 0.65]
    }
  ]
}
```

---

## Author & Contact

**Jayant Sharma**  
Email: [jayant.sharma2023@vitstudent.ac.in](mailto:jayant.sharma2023@vitstudent.ac.in)  
Student, Vellore Institute of Technology (VIT)  
Repository: [https://github.com/jayant733/Cora-project](https://github.com/jayant733/Cora-project)
