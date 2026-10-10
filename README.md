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
│  (1433-dim      (1433-dim     │ ──► │  Layer 1: 1433 -> 64       │ ──► │  POST /predict/cora_node     │
│   word vector)   word vector) │     │  Layer 2: 64 -> 7          │     │  Latency: ~1-3 ms (CPU)      │
└───────────────────────────────┘     └────────────────────────────┘     └──────────────────────────────┘
```

---

## Dataset: Cora Benchmark

The Cora dataset consists of machine learning publications categorized into 7 scientific disciplines:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Nodes (Papers)** | 2,708 | Academic papers in computer science / machine learning |
| **Edges (Citations)** | 5,429 | Undirected citation links (10,556 directed graph edges) |
| **Features per Node** | 1,433 | Binary bag-of-words dictionary (1 = present, 0 = absent) |
| **Classes** | 7 | Subject categories |

### Topic Classes
1. `Case_Based`
2. `Genetic_Algorithms`
3. `Neural_Networks`
4. `Probabilistic_Methods`
5. `Reinforcement_Learning`
6. `Rule_Learning`
7. `Theory`

---

## Methodology & Architecture

### 1. Baseline Model (Random Forest)
- **Features:** 1,433 binary word indicators reweighted with `TfidfTransformer`, supplemented by node graph degree (number of connected papers).
- **Classifier:** Random Forest with 100 estimators.
- **Limitation:** Treats nodes mostly independently and cannot effectively leverage multi-hop neighborhood citation patterns.

### 2. Graph Convolutional Network (SimpleGCN)
- **Architecture:**
  - `GCNConv(in_channels=1433, out_channels=64)`
  - `ReLU()` activation + `Dropout(p=0.5)`
  - `GCNConv(in_channels=64, out_channels=7)`
- **Mechanism:** Message passing aggregates feature representations from directly cited and citing papers, allowing topologically clustered papers to share contextual representations.
- **Optimization:** Adam optimizer (`lr=0.01`, `weight_decay=5e-4`), Cross-Entropy loss with early stopping (`patience=20`).

### 3. ONNX Model Export & Serving
- The PyTorch Geometric model is exported to an optimized ONNX computational graph (`simple_gcn_cora.onnx`, Opset 18) with dynamic axes for node counts and edge lengths.
- Predictions run through **ONNX Runtime** on CPU (`CPUExecutionProvider`), achieving sub-millisecond inference times without loading the full PyTorch stack in production.

---

## Performance Comparison

| Model | Test Accuracy | Macro-F1 Score |
| :--- | :---: | :---: |
| **Random Forest (Baseline + Degree)** | ~72.5% | ~0.70 |
| **Two-Layer GCN (Graph Convolutions)** | **~81.5%** | **~0.80** |

*Graph convolutions achieve a ~9% absolute accuracy improvement over tabular classification by incorporating topological citation context.*

---

## Repository Structure

```
CORA-Project/
├── cora_citation_network_classification_(1).ipynb  # Training notebook & experiments
├── main.py                                        # FastAPI inference server & ONNX runtime
├── streamlit_app.py                               # Streamlit interactive dashboard
├── simple_gcn_cora.onnx                           # Exported ONNX GCN model
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
git clone <repo-url>
cd CORA-Project

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
