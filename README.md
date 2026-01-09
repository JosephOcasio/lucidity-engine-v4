# ⚛️ Lucidity Engine v4.0
### Semantic Signal-vs-Noise Detection Platform

This platform uses **Sentence-Transformers (ML)** and **Vector Embeddings** to differentiate between rigorous scientific structure and unstructured "woo" or static.

## 🚀 Key Features
* **Semantic Analysis:** Uses `all-MiniLM-L6-v2` to understand conceptual meaning beyond keywords.
* **Real-time Visualization:** Reactive Gradio dashboard showing "Conceptual Density" over time.
* **Persistent History:** SQLite backend to track and log analysis results.
* **Production Grade:** Modular architecture with separated Engine, Strategy, and UI layers.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **ML Library:** Hugging Face (Sentence-Transformers)
* **Frontend:** Gradio
* **Database:** SQLAlchemy / SQLite
