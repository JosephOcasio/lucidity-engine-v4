
import gradio as gr
import pandas as pd
from src.engine import VectorLucidityEngine, SemanticVectorStrategy

engine = VectorLucidityEngine(strategy=SemanticVectorStrategy())

def run_app(text):
    scores = engine.analyze(text)
    avg = sum(scores)/len(scores) if scores else 0
    verdict = "⚛️ SCIENCE" if avg > 60 else "🔮 WOO"
    df = pd.DataFrame({"Window": range(len(scores)), "Score": scores})
    return verdict, df

demo = gr.Interface(
    fn=run_app,
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=[gr.Label(label="Verdict"), gr.LinePlot(x="Window", y="Score")],
    title="Lucidity Engine v4"
)

if __name__ == "__main__":
    demo.launch()
    