
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticVectorStrategy:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.physics_anchor = self.model.encode("rigorous scientific physics mathematics empirical data")
        self.woo_anchor = self.model.encode("spiritual energy quantum mysticism magic vibes")

    def score_chunk(self, text_window: str) -> float:
        input_vector = self.model.encode(text_window)
        physics_sim = util.cos_sim(input_vector, self.physics_anchor).item()
        woo_sim = util.cos_sim(input_vector, self.woo_anchor).item()
        return max(0.0, min(100.0, 50 + ((physics_sim - woo_sim) * 100)))

class VectorLucidityEngine:
    def __init__(self, strategy):
        self.strategy = strategy

    def analyze(self, text: str, window_size: int = 1):
        sentences = [s.strip() for s in text.split('.') if len(s) > 10]
        return [self.strategy.score_chunk(". ".join(sentences[i : i + window_size])) 
                for i in range(len(sentences) - window_size + 1)]
    