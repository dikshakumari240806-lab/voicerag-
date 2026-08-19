from sentence_transformers import SentenceTransformer
from .config import settings

class Embedder:
    def __init__(self):
        self.model=SentenceTransformer(settings.embed_model)
        self.dimension=self.model.get_sentence_embedding_dimension()
    def encode(self,texts):
        return self.model.encode(texts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
