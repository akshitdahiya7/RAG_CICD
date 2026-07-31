from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self,model_name:str="BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    def embed(self,texts:list[str])-> list[list[float]]:
        return self.model.encode(texts,show_progress_bar=False).tolist()

    @property
    def dimension(self)->int:
        return self.model.get_sentence_embedding_dimension()