"""Optional local sentence embeddings; cached and stored on disk.

Install optional dependency explicitly: python -m pip install -e '.[semantic]'.
The indexed canonical text remains recoverable without the embedding model.
"""


class MiniLMEmbedder:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, *, device: str = "cpu", offline: bool = False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            self.model_id, device=device,
            local_files_only=offline
        )

    def encode(self, text: str) -> list[float]:
        return self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32").tolist()
