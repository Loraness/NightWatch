import asyncio
from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CACHE_DIR = "models_cache"


class EmbeddingService:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> None:
        self._model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=cache_dir,
        )

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts)

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


class RerankerService:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> None:
        self._model = CrossEncoder(
            model_name,
            device="cpu",
            cache_folder=cache_dir,
        )

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]

    async def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        return await asyncio.to_thread(self._predict_sync, pairs)


@lru_cache(maxsize=1)
def get_reranker_service() -> RerankerService:
    return RerankerService()
