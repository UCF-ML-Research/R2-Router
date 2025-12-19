"""
Query Embedding Module
Converts text queries to embedding vectors for routing
"""

import numpy as np
from typing import Union, List
import config


class VLLMEmbedder:
    """Embeds queries using vLLM deployed embedding model"""

    def __init__(
        self,
        model_name: str = None,
        tensor_parallel_size: int = None,
        max_model_len: int = None,
        batch_size: int = None,
        dtype: str = None
    ):
        """
        Initialize the vLLM embedder

        Args:
            model_name: Model name or path (uses config if None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            batch_size: Batch size for embedding generation
            dtype: Data type for model
        """
        self.model_name = model_name or config.VLLM_ENCODER_NAME
        self.tensor_parallel_size = tensor_parallel_size or config.VLLM_TENSOR_PARALLEL_SIZE
        self.max_model_len = max_model_len or config.VLLM_MAX_MODEL_LEN
        self.batch_size = batch_size or config.VLLM_BATCH_SIZE
        self.dtype = dtype or config.VLLM_DTYPE

        self.llm = None
        self.embedding_dim = None

    def load_model(self):
        """Load the vLLM embedding model (lazy loading)"""
        if self.llm is not None:
            return

        print(f"Loading vLLM embedding model: {self.model_name}")
        print(f"  Tensor parallel size: {self.tensor_parallel_size}")
        print(f"  Max model length: {self.max_model_len}")
        print(f"  Dtype: {self.dtype}")

        try:
            from vllm import LLM

            self.llm = LLM(
                model=self.model_name,
                task="embed",
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=True,
                gpu_memory_utilization=0.20,
            )

            # Determine embedding dimension by generating a test embedding
            test_output = self.llm.embed(["test"], use_tqdm=False)
            self.embedding_dim = len(test_output[0].outputs.embedding)

            print(f"✅ Loaded vLLM embedding model (dimension: {self.embedding_dim})")

        except ImportError:
            raise ImportError(
                "vLLM not installed. "
                "Install with: pip install vllm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load vLLM embedding model: {e}")

    def embed(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or more queries using vLLM

        Args:
            queries: Single query string or list of queries

        Returns:
            numpy array of shape (n_queries, embedding_dim)
        """
        # Ensure model is loaded
        if self.llm is None:
            self.load_model()

        # Handle single query
        if isinstance(queries, str):
            queries = [queries]

        # Generate embeddings in batches
        all_embeddings = []

        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            outputs = self.llm.embed(batch_queries, use_tqdm=False)

            for output in outputs:
                all_embeddings.append(output.outputs.embedding)

        embeddings = np.array(all_embeddings)

        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def __call__(self, queries: Union[str, List[str]]) -> np.ndarray:
        """Shorthand for embed()"""
        return self.embed(queries)


class QueryEmbedder:
    """Embeds queries using sentence transformers"""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedder

        Args:
            model_name: Sentence-transformer model name (uses config if None)
        """
        self.model_name = model_name or config.SENTENCE_TRANSFORMER_MODEL
        self.model = None
        self.embedding_dim = None

    def load_model(self):
        """Load the embedding model (lazy loading)"""
        if self.model is not None:
            return

        print(f"Loading embedding model: {self.model_name}")

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Loaded embedding model (dimension: {self.embedding_dim})")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def embed(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or more queries

        Args:
            queries: Single query string or list of queries

        Returns:
            numpy array of shape (n_queries, embedding_dim)
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Handle single query
        if isinstance(queries, str):
            queries = [queries]

        # Generate embeddings
        embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def __call__(self, queries: Union[str, List[str]]) -> np.ndarray:
        """Shorthand for embed()"""
        return self.embed(queries)


class MockEmbedder:
    """Mock embedder for testing without dependencies"""

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize mock embedder

        Args:
            embedding_dim: Dimension of embeddings to generate
        """
        self.embedding_dim = embedding_dim
        print(f"⚠️  Using MockEmbedder (dimension: {embedding_dim})")

    def load_model(self):
        """No-op for mock embedder"""
        pass

    def embed(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Generate mock embeddings (deterministic based on query text)

        Args:
            queries: Single query string or list of queries

        Returns:
            numpy array of shape (n_queries, embedding_dim)
        """
        if isinstance(queries, str):
            queries = [queries]

        # Generate deterministic embeddings based on query hash
        embeddings = []
        for query in queries:
            # Use query hash as seed for reproducibility
            seed = abs(hash(query)) % (2**32)
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self.embedding_dim)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def __call__(self, queries: Union[str, List[str]]) -> np.ndarray:
        """Shorthand for embed()"""
        return self.embed(queries)


# ============================================================================
# Factory Function
# ============================================================================

def get_embedder(mock: bool = None):
    """
    Get an embedder instance

    Args:
        mock: Use mock embedder (uses config.ENABLE_MOCK_MODE if None)

    Returns:
        VLLMEmbedder, QueryEmbedder, or MockEmbedder instance
    """
    if mock is None:
        mock = config.ENABLE_MOCK_MODE

    if mock:
        # Determine mock dimension based on config
        if config.EMBEDDING_MODEL == "vllm":
            mock_dim = 896  # Qwen2.5-Embedding-0.6B dimension
        else:
            mock_dim = 384  # Default
        return MockEmbedder(embedding_dim=mock_dim)
    else:
        # Use real embedder based on config
        if config.EMBEDDING_MODEL == "vllm":
            return VLLMEmbedder()
        else:
            return QueryEmbedder()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing QueryEmbedder...")
    print()

    # Test 1: Real embedder
    try:
        embedder = QueryEmbedder()
        embedder.load_model()

        # Single query
        query = "What is the capital of France?"
        embedding = embedder.embed(query)
        print(f"Single query embedding shape: {embedding.shape}")

        # Batch queries
        queries = [
            "What is the capital of France?",
            "Solve 2x + 5 = 13",
            "Explain photosynthesis"
        ]
        embeddings = embedder.embed(queries)
        print(f"Batch embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        print()

    except Exception as e:
        print(f"⚠️  Real embedder failed: {e}")
        print()

    # Test 2: Mock embedder
    print("Testing MockEmbedder...")
    mock_embedder = MockEmbedder(embedding_dim=384)

    query = "What is the capital of France?"
    embedding = mock_embedder.embed(query)
    print(f"Mock embedding shape: {embedding.shape}")

    # Test determinism
    embedding2 = mock_embedder.embed(query)
    assert np.allclose(embedding, embedding2), "Mock embeddings should be deterministic"
    print("✅ Mock embeddings are deterministic")
    print()

    # Test factory
    print("Testing factory function...")
    embedder_real = get_embedder(mock=False)
    embedder_mock = get_embedder(mock=True)
    print(f"Real embedder type: {type(embedder_real).__name__}")
    print(f"Mock embedder type: {type(embedder_mock).__name__}")
