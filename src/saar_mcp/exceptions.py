"""Package-specific exceptions with actionable messages."""


class SaarError(RuntimeError):
    """Base error for expected operational failures."""


class ConfigurationError(SaarError):
    """Raised when configuration is invalid or incomplete."""


class ProviderError(SaarError):
    """Raised when an embedding or reranking provider fails."""


class StorageError(SaarError):
    """Raised when Qdrant is unavailable or incompatible."""


class IndexingError(SaarError):
    """Raised when a source cannot be indexed safely."""
