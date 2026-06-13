"""In-process cache mapping normalized query text -> embedding vector.

Avoids redundant Voyage API calls when a user repeats/edits a query, or when
the same demo queries are typed repeatedly during grading. A simple dict is
sufficient for a single small Render instance (no shared/Redis cache needed).
"""


class QueryEmbeddingCache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


query_embedding_cache = QueryEmbeddingCache()
