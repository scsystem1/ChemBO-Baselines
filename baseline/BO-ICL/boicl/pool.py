"""utilities for building and selecting from a pool"""
from typing import List, Any, Callable
from difflib import SequenceMatcher

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


class Pool:
    """Class for sampling from pool of possible data points

    Example:
        >>> pool = Pool(['a', 'b', 'c', 'd', 'e'])
        >>> pool.sample(3)
        ['a', 'd', 'c']
        >>> pool.choose('a')
        >>> pool.sample(3)
        ['b', 'c', 'd']
        >>> pool.approx_sample('a', 3)
        ['b', 'c', 'd']
    """

    def __init__(self, pool: List[Any], formatter: Callable = lambda x: str(x)) -> None:
        if type(pool) is not list:
            raise TypeError("Pool must be a list")
        self._pool = pool
        self._selected = []
        self._available = pool[:]
        self.format = formatter
        self._formatted = [formatter(x) for x in pool]
        if TfidfVectorizer is not None:
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            self._matrix = self._vectorizer.fit_transform(self._formatted)
        else:
            self._vectorizer = None
            self._matrix = None

    def sample(self, n: int) -> List[str]:
        """Sample n items from the pool"""
        if n > len(self._available):
            raise ValueError("Not enough items in pool")
        samples = np.random.choice(self._available, size=n, replace=False)
        return samples

    def choose(self, x: str) -> None:
        """Choose a specific item from the pool"""
        if x not in self._available:
            raise ValueError("Item not in pool")
        self._selected.append(x)
        self._available.remove(x)

    def approx_sample(self, x: str, k: int, lambda_mult: float = 0.5) -> None:
        """Given an approximation of x, return k similar"""
        if self._vectorizer is not None and cosine_similarity is not None:
            query_vec = self._vectorizer.transform([self.format(x)])
            scores = cosine_similarity(query_vec, self._matrix).ravel()
        else:  # pragma: no cover
            scores = np.array(
                [SequenceMatcher(None, self.format(x), text).ratio() for text in self._formatted]
            )

        ranked = np.argsort(scores)[::-1]
        docs = [self._pool[idx] for idx in ranked if self._pool[idx] not in self._selected]
        return docs[:k]

    def reset(self) -> None:
        """Reset the pool"""
        self._selected = []
        self._available = self._pool[:]

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __str__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __iter__(self):
        return iter(self._available)
