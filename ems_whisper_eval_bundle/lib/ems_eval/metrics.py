"""Speech-to-text evaluation metrics for EMS terminology."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

# Optional dependencies - metrics work without them but with reduced functionality
def _get_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        return TfidfVectorizer, cosine_similarity, True
    except Exception:
        return None, None, False

def _get_nltk_bleu():
    """Lazy import of nltk BLEU to avoid pulling in scipy at module load."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        return sentence_bleu, SmoothingFunction, True
    except Exception:
        return None, None, False


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def _word_error_rate(ref: list[str], hyp: list[str]) -> float:
    """Standard WER: (S + D + I) / N where N = ref word count."""
    if not ref:
        return 0.0 if not hyp else float("inf")
    # Wagner-Fischer for word-level
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / n


class STTMetrics:
    """
    Speech-to-text metrics for EMS evaluation.
    Supports WER, CER, MWER (medical-aware), TF-IDF cosine, and BLEU.
    """

    def __init__(
        self,
        medical_vocab_path: Optional[Path] = None,
        tfidf_similarity_threshold: float = 0.80,
    ) -> None:
        """
        Args:
            medical_vocab_path: Path to medical_vocab.csv for MWER.
            tfidf_similarity_threshold: Cosine threshold for MWER word equivalence.
        """
        self.tfidf_similarity_threshold = tfidf_similarity_threshold
        self._vectorizer: Optional[Any] = None
        self._medical_terms: list[str] = []

        if medical_vocab_path is None:
            default_path = Path(__file__).parent / "data" / "medical_vocab.csv"
            medical_vocab_path = default_path

        if medical_vocab_path.exists():
            self._load_medical_vocab(medical_vocab_path)

    def _load_medical_vocab(self, path: Path) -> None:
        """Load medical vocabulary and build char-level TF-IDF."""
        self._medical_terms = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row.get("term", "").strip()
                if term:
                    self._medical_terms.append(term.lower())

        TfidfVectorizer, _, has_sklearn = _get_sklearn()
        if self._medical_terms and has_sklearn and TfidfVectorizer is not None:
            self._vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                lowercase=True,
            )
            self._vectorizer.fit(self._medical_terms)

    def compute_tfidf_cosine(self, a: str, b: str) -> float:
        """
        Character-level TF-IDF cosine similarity between two strings.
        Uses medical vocab if loaded; otherwise falls back to simple char n-gram overlap.
        """
        if not a or not b:
            return 0.0

        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0

        if self._vectorizer is not None:
            _, cosine_similarity, has_sklearn = _get_sklearn()
            if has_sklearn and cosine_similarity is not None:
                try:
                    vecs = self._vectorizer.transform([a, b])
                    sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
                    return float(sim)
                except Exception:
                    pass

        # Fallback: Jaccard on character trigrams
        def trigrams(s: str) -> set[str]:
            return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()

        ta, tb = trigrams(a), trigrams(b)
        if not ta and not tb:
            return 1.0 if a == b else 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Word Error Rate: (S + D + I) / N."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        return _word_error_rate(ref_words, hyp_words)

    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """Character Error Rate: edit distance / len(reference chars)."""
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        if not ref_chars:
            return 0.0 if not hyp_chars else float("inf")
        dist = _levenshtein_distance(ref_chars, hyp_chars)
        return dist / len(ref_chars)

    def _mwer_word_match(self, ref_word: str, hyp_word: str) -> bool:
        """True if words match exactly or have TF-IDF cosine > threshold."""
        if ref_word == hyp_word:
            return True
        return self.compute_tfidf_cosine(ref_word, hyp_word) >= self.tfidf_similarity_threshold

    def compute_mwer(self, reference: str, hypothesis: str) -> float:
        """
        Medical Word Error Rate: WER with medical-aware matching.
        Words that differ but have char-level TF-IDF cosine > threshold count as matches.
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if not ref_words:
            return 0.0 if not hyp_words else float("inf")

        n, m = len(ref_words), len(hyp_words)
        # DP with medical-aware substitution cost
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = float(i)
        for j in range(m + 1):
            dp[0][j] = float(j)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                ref_w, hyp_w = ref_words[i - 1], hyp_words[j - 1]
                if ref_w == hyp_w:
                    cost = 0.0
                elif self._mwer_word_match(ref_w, hyp_w):
                    cost = 0.0
                else:
                    cost = 1.0
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + cost,
                )

        return dp[n][m] / n

    def compute_bleu(self, reference: str, hypothesis: str) -> float:
        """BLEU score (sentence-level, 4-gram)."""
        sentence_bleu, SmoothingFunction, has_nltk = _get_nltk_bleu()
        if not has_nltk or sentence_bleu is None:
            return 0.0
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        try:
            refs = [ref_tokens]
            smoothing = SmoothingFunction()
            return float(sentence_bleu(refs, hyp_tokens, smoothing_function=smoothing.method1))
        except Exception:
            return 0.0

    def compute_all(
        self,
        reference: str,
        hypothesis: str,
    ) -> dict[str, float]:
        """
        Compute all metrics for a reference-hypothesis pair.

        Returns:
            Dict with keys: wer, cer, mwer, tfidf_cosine, bleu
        """
        ref = reference.strip()
        hyp = hypothesis.strip()
        tfidf_cos = self.compute_tfidf_cosine(ref, hyp)

        return {
            "wer": self.compute_wer(ref, hyp),
            "cer": self.compute_cer(ref, hyp),
            "mwer": self.compute_mwer(ref, hyp),
            "tfidf_cosine": tfidf_cos,
            "bleu": self.compute_bleu(ref, hyp),
        }
