from typing import Optional, TextIO

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import pandas as pd
import numpy as np
from tqdm import tqdm

from .duplex_vec import is_valid_dna

WC_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def load_mismatch_model(model_fname):
    return Word2Vec.load(model_fname)


def to_mismatch_symbols(top: str, bottom: str) -> list:
    """
    Collapse each duplex position to a single symbol.

    A Watson-Crick match keeps the top-strand base, so a perfectly
    complementary duplex reads exactly like plain ssDNA (and shares
    vocabulary with the traditional single-stranded n-gram baseline). A
    mismatched position is instead replaced by its own two-letter
    "top+bottom" code (e.g. 'GA'), so mismatches are visible as distinct
    "characters" in the sequence rather than being silently discarded.
    """
    assert len(top) == len(bottom)
    symbols = []
    for t, b in zip(top, bottom):
        symbols.append(t if (t, b) in WC_PAIRS else t + b)
    return symbols


def generate_mismatch_ngrams(top: str, bottom: str, n: int = 3) -> list:
    """
    Slide an n-window over the per-position mismatch symbols (not raw
    characters) and join each window into a single token.
    e.g. Top='ACGTA', Bottom='TGAAT' (mismatch at position 3: G/A), n=3
      symbols -> ['A', 'C', 'GA', 'T', 'A']
      tokens  -> ['ACGA', 'CGAT', 'GATA']
    """
    symbols = to_mismatch_symbols(top, bottom)
    return ["".join(symbols[i:i + n]) for i in range(len(symbols) - n + 1)]


class MismatchVec(Word2Vec):
    """Mismatch Character duplex encoding."""

    def __init__(self, df: pd.DataFrame, corpus=None, n=3, size=64, corpus_file="mismatch_corpus.txt",
                 sg=1, window=5, min_count=1, workers=3):
        self.n = n
        self.size = size

        if corpus is None and df is None:
            raise Exception("Minimum Input Requirements:\ndf or corpus.")

        if df is not None:
            print("Generating corpus file from DataFrame...")
            self._generate_corpus(df, corpus_file)
            corpus = Text8Corpus(corpus_file)

        super().__init__(sentences=corpus, vector_size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def _generate_corpus(self, df, corpus_file):
        with open(corpus_file, 'w') as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating corpus"):
                try:
                    top = row['Top'].strip().upper()
                    bottom = row['Bottom'].strip().upper()
                except Exception:
                    continue

                if len(top) != len(bottom) or len(top) < self.n:
                    continue

                tokens = generate_mismatch_ngrams(top, bottom, self.n)
                f.write(" ".join(tokens) + "\n")

    def encode(self, top: str, bottom: str) -> np.ndarray:
        top = top.strip().upper()
        bottom = bottom.strip().upper()

        if not is_valid_dna(top) or not is_valid_dna(bottom):
            raise ValueError(f"Encode: Invalid DNA Sequence: {top}-{bottom}")
        if len(top) != len(bottom):
            raise ValueError("Encode: Top and Bottom must be of the same length.")

        tokens = generate_mismatch_ngrams(top, bottom, self.n)
        vecs = []
        for token in tokens:
            try:
                vecs.append(self.wv[token])
            except KeyError:
                raise Exception("Model has never trained this token: " + token)

        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def batch_encode(self, df: pd.DataFrame, verbose: bool = False, error_log_path: Optional[str] = None) -> np.ndarray:
        if verbose and not error_log_path:
            raise ValueError("Batch Encode: Error logging path not specified.")

        log: Optional[TextIO] = None
        if verbose:
            assert error_log_path is not None
            log = open(error_log_path, 'w')
            log.write("Failed to encode...\nIdx Top Bottom Error\n")

        vectors = []
        for i, row in tqdm(enumerate(df.itertuples(index=False)), total=len(df), desc="Corpus generation progress"):
            top = str(row.Top).strip().upper()
            bottom = str(row.Bottom).strip().upper()

            if not is_valid_dna(top) or not is_valid_dna(bottom):
                if verbose:
                    assert log is not None
                    log.write(f"{i} {top} {bottom} Invalid characters\n")
                continue

            if len(top) != len(bottom):
                if verbose:
                    assert log is not None
                    log.write(f"{i} {top} {bottom} Length mismatch\n")
                continue

            try:
                vec = self.encode(top, bottom)
                vectors.append(vec)
            except Exception as e:
                if verbose:
                    assert log is not None
                    log.write(f"{i}\t{top}\t{bottom}\t{e}\n")
                continue

        if verbose:
            assert log is not None
            log.close()

        return np.vstack(vectors) if vectors else np.empty((0, self.vector_size))
