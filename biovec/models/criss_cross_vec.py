from typing import Optional, TextIO

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import pandas as pd
import numpy as np
from tqdm import tqdm

from .duplex_vec import is_valid_dna


def load_criss_cross_model(model_fname):
    return Word2Vec.load(model_fname)


def generate_criss_cross_tokens(top: str, bottom: str, n: int = 1) -> list:
    """
    Criss-cross Encoding: cross strand 1 with strand 2 over an n-position
    sliding window. Each token is n consecutive base-pairs (2n characters),
    aligned to base-pair boundaries -- a token never splits a single
    position's top/bottom pair the way a raw n-gram over an interleaved
    top-bottom-top-bottom... string would -- but overlapping across
    positions (stride 1) the same way DuplexVec/MismatchVec's n-grams do,
    so position/context survives instead of being averaged away.

    e.g. Top='ACGTA', Bottom='TGCAT':
      n=1 -> ['AT', 'CG', 'GC', 'TA', 'AT']                  (16-word vocab: 4x4)
      n=2 -> ['ATCG', 'CGGC', 'GCTA', 'TAAT']                (256-word vocab: 4x4x4x4)
      n=3 -> ['ATCGGC', 'CGGCTA', 'GCTAAT']                  (4096-word vocab, matches
                                                               DuplexVec's n=3 vocab size)

    A mismatch (e.g. top='G', bottom='A' -> 'GA') is automatically a
    distinct vocabulary word from a Watson-Crick pair (e.g. 'GC') without
    any special-casing, same as the n=1 version.
    """
    assert len(top) == len(bottom)
    return [
        "".join(top[i + k] + bottom[i + k] for k in range(n))
        for i in range(len(top) - n + 1)
    ]


class CrissCrossVec(Word2Vec):
    """Criss-cross (positional base-pair) duplex encoding."""

    def __init__(self, df: pd.DataFrame, n: int = 3, corpus=None, size=64, corpus_file="criss_cross_corpus.txt",
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

                tokens = generate_criss_cross_tokens(top, bottom, self.n)
                f.write(" ".join(tokens) + "\n")

    def encode(self, top: str, bottom: str) -> np.ndarray:
        top = top.strip().upper()
        bottom = bottom.strip().upper()

        if not is_valid_dna(top) or not is_valid_dna(bottom):
            raise ValueError(f"Encode: Invalid DNA Sequence: {top}-{bottom}")
        if len(top) != len(bottom):
            raise ValueError("Encode: Top and Bottom must be of the same length.")

        tokens = generate_criss_cross_tokens(top, bottom, self.n)
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
