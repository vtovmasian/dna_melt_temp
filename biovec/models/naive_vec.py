from typing import Optional, TextIO

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import pandas as pd
import numpy as np
from tqdm import tqdm

from .duplex_vec import is_valid_dna


def load_naive_model(model_fname):
    return Word2Vec.load(model_fname)


def split_ngrams(seq: str, n: int) -> list:
    """
    Overlapping n-grams of a single strand.
    e.g. 'ACGTA', n=3 -> ['ACG', 'CGT', 'GTA']
    """
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


class NaiveVec(Word2Vec):
    """
    Encoding #2: Double Vector / Naive (Baseline) Encoding.

    A single Word2Vec model is trained on the n-grams of both strands, each
    strand supplied as its own independent "sentence" -- exactly the
    traditional single-stranded BioVec approach, just fed both halves of the
    duplex. A duplex is represented by encoding each strand independently
    with this shared model and concatenating the two mean-pooled vectors:
    encode(top, bottom) -> [vec(top) | vec(bottom)].

    Unlike DuplexVec, top and bottom are not required to be the same length.
    """

    def __init__(self, df: pd.DataFrame, corpus=None, n=3, size=64, corpus_file="naive_corpus.txt",
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

                if len(top) >= self.n:
                    f.write(" ".join(split_ngrams(top, self.n)) + "\n")
                if len(bottom) >= self.n:
                    f.write(" ".join(split_ngrams(bottom, self.n)) + "\n")

    def _encode_strand(self, seq: str) -> np.ndarray:
        ngrams = split_ngrams(seq, self.n)
        vecs = [self.wv[g] for g in ngrams if g in self.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def encode(self, top: str, bottom: str) -> np.ndarray:
        top = top.strip().upper()
        bottom = bottom.strip().upper()

        if not is_valid_dna(top) or not is_valid_dna(bottom):
            raise ValueError(f"Encode: Invalid DNA Sequence: {top}-{bottom}")

        return np.concatenate([self._encode_strand(top), self._encode_strand(bottom)])

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

        return np.vstack(vectors) if vectors else np.empty((0, 2 * self.vector_size))
