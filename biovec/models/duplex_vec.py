from typing import Optional, TextIO

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus 
import pandas as pd
import numpy as np
from tqdm import tqdm

complement =  {
  "A": "T", 
  "T": "A",
  "C": "G", 
  "G": "C",
}

def load_duplex_model(model_fname):
   return Word2Vec.load(model_fname)

def generate_duplex_tokens(seq: str, c_seq: str, n=3):
    """
    Token Format: <seq_ngram>-<c_seq_ngram>
    """
    assert len(seq) == len(c_seq)
    return [f"{seq[i:i+n]}-{c_seq[i:i+n]}" for i in range(len(seq) - n + 1)]

def generate_complement(seq: str) -> str:
    return ''.join(complement[nt] for nt in seq)

def is_valid_dna(seq: str) -> bool:
   return bool(seq) and set(seq).issubset({'A', 'C', 'T', 'G'})

class DuplexVec(Word2Vec):
  def __init__(self, df: pd.DataFrame, corpus=None, n=3, size=64, corpus_file="corpus.txt", sg=1, window=3, min_count=1, workers=3, alpha=0.025, negative=5, hs=1, sample=1e-3):
    """
    df: Must contain columns "Top" and "Bottom"
    leftFlank: DNA sequence prepended to top strand
    rightFlank: DNA sequence appended to top strand
    n: length of n-gram
    size: the # of dimensions in embedding vector
    window: maximum distance between current and predicted word
    sg: 1 for skip-gram, 0 for CBOW
    minCount: if (token_occurences < minCount) ignore
    workers: # of worker threads
    """

    self.n = n
    self.size = size

    if corpus is None and df is None:
      raise Exception("Minimum Input Requirements:\ndf or corpus.")
    
    if df is not None:
      print("Generating corpus file from DataFrame...")
      self._generate_corpus(df, corpus_file)
      corpus = Text8Corpus(corpus_file)
    
    super().__init__(sentences=corpus, vector_size=size, sg=sg, window=window, min_count=min_count, workers=workers, alpha=alpha, negative=negative, hs=hs, sample=sample)
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

            tokens = generate_duplex_tokens(top, bottom, self.n)
            f.write(" ".join(tokens) + "\n")
    
  def encode(self, top: str, bottom: str) -> np.ndarray:
    seq = top.strip().upper() 
    c_seq = bottom.strip().upper()

    if not is_valid_dna(top) or not is_valid_dna(bottom):
       raise ValueError(f"Encode: Invalid DNA Sequence: {seq}-{c_seq}")
    if len(seq) != len(c_seq):
       raise ValueError("Encode: Top and Bottom must be of the same length.")

    tokens = generate_duplex_tokens(seq, c_seq, self.n)
    vecs = []
    for token in tokens:
      try:
        vecs.append(self.wv[token]) # If token is in embedding vector retrieve its value.
      except:
         raise Exception("Model has never trained this token: " + token)

    return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size) # Note: Try sum vs mean
    
  def batch_encode(self, df: pd.DataFrame, verbose: bool = False, error_log_path: Optional[str] = None) -> np.ndarray:
    if verbose and not error_log_path:
      raise ValueError("Batch Encode: Error logging path not specified.")
    
    log: Optional[TextIO] = None
    if verbose:
      assert error_log_path is not None
      log = open(error_log_path, 'w')
      assert log is not None
      log_file = log
      log_file.write("Failed to encode...\nIdx Top Bottom Error\n")
    
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
  