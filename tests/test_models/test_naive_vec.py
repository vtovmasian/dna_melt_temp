import pytest
import os
import pandas as pd
from biovec.models.naive_vec import split_ngrams, NaiveVec


def test_split_ngrams():
  seq = "ATTGG"
  ngrams = split_ngrams(seq, n=3)
  expected = ["ATT", "TTG", "TGG"]
  assert ngrams == expected


def test_naive_encode():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "GCG", "TAC"]
  })

  model = NaiveVec(df, n=3, size=64)

  vecs = model.batch_encode(df)
  assert vecs.shape == (3, 128)


def test_naive_encode_allows_mismatched_lengths():
  df = pd.DataFrame({
    "Top": ["ATA", "CGCAA", "ATG"],
    "Bottom": ["TAT", "GCG", "TACGG"]
  })

  model = NaiveVec(df, n=3, size=64)

  vec = model.encode("ATA", "TACGG")
  assert vec.shape == (128,)


def test_naive_encode_log_error():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "GCG", "TAC"]
  })

  model = NaiveVec(df, n=3, size=64)

  log_path = "test_naive_error_log.txt"
  df_invalid = pd.DataFrame({
    "Top": ["ATA", "#!@", "1234"],
    "Bottom": ["TAT", "GCG", "TAC"]
  })
  vecs = model.batch_encode(df_invalid, verbose=True, error_log_path=log_path)
  assert vecs.shape == (1, 128), "Failed, output shape mismatch."
  assert os.path.exists(log_path), "Failed, log file does not exist."
  HEADER_LINES = 2
  INVALID_SEQUENCES = 2
  with open(log_path, 'r') as f:
    lines = f.readlines()

  assert len(lines) == INVALID_SEQUENCES + HEADER_LINES, "Failed, expected two errors."
