import pytest
import os
import pandas as pd
from biovec.models.mismatch_vec import to_mismatch_symbols, generate_mismatch_ngrams, MismatchVec


def test_to_mismatch_symbols_perfect_complement():
  top = "ACGTA"
  bottom = "TGCAT"
  symbols = to_mismatch_symbols(top, bottom)
  assert symbols == ["A", "C", "G", "T", "A"]


def test_to_mismatch_symbols_with_mismatch():
  # Position 2 (0-indexed) is a G/A mismatch instead of the WC pair G/C.
  top = "ACGTA"
  bottom = "TGAAT"
  symbols = to_mismatch_symbols(top, bottom)
  assert symbols == ["A", "C", "GA", "T", "A"]


def test_generate_mismatch_ngrams():
  top = "ACGTA"
  bottom = "TGAAT"
  tokens = generate_mismatch_ngrams(top, bottom, n=3)
  expected = ["ACGA", "CGAT", "GATA"]
  assert tokens == expected


def test_mismatch_encode():
  df = pd.DataFrame({
    "Top": ["ATAG", "CGCA", "ATGC"],
    "Bottom": ["TATC", "GCGT", "TACG"]
  })

  model = MismatchVec(df, n=3, size=64)

  vecs = model.batch_encode(df)
  assert vecs.shape == (3, 64)


def test_mismatch_encode_log_error():
  df = pd.DataFrame({
    "Top": ["ATAG", "CGCA", "ATGC"],
    "Bottom": ["TATC", "GCGT", "TACG"]
  })

  model = MismatchVec(df, n=3, size=64)

  log_path = "test_mismatch_error_log.txt"
  df_invalid = pd.DataFrame({
    "Top": ["ATAG", "CGCA", "1234"],
    "Bottom": ["TATC", "#!@", "TACG"]
  })
  vecs = model.batch_encode(df_invalid, verbose=True, error_log_path=log_path)
  assert vecs.shape == (1, 64), "Failed, output shape mismatch."
  assert os.path.exists(log_path), "Failed, log file does not exist."
  HEADER_LINES = 2
  INVALID_SEQUENCES = 2
  with open(log_path, 'r') as f:
    lines = f.readlines()

  assert len(lines) == INVALID_SEQUENCES + HEADER_LINES, "Failed, expected two errors."
