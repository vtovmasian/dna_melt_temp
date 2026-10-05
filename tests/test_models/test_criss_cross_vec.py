import pytest
import os
import pandas as pd
from biovec.models.criss_cross_vec import generate_criss_cross_tokens, CrissCrossVec


def test_generate_criss_cross_tokens():
  top = "ACGTA"
  bottom = "TGCAT"
  tokens = generate_criss_cross_tokens(top, bottom)
  expected = ["AT", "CG", "GC", "TA", "AT"]
  assert tokens == expected


def test_generate_criss_cross_tokens_captures_mismatch():
  # Position 3 (0-indexed) is a G/A mismatch instead of the WC pair G/C.
  top = "ACGTA"
  bottom = "TGAAT"
  tokens = generate_criss_cross_tokens(top, bottom)
  expected = ["AT", "CG", "GA", "TA", "AT"]
  assert tokens == expected


def test_generate_criss_cross_tokens_windowed_n2():
  top = "ACGTA"
  bottom = "TGCAT"
  tokens = generate_criss_cross_tokens(top, bottom, n=2)
  expected = ["ATCG", "CGGC", "GCTA", "TAAT"]
  assert tokens == expected


def test_generate_criss_cross_tokens_windowed_n3():
  top = "ACGTA"
  bottom = "TGCAT"
  tokens = generate_criss_cross_tokens(top, bottom, n=3)
  expected = ["ATCGGC", "CGGCTA", "GCTAAT"]
  assert tokens == expected


def test_criss_cross_encode():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "GCG", "TAC"]
  })

  model = CrissCrossVec(df, size=64)

  vecs = model.batch_encode(df)
  assert vecs.shape == (3, 64)


def test_criss_cross_default_n_is_3():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "GCG", "TAC"]
  })

  model = CrissCrossVec(df, size=64)
  assert model.n == 3


def test_criss_cross_encode_log_error():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "GCG", "TAC"]
  })

  model = CrissCrossVec(df, size=64)

  log_path = "test_criss_cross_error_log.txt"
  df_invalid = pd.DataFrame({
    "Top": ["ATA", "CGC", "1234"],
    "Bottom": ["TAT", "#!@", "TAC"]
  })
  vecs = model.batch_encode(df_invalid, verbose=True, error_log_path=log_path)
  assert vecs.shape == (1, 64), "Failed, output shape mismatch."
  assert os.path.exists(log_path), "Failed, log file does not exist."
  HEADER_LINES = 2
  INVALID_SEQUENCES = 2
  with open(log_path, 'r') as f:
    lines = f.readlines()

  assert len(lines) == INVALID_SEQUENCES + HEADER_LINES, "Failed, expected two errors."
