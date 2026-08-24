import pytest
import os
import pandas as pd
from biovec.models.duplex_vec import generate_duplex_tokens, DuplexVec

def test_generate_duplex_tokens():
  seq = "ATTGG"
  c_seq = "TAACC"
  tokens = generate_duplex_tokens(seq, c_seq, n=3)
  expected = ["ATT-TAA", "TTG-AAC", "TGG-ACC"]
  assert tokens == expected


def test_duplex_encode():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "GCG", "TAC"] 
  })

  model = DuplexVec(df, n=3, size=64)

  vecs = model.batch_encode(df)
  assert vecs.shape == (3, 64)


def test_duplex_encode_log_error():
  df = pd.DataFrame({
    "Top": ["ATA", "CGC", "ATG"],
    "Bottom": ["TAT", "#!@", "1234"]
  })

  model = DuplexVec(df, n=3, size=64)

  log_path = "test_error_log.txt"
  vecs = model.batch_encode(df, verbose=True, error_log_path=log_path)
  assert vecs.shape == (1, 64), "Failed, output shape mismatch."
  assert os.path.exists(log_path), "Failed, log file does not exist."
  HEADER_LINES = 2
  INVALID_SEQUENCES = 2
  with open(log_path, 'r') as f:
    lines = f.readlines()
  
  assert len(lines) == INVALID_SEQUENCES + HEADER_LINES, "Failed, expected two errors."


def test_duplex_encode_invalid_dna():
  df = pd.DataFrame({
    "Top": [" ", "SAK", "ATG"],
    "Bottom": ["TAT", "#!@", "1234"]
  })

  model = DuplexVec(df, n=3, size=64)

  log_path = "test_error_log.txt"
  vecs = model.batch_encode(df, verbose=True, error_log_path=log_path)
  assert vecs.shape == (0, 64), "Failed, output shape mismatch."
  assert os.path.exists(log_path), "Failed, log file does not exist."
  HEADER_LINES = 2
  INVALID_SEQUENCES = 3
  with open(log_path, 'r') as f:
    lines = f.readlines()
  
  assert len(lines) == INVALID_SEQUENCES + HEADER_LINES, "Failed, expected two errors."
  
