import random
import csv
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

def random_dna(length=8):
    return ''.join(random.choice('ATCG') for _ in range(length))

def introduce_mismatches(seq, num_mismatches=1):
    seq = list(seq)
    positions = random.sample(range(len(seq)), num_mismatches)
    for pos in positions:
        original = seq[pos]
        options = list(set('ATCG') - set(original))
        seq[pos] = random.choice(options)
    return ''.join(seq)

def generate_sequence_pairs(n=1000):
    data = []
    for _ in range(n):
        seq1 = random_dna(8)
        seq2_perfect = str(Seq(seq1).reverse_complement())

        #perfect match
        data.append((seq1, seq2_perfect, "perfect"))

        #single mismatch
        seq2_single = introduce_mismatches(seq2_perfect, 1)
        data.append((seq1, seq2_single, "single_mismatch"))

        #doublexs mismatch
        seq2_double = introduce_mismatches(seq2_perfect, 2)
        data.append((seq1, seq2_double, "double_mismatch"))
    return data

def bucket_tm(tm, size=5):
    return size * round(tm / size)

def main():
    sequence_data = generate_sequence_pairs(500)  
    bucket_size = 5

    with open('sequence_tm_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Seq1', 'Seq2', 'Type', 'MeltingTemp', f'Tm_Bucket_{bucket_size}C'])
        
        for seq1, seq2, label in sequence_data:
            try:
                tm = mt.Tm_NN(Seq(seq1), c_seq=Seq(seq2))
                bucket = bucket_tm(tm, size=bucket_size)
                writer.writerow([seq1, seq2, label, round(tm, 2), bucket])
            except Exception as e:
                print(f"Error computing Tm for {seq1}, {seq2}: {e}")

if __name__ == "__main__":
    main()
