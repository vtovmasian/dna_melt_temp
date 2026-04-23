from gensim.models import word2vec
from biovec.utils import split_ngrams, generate_corpusfile
import numpy as np
import re


def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)


class ProtVec(word2vec.Word2Vec):
    def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=3):
        """
        Either fname or corpus is required.

        fasta_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        corpus_fname: corpus file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """
        self.n = n
        self.size = size
        self.fasta_fname = fasta_fname
        if corpus is None and fasta_fname is None:
            raise Exception("Either fasta_fname or corpus is needed!")
        if fasta_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(fasta_fname, n, corpus_fname)
            corpus = word2vec.Text8Corpus(corpus_fname)
        word2vec.Word2Vec.__init__(self, corpus, vector_size=size, sg=sg, window=window, min_count=min_count, workers=workers)

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        seq = seq.replace("*","") # for sequences with * to indicate stop
        ngram_patterns = split_ngrams(seq, self.n)
        protvecs = np.zeros([3,self.size]) # numpy for easy integration with ML libraries
        for num in range(3):
            ngram_vecs = []
            for ngram in ngram_patterns[num]:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("Model has never trained this n-gram: " + ngram)
            protvecs[num,] = sum(ngram_vecs)
        return protvecs
    
    def multiseq_to_vecs(self, inputfastafilename, outputfilename, seqtype = 'amino acid'):
        """
        convert a multi-sequence fasta file to a numpy array of protvec arrays
        with axis 0 being the sample axis

        seqtype = 'nucleotide' or 'amino acid'
        nucleotide and aa codes from https://www.genome.jp/kegg/catalog/codes1.html
        """
        try:
            sequencefile = open(inputfastafilename, 'r')
        except:
            raise Exception("cannot open the input FASTA file")
        seqDict = {}
        
        for line in sequencefile:
            line = line.strip()
            if len(line) > 0 : 
                if line[0] == '>':
                    accession = str(line[1:]).replace(" ","_")
                    if accession not in seqDict:
                        seqDict[accession] = ''
                    else:
                        raise KeyError('Duplicate record present for ',accession)
                else:
                    line = line.replace(" ","")

                    if seqtype == 'amino acid':
                        if not re.search('[^ARNDCQEGHILKMFPSTWYVBZJUOX]',line.upper()):
                            seqDict[accession] += line.upper()
                        else:
                            seqDict[accession] += line.upper()
                            print('Caution: Illegal character in the sequence',accession )
                    elif seqtype == 'nucleotide':
                        if not re.search('[^AGCTURYNWSMKBHDV]',line.upper()):
                            seqDict[accession] += line.upper()
                        else:
                            seqDict[accession] += line.upper()
                            print('Caution: Illegal character in the sequence',accession )

        multiseq_protvec = np.empty([len(seqDict), 3,self.size])
        accessionlist = list(seqDict.keys())
        accessionlist.sort()
        print('Number of sequences in the input file : ', len(accessionlist))
        indextrack = 0
        outputindexfile = open(outputfilename + '_index.tsv', 'w')
        outputindexfile.write('index' + '\t' + 'accession' + '\t' + 'sequence' + '\n')

        outputlogfile = open(outputfilename + '_failed.tsv', 'w')
        outputlogfile.write('failed accession' + '\t' + 'sequence' + '\n')

        for number in range(len(accessionlist)):
            try:
                if seqDict[accessionlist[number]][-1] == '*':
                    seqDict[accessionlist[number]] = seqDict[accessionlist[number]][:-1]
                accessionarray = self.to_vecs(seqDict[accessionlist[number]])
            except:
                print('cannot output protvec representation for ',str(accessionlist[number]))
                outputlogfile.write(str(accessionlist[number]) + '\t' + str(seqDict[accessionlist[number]]) + '\n')
                continue
            outputindexfile.write(str(indextrack) + '\t' + str(accessionlist[number]) + '\t' + str(seqDict[accessionlist[number]]) + '\n')
            multiseq_protvec[indextrack,] = accessionarray
            indextrack += 1
        
        if indextrack < len(seqDict):
            multiseq_protvec = multiseq_protvec[0:indextrack,]
        
        outputindexfile.close()
        outputlogfile.close()
        del seqDict
        del accessionarray
        del indextrack
        del accessionlist

        return multiseq_protvec