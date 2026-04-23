# -*- coding: utf-8 -*-

from nose.tools import eq_, ok_
import biovec
import tempfile
import os
import numpy


class TestProtVec():
    def setUp(self):
        self.FASTA_PATH = 'some_fasta_file.fasta'
        with open(self.FASTA_PATH, 'w') as f:
            f.write('>sample_record1\nGSRATATQSQATGVLSMTIMEELP')
        self.pv = biovec.models.ProtVec(self.FASTA_PATH, corpus_fname="output_corpusfile_path.txt")

    def test_init_model(self):
        ok_(biovec.models.ProtVec(self.FASTA_PATH, corpus_fname="output_corpusfile_path.txt"))

    def test_indicing(self):
        # The n-gram "QAT" should be trained in advance
        eq_(
            self.pv["QAT"].shape,
            (100,)
        )

    def test_to_vecs(self):
        # convert whole amino acid sequence into vector
        input_seq = 'QSQATGVLS'
        eq_(
            self.pv.to_vecs(input_seq).shape,
            (3,100)
        )

        input_seq_2 = 'TGV'

        if len(input_seq_2) >=3 and len(input_seq_2) < 5:
            numpy.testing.assert_almost_equal(
                self.pv.to_vecs(input_seq_2)[-(5-len(input_seq_2)):],
                numpy.zeros([5-len(input_seq_2), 100])
            )
    
    def test_multiseq_to_vecs(self):
        # read a multi-sequence FASTA file and output a numpy array with biovec representation for each sequence
        self.FASTA_PATH_2 = 'testing_multi_fasta_file.fasta'

        LIST_OF_ACCESSION = ['sample_record1','sample_record2','sample_record3','sample_record4','sample_record5','sample_record6']
        LIST_OF_SEQUENCES = ['GSRATATQSQATGVLSMTIMEELP','GSRATATQSQATGVLSMTIMEELP*','GSRATATQSQATGVLSMTIMEELPX','GSRATATQSQ\nATGVLSMTIMEELP','GSRATATQSQATGVLSMTIMEELPATATQS','GSRATATQSQATGVLSMTIMEELPU-X']

        with open(self.FASTA_PATH_2, 'w') as f2:
            for n in range(len(LIST_OF_ACCESSION)):
                f2.write('>' + LIST_OF_ACCESSION[n] + '\n' + LIST_OF_SEQUENCES[n] + '\n')
        
        multiseq_output = self.pv.multiseq_to_vecs(self.FASTA_PATH_2, 'test_multiseq')

        eq_(
            multiseq_output.shape,
            (3,3,100)
            )

        numpy.testing.assert_almost_equal(
            multiseq_output[0],
            self.pv.to_vecs('GSRATATQSQATGVLSMTIMEELP')
        )
        # sequence with * at stop
        numpy.testing.assert_almost_equal(
            multiseq_output[0],
            multiseq_output[1]
        )

        # sequence spanning multiple lines
        numpy.testing.assert_almost_equal(
            multiseq_output[0],
            multiseq_output[2]
        )

        
        output_seq_count = len(open('test_multiseq_index.tsv').readlines())

        eq_(
            output_seq_count,
            3 + 1  
        ) # header line --> +1

        failed_seq_count = len(open('test_multiseq_failed.tsv').readlines())

        eq_(
            failed_seq_count,
            3 + 1
        ) # header line --> +1


    def test_save_and_load(self):
        f = tempfile.NamedTemporaryFile()
        saved_model_path = f.name
        # save trained model into file
        self.pv.save(saved_model_path)
        eq_(
            os.path.exists(saved_model_path),
            True
        )

        # load trained model from file
        ok_(biovec.models.load_protvec(saved_model_path))
        f.flush()
