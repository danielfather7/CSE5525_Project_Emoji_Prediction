import os
import sys

#Sparse matrix implementation
from scipy.sparse import csr_matrix
from Vocab import Vocab
import numpy as np
from collections import Counter

#np.random.seed(1)

class EMOJIdata:
    def __init__(self, textfiledir, vocab=None):
        """ Reads in data into sparse matrix format """
        textFile = os.path.abspath(textfiledir)
        labelFile = textFile.replace(".text", ".labels")

        if not vocab:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        #For csr_matrix (see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        X_values = []
        X_row_indices = []
        X_col_indices = []
        Y = []

        #Read positive files
        i=0
        for line in open(textFile):
            wordCounts = Counter([self.vocab.GetID(w.lower()) for w in line.split()])
            for (wordId, count) in wordCounts.items():
                if wordId >= 0:
                    X_row_indices.append(i)
                    X_col_indices.append(wordId)
                    X_values.append(count)
            i = i+1
        
        for line in open(labelFile):
            Y.append(int(line))
        self.mostFreqLbl = Counter(Y).most_common(1)[0][0]
            
        self.vocab.Lock()

        #Create a sparse matrix in csr format
        self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.GetVocabSize()))
        self.Y = np.array(Y)

        #Randomly shuffle
        index = np.arange(self.X.shape[0])
        
        np.random.shuffle(index)
        self.X = self.X[index,:]
        self.Y = self.Y[index]

if __name__ == "__main__":
    data = EMOJIdata("./data/us_train.text")
    print(data.X.toarray())
    print(data.Y)
    print(data.mostFreqLbl)
