# import os 
import numpy as np

class WordEmbeddings_English():
    def __init__(self,
                 embedding_file:str = './Glove/glove.6B.100d.txt'):
        """
        A class that reads a GloVe text file and performs index preprocessing.

        Args:
            embedding_file (str): Path to the GloVe embedding file.

        Attributes:
            word2idx (dict):        Dictionary mapping words to indices.
            index2word (dict):      Dictionary mapping indices to words.
            n_words (int):          Total number of words in the vocabulary.
            embeddings (ndarray):   Matrix storing the word embeddings.
            preproce (bool):        Flag indicating if preprocessing has been performed.

        """
        self.word2index   = {}
        self.index2word = {}
        self.n_words    = 400001
        self.embeddings = np.zeros((400001, 100), dtype=np.float32)
        self.load_embeddings(embedding_file)
        self.preproce = False
        
        if self.preproce == False:
            self.Classic_index()
            self.pad()
            self.preproce =True
            
    def addSentence(self, sentence):
        """
        Splits the sentence into individual words and adds each word to the container using the addWord() method.

        Args:
            sentence (str): The input sentence.

        """
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word]= 201535


        
    def load_embeddings(self, embedding_file):
        """
        Reads the GloVe file and generates the embedding matrix.

        Args:
            embedding_file (str): Path to the GloVe embedding file.

        """
        index = 1
        with open(embedding_file, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                embedding = np.asarray(line[1:], dtype='float32')
                
                self.word2index[word] = index
                self.index2word[index] = word
                
                self.embeddings[index] = embedding
                index +=1
                
    def get_embedding(self,
                      word:str):
        """
        Searches for the embedding of a given word and returns the corresponding embedding.
        If the word does not exist, it returns the embedding for the "unk" token.
        
        >>>self.word2index["unk"]
            201535

        Args:
            word (str): The word to retrieve the embedding for.

        Returns:
            ndarray: The embedding vector for the given word.
        """
        if word in self.word2index:
            return self.embeddings[self.word2index[word]]
        else:
            
            return self.embeddings[201535]
        
    def Classic_index(self):
        """
        Reorders the indices for the 'sos' and 'eos' tokens.
        """
        self._swapper('sos',1)
        self._swapper('eos',2)
        
    def pad(self):
        """
        Adds the padding token to the vocabulary.
        """
        self.word2index['<pad>']  = 0
        self.index2word[0]   = '<pad>'
        
        
    def _swapper(self,
                Word:int,
                index_that_will_be:str,):
        """
        Swaps the indices and embeddings of two words.

        Args:
            Word (int): The index of the word to swap.
            index_that_will_be (str): The index that will be assigned to the word.

        """
        
        second_index ,second_word = self.word2index[Word],self.index2word[index_that_will_be]
        
        self.embeddings[index_that_will_be], self.embeddings[second_index]  = self.embeddings[second_index] , self.embeddings[index_that_will_be]
        self.word2index[second_word]       , self.word2index[Word]          = self.word2index[Word]         , index_that_will_be
        self.index2word[index_that_will_be],  self.index2word[second_index] = Word                          , second_word
