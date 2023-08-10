import  random
import  torch
import  dataloaders
import  utils
from    torch.utils.data           import Dataset


class Seq2SeqDataset_stratified(Dataset):
    """
    This is a PyTorch Dataset class is used for loading the data in a format that can be fed into a
    sequence-to-sequence model.

    The source and target sentences are then retrieved from the list of sentence pairs, split into
    individual words, and converted to tensors. These tensors are returned along with their lengths 
    as the output of the function.
    """
    def __init__(self, source, target, pairs, Mode):
        """
            The __init__ function initializes the dataset with the given source and target languages, as well
            as a list of sentence pairs. The Mode argument determines whether the dataset is being used for
            training, validation, or testing, while split_ratio specifies the proportion of data to be used
            for training.
        """
        self.Mode = Mode
        self.source, self.target, self.pairs = source, target, pairs
        self.counter=1
        
        
        ###########TODO move these to utils
        ## hyperparam
        self.Train          = utils.train_split
        self.Validation     = utils.Validation_split
        self.Test           = 1 - self.Train - self.Validation 
        
    def __len__(self):
        """
            The __len__ function returns the length of the dataset, which is the number of sentence pairs.

        """
        if self.Mode =='Train':
            return int(len(self.pairs)*self.Train       )
        if self.Mode =='Validation':
            return int(len(self.pairs)*self.Validation  )
        elif self.Mode =='Test':
            return int(len(self.pairs)*self.Test        )

    def __getitem__(self, idx):
        """
            The __getitem__ function retrieves a specific sentence pair based on the given index idx. It 
            first calculates the appropriate start and end indices based on the current mode and the length
            of the target sentence. It then randomly selects an index within this range from the list of
            indices corresponding to the current length of the target sentence.
        """
        
        period = self.counter%len(dataloaders.sorted__keies__of__the__target_dic_index)
        
        if self.Mode=='Train':
            self.start   = 0
            self.end     = int(dataloaders.sorted__lenght__of__the__target_dic_index[period]*self.Train)-1
            
        elif self.Mode=='Validation':
            self.bias    = int(dataloaders.sorted__lenght__of__the__target_dic_index[period]*self.Train)
            self.start   = self.bias
            self.end     = self.bias + int(dataloaders.sorted__lenght__of__the__target_dic_index[period]*self.Validation)-1
            
        elif self.Mode=='Test':
            self.bias    = int(dataloaders.sorted__lenght__of__the__target_dic_index[period]*(self.Validation+self.Train))
            self.start   = self.bias
            self.end     = dataloaders.sorted__lenght__of__the__target_dic_index[period]-1

            
        idx = dataloaders.target_dic_index[dataloaders.sorted__keies__of__the__target_dic_index[period]][random.randint(self.start,self.end)]
        
          
        source_local, target_local = self.pairs[idx]
        source_local=source_local.split(" ")
        target_local=target_local.split(" ")
        source = [self.source.word2index[y] for x,y in enumerate(source_local)]
        target = [self.target.word2index[y] for x,y in enumerate(target_local)]
        source = torch.tensor(source, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
       
        self.counter+=1    
        return source, target
    
    def reset(self,
              ):
        """
            Finally, the reset function resets the counter variable to 0, which is useful when using the
            dataset in multiple epochs. 
        """
        self.counter = 0

def collate_fn2(batch):
    """
    This is a function for collating a batch of variable-length sequences into a single tensor, which is
    useful for training a neural network with PyTorch.

    The input to this function is a batch of samples, each containing a source and target sequence. 
    The function extracts the source and target sequences from each sample, and then pads them to ensure
    that all sequences in the batch have the same length. This is necessary because PyTorch requires all
    inputs to a neural network to have the same shape.

    The function uses the PyTorch pad_sequence function to pad the sequences. pad_sequence is called with
    the batch_first=True argument to ensure that the batch dimension is the first dimension of the output
    tensor. The padding_value argument is set to 0 to pad with zeros.

    The function returns the padded source and target sequences as a tuple.
    """
    sources = [item[0] for item in batch]
    targets = [item[1] for item in batch]
              
    sources = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return sources, targets


def collate_fn(batch):
    # sources = [item[0] for item in batch]
    # targets = [item[1] for item in batch]
    
    # but the tensor size is max_len plus one
    max_len_en = utils.MAX_LENGTH_Lang1+2
    max_len_sp = utils.MAX_LENGTH_Lang2+2
    # Pad sequences to the maximum length
    sources = torch.zeros(size=[len(batch),max_len_en])
    targets = torch.zeros(size=[len(batch),max_len_sp])
    
    sources[:,0] = 1
    targets[:,0] = 1
    
    i=0
    for (source,target) in batch:
        sources[i,1:] = torch.nn.functional.pad(source, (0, max_len_en -1 - len(source)))
        targets[i,1:] = torch.nn.functional.pad(target, (0, max_len_sp -1  - len(target)))
        
        sources[i,len(source)+1] = 2
        targets[i,len(target)+1] = 2
        i+=1
    return sources, targets



class Seq2SeqDataset(Dataset):

    def __init__(self, source, target, pairs, Mode):

        self.Mode = Mode
        self.source, self.target, self.pairs = source, target, pairs
        self.counter=1
        
        self.Train          = utils.train_split
        self.Validation     = utils.Validation_split
        self.Test           = 1 - self.Train - self.Validation 
        
        self.Allowed = []
        for i,ii in enumerate(iterable = dataloaders.sorted__keies__of__the__target_dic_index,start=0):
            
            if self.Mode=='Train':
                Variable = self.Train
                self.start   = 0
                self.end     = int(dataloaders.sorted__lenght__of__the__target_dic_index[i]*self.Train)-1
                
            elif self.Mode=='Validation':
                self.bias    = int(dataloaders.sorted__lenght__of__the__target_dic_index[i]*self.Train)
                self.start   = self.bias
                self.end     = self.bias + int(dataloaders.sorted__lenght__of__the__target_dic_index[i]*self.Validation)-1
                
            elif self.Mode=='Test':
                self.bias    = int(dataloaders.sorted__lenght__of__the__target_dic_index[i]*(self.Validation+self.Train))
                self.start   = self.bias
                self.end     = dataloaders.sorted__lenght__of__the__target_dic_index[i]-1               

            self.Allowed.extend(dataloaders.target_dic_index[ii][self.start:self.end])
    
        
    def __len__(self):
        if self.Mode=='Train':
            return int(len(self.pairs)*self.Train)
        if self.Mode =='Validation':
            return int(len(self.pairs)*self.Validation  )
        elif self.Mode=='Test':
            return int(len(self.pairs)*self.Test )

    def __getitem__(self, idx):
        period = self.counter%len(self.Allowed)
       
        source_local, target_local = self.pairs[period]
        source_local=source_local.split(" ")
        target_local=target_local.split(" ")
        source = [self.source.word2index[y] for x,y in enumerate(source_local)]
        target = [self.target.word2index[y] for x,y in enumerate(target_local)]
        source = torch.tensor(source, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
       
        self.counter+=1    
        return source, target
