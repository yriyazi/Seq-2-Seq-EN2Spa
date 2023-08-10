import  dataloaders
import  utils
import  numpy       as      np
import  pandas      as      pd


def read_dataset(dataset_path:str=utils.dataset_path,
                 names:list = [utils.Language1,	utils.Language2,	'etc']):
    """
    reading the datas

    """
    data_frame = pd.read_csv(dataset_path, sep="\t",names=names)
    del data_frame[names[-1]]
    return data_frame

def normalize_sentence(df:pd.core.frame.DataFrame,
                       Language:str):
    """    
    The normalize_sentence function takes a dataframe df and a Container class as input and normalizes the sentences 
    in the given language. 
    -------------------------------
        * The function converts the sentences to lowercase 
        * removes any non-alphabetic characters using a regular expression
        * It then normalizes the text to Unicode NFD (Normalization Form Decomposition) 
        and encodes it to ASCII to remove any non-ASCII characters
        * it decodes the text back to UTF-8 format and returns the normalized sentence.
    -------------------------------    
    Overall, this function is useful for standardizing the text and removing any unwanted characters that may 
    interfere with downstream natural language processing tasks such as text classification or sentiment analysis.
    However, it is important to note that this normalization technique may not be suitable for all languages or text
    types, and more advanced normalization techniques may be required in some cases.
    -------------------------------
    this class is inspiered from prepareData & filterpair in 
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html & 
    https://www.guru99.com/seq2seq-model.html
    """
    sentence = df[Language].str.lower()
    sentence = sentence.str.replace('[^A-Za-z\s]+', '',regex=True)
    sentence = sentence.str.normalize('NFD')
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence


def read_sentence(df:pd.core.frame.DataFrame,
                  Language1:str = utils.Language1,
                  Language2:str = utils.Language2,):
    """it just call the Normalize method on the input and out put language sepratly"""
    Column1 = normalize_sentence(df, Language1)
    Column2 = normalize_sentence(df, Language2)
    return Column1, Column2


def process_data(df:pd.core.frame.DataFrame,
                 Language1:str = utils.Language1,
                 Language2:str = utils.Language2,
                 MAX_LENGTH:int = utils.MAX_LENGTH):
    """
        Main tokenizer funtion 
        
        ---> in case of future changes in the Container remember to to change the class in here to 
    """
    print("Read %s sentence pairs" % len(df))
    sentence1, sentence2 = read_sentence(df, Language1, Language2)

    source = dataloaders.WordEmbeddings_English()
    target = dataloaders.Countainer()
    pairs = []
    for i in range(100000):#len(df)
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]
            source.addSentence(sentence1[i])
            target.addSentence(sentence2[i])
            pairs.append(full)

    return source, target, pairs


############################################################################################################

# defing the class of words and words pair
data_frame = read_dataset()#'dataset/Eng_Spa.txt'

source, target, pairs = process_data(data_frame,
                                    Language1 = utils.Language1,
                                    Language2 = utils.Language2,
                                    MAX_LENGTH = utils.MAX_LENGTH)

"""
This code is creating a list of lengths for the source and target languages in a given dataset,
and also creating a dictionary that maps the length of the target language to the indices of the
corresponding length.

To achieve this, the code iterates through the pairs of data in the dataloader and computes the 
lengths of the source and target languages by splitting the strings by whitespace.
Then, it appends these lengths to the respective source and target length lists.

The code then creates a target_dic_index dictionary where the keys are the lengths of the target
language and the values are lists of indices of the pairs that have that corresponding length.
If a key already exists in the dictionary, then the index of the current pair is 
appended to the existing list. 
Otherwise, a new key-value pair is created in the dictionary with the length as the key and a 
list containing the current index as the value.

Finally, the source and target length lists are converted into numpy arrays.
"""
source_lenght = []
target_lenght = []
target_dic_index ={}

for iter,(start,desti) in enumerate(pairs):
    source_lenght.append(len(start.split(" ")))
    target_lenght.append(len(desti.split(" ")))
    
    try :
        target_dic_index[len(desti.split(" "))].append(iter)
    except:
        target_dic_index[len(desti.split(" "))]=[iter]

    
source_lenght = np.array(source_lenght)
target_lenght = np.array(target_lenght)

"""
This code is refining the dataset by finding a suitable length for the target language sentences 
and deleting the shorter sentences.

First, the code retrieves the sorted keys of the target_dic_index dictionary and stores them in 
sorted__keies__of__the__target_dic_index. 
It also initializes an empty list sorted__lenght__of__the__target_dic_index and a list dump to 
store the allowed indices.

Next, the code iterates through the sorted__keies__of__the__target_dic_index list and checks 
if the number of indices corresponding to a particular key in target_dic_index is greater than or 
equal to 12. If it is, then the code appends the number of indices to the 
sorted__lenght__of__the__target_dic_index list and adds the corresponding key to the dump list.

After the iteration is complete, the sorted__keies__of__the__target_dic_index list is updated 
to contain only the keys that correspond to the indices in the dump list. The dummy and dump 
variables are then deleted to free up memory.

Overall, this code is useful for ensuring that the dataset contains only a certain length of 
target language sentences, and can help to improve the quality of the data for downstream tasks.
"""
sorted__keies__of__the__target_dic_index    = sorted(list(target_dic_index.keys()))
sorted__lenght__of__the__target_dic_index   = []
dump = []
for itter,allowed__index in enumerate(sorted__keies__of__the__target_dic_index):
    dummy = len(target_dic_index[allowed__index])
    if (dummy) >= 12:
        sorted__lenght__of__the__target_dic_index.append(dummy)
        dump.append(allowed__index)
sorted__keies__of__the__target_dic_index = dump
del dummy , dump