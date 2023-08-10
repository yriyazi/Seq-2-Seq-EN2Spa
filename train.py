import  dataloaders
import  Model,nets
import  deeplearning
import  utils
import  torch
import  pandas                      as      pd
import  numpy                       as      np
from    torch.utils.data            import  DataLoader
device = 'cuda'
model       = nets.Seq2Seq(device=device,Num_layer=utils.Hidden_layer)

#%% Warm Start
warm_start = False
if warm_start == True:
    
    train_dataset=dataloaders.Seq2SeqDataset(dataloaders.source, dataloaders.target, dataloaders.pairs,Mode='Train')
    test_dataset=dataloaders.Seq2SeqDataset(dataloaders.source, dataloaders.target, dataloaders.pairs,Mode='Test')

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False, collate_fn=dataloaders.collate_fn)
    valid_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=dataloaders.collate_fn)
    
    model, optimizer, report = deeplearning.train(      train_loader    = train_dataloader,
                                                    val_loader      = valid_dataloader ,
                                                    model           = model,
                                                    model_name      = 'seq2seq_L2_000_001',
                                                    epochs          = 1,
                                                    learning_rate   = 0.001,
                                                    device          = device,
                                                    load_saved_model= False,
                                                    ckpt_save_freq  = 1,
                                                    ckpt_save_path  = './Model/.Checkpoints',
                                                    ckpt_path       = './Model/.Checkpoints/ckpt_seq2seq_000_001_epoch1.ckpt',
                                                    report_path     = './Model/',
                                                    
                                                    test_ealuate    = False,
                                                    tets_loader     = None,
                                                    total_iters     = 1)

#%% Fine Tuning

train_dataset=dataloaders.Seq2SeqDataset_stratified(dataloaders.source, dataloaders.target, dataloaders.pairs,Mode='Train')
valid_dataset=dataloaders.Seq2SeqDataset_stratified(dataloaders.source, dataloaders.target, dataloaders.pairs,Mode='Validation')

train_dataloader = DataLoader(train_dataset, batch_size=utils.batch_train_size      , shuffle=True, collate_fn=dataloaders.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=utils.batch_validation_size , shuffle=True, collate_fn=dataloaders.collate_fn)

model, optimizer, report = deeplearning.train(      train_loader    = train_dataloader,
                                                    val_loader      = valid_dataloader ,
                                                    model           = model,
                                                    model_name      = 'ckpt_seq2seq_L2_101_121',
                                                    epochs          = 20,
                                                    learning_rate   = utils.learning_rate,
                                                    device          = device,
                                                    load_saved_model = True,
                                                    ckpt_save_freq  = 20,
                                                    ckpt_save_path  = './Model/.Checkpoints',
                                                    ckpt_path       = './Model/.Checkpoints/ckpt_ckpt_seq2seq_L2_081_101_epoch20.ckpt',
                                                    report_path     = './Model/',
                                                    
                                                    test_ealuate    = False,
                                                    tets_loader     = None,
                                                    total_iters     =1)

#%% Plotting the Results
import pandas as pd
import numpy as np
import utils


_0=pd.read_csv('Model/seq2seq_L2_000_001_report.csv')

_1=pd.read_csv('Model/ckpt_seq2seq_L2_001_021.csv')
_2=pd.read_csv('Model/ckpt_seq2seq_L2_021_041_report.csv')
_3=pd.read_csv('Model/ckpt_seq2seq_L2_041_061_report.csv')
_4=pd.read_csv('Model/ckpt_seq2seq_L2_061_081_report.csv')
_5=pd.read_csv('Model/ckpt_seq2seq_L2_081_101_report.csv')
df=pd.concat([_1, _2,_3,_4,], ignore_index=True)

def accu_2_int(df:pd.DataFrame,
               column:str='avg_train_acc_nopad_till_current_batch'):
    Dump = []
    for index in range(len(df)):
        Dump.append(float(np.array(df[column])[index][7:-18]))
    return Dump

train=df.query('mode == "train"').query('batch_index == 547')
test=df.query('mode == "val"').query('batch_index == 30')

Model_name = 'Sqe2seq_layers=2_'

utils.plot.result_plot(Model_name+"loss","loss",
                        np.array(train['loss_batch']),
                        np.array(test['loss_batch']),
                        DPI=400)

utils.plot.result_plot(Model_name+"Accuracy","Accuracy",
                        accu_2_int(train,'avg_train_acc_nopad_till_current_batch'),
                        accu_2_int(test,'avg_val_acc_nopad_till_current_batch'),
                        DPI=400)