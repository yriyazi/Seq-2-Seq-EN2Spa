import  dataloaders
import  nets
import  deeplearning
import  utils
import  torch
import  pandas                      as      pd
import  numpy                       as      np
from    torch.utils.data            import  DataLoader
device = 'cpu'
#%%
model       = nets.Seq2Seq(device=device,Num_layer=utils.Hidden_layer)

model.load_state_dict(torch.load('Model/ckpt_seq2seq_L2_081_101.pt'))
model.eval()

#%%
test__dataset=dataloaders.Seq2SeqDataset_stratified(dataloaders.source, dataloaders.target, dataloaders.pairs,Mode='Test')
test__dataloader = DataLoader(test__dataset, batch_size=utils.batch_test_size, shuffle=True, collate_fn=dataloaders.collate_fn)

#%% BLEU

score = 0
count_failed = 0
with torch.no_grad():
    for batch_idx, (En, Spa)in enumerate(test__dataloader, 1):
        En  = deeplearning.reformat_tensor(En).to(device)
        Spa = deeplearning.reformat_tensor(Spa).to(device)
        output, predicted = model(En, Spa,teacher_forcing_ratio=0)
        
        
        for index in range(Spa.shape[0]):
            Ground = Spa[:,index+1][Spa[:,index+1] != 0][1:-1]
            TTarget = predicted[:,index+1][1:len(Ground)+1]
            dump_spa =[]
            dump_spa_pred = []
            for i in range(len(Ground)):
                dump_spa.append(dataloaders.target.index2word[int(Ground[i].to('cpu'))])
                dump_spa_pred.append(dataloaders.target.index2word[int(TTarget[i].to('cpu'))])
                try:
                    score += utils.calculate_bleu_score(dump_spa,dump_spa_pred)
                except:
                    count_failed += 1
            # break
    BLEU = score/len(test__dataloader)/(Spa.shape[0])

print(BLEU)