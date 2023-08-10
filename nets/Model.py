import torch
import random
import dataloaders
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self,
                 embeddings_dim:int,
                 hidden_Dimention:int,
                 Num_layer:int = 1):
        super().__init__()
        
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(dataloaders.source.embeddings),freeze=True)
        self.LSTM         = nn.LSTM     (input_size = embeddings_dim,
                                         hidden_size = hidden_Dimention,
                                         num_layers =Num_layer,
                                         dropout=0.20,)

    def forward(self, input):
        """
        output embedded is shape of [len(input),Batch_size,embedding_lenght]
        
        output _            is shape of [len(input),Batch_size,hidden_Dimention      ]
        output hidden_state is shape of [1         ,Batch_size,hidden_Dimention      ]
        output cell_state   is shape of [1         ,Batch_size,hidden_Dimention      ]
        """
        embedded = self.embeddings(input.to(torch.int))
        _ , (hidden_state, cell_state) = self.LSTM(embedded)

        return hidden_state, cell_state


class Decoder(nn.Module):
    def __init__(self, 
                 word_in_target:int,
                 embeddings_dim:int,
                 hidden_Dimention:int,
                 device:str,
                 Num_layer:int = 1):
        
        super().__init__()
        self.device     = device
        self.word_in_target = word_in_target
        self.embeddings   = nn.Embedding(self.word_in_target   , embeddings_dim)
        self.LSTM         = nn.LSTM     (input_size = embeddings_dim,
                                         hidden_size = hidden_Dimention,
                                         num_layers =Num_layer,
                                         dropout=0.20,)
        
        self.Fc_LinearMap = nn.Linear   (hidden_Dimention , self.word_in_target)
        # self.embeddings.weight[0].data = torch.zeros(size=[1,100])#.to(device=self.device)

    def forward(self, input, hidden_state, cell_state):
        """
        output embedded is shape of     [1,Batch_size,embedding_lenght]
        output hidden_state is shape of [1,Batch_size,hidden_Dimention]
        output cell_state   is shape of [1,Batch_size,hidden_Dimention]
        
        output Fc_LinearMap is shape of [NONE,batch size,word_in_target]

        """
        
        embedded = self.embeddings(input.unsqueeze(0).to(torch.int))
        embedded = nn.functional.relu(embedded)
        
        output, (hidden_state, cell_state) = self.LSTM(embedded, (hidden_state, cell_state))
        prediction = self.Fc_LinearMap(output.squeeze(0))
        
        return prediction, hidden_state, cell_state


class Seq2Seq(nn.Module):
    def __init__(self,
                 embeddings_dim_encoder     :int = 100,
                 hidden_Dimention_encoder   :int = 100,
                 embeddings_dim_decoder     :int = 100,
                 hidden_Dimention_decoder   :int = 100,
                 Num_layer = 1,
                 device:str='cuda'):
        
        super().__init__()

        self.encoder = Encoder(embeddings_dim=embeddings_dim_encoder,
                               hidden_Dimention=hidden_Dimention_encoder,
                               Num_layer = Num_layer).to(device=device)
        
        self.decoder = Decoder(word_in_target=dataloaders.target.n_words,
                               embeddings_dim=embeddings_dim_decoder,
                               hidden_Dimention=hidden_Dimention_decoder,
                               device=device,
                               Num_layer = Num_layer).to(device=device)
        self.device = device

    def forward(self, input, ground_truth, teacher_forcing_ratio=0.25):
        """
        output embedded is shape of     [1,Batch_size,embedding_lenght]
        output hidden_state is shape of [1,Batch_size,hidden_Dimention]
        output cell_state   is shape of [1,Batch_size,hidden_Dimention]
        
        output Fc_LinearMap is shape of [NONE,batch size,word_in_target]

        """
        
        # src = [src len, batch size]
        # trg = [trg len, batch size]


        trg_len , batch_size =ground_truth.shape[0] ,  ground_truth.shape[1]
        trg_vocab_size = self.decoder.word_in_target

        hidden_state, cell_state = self.encoder(input)


        # first input to the decoder is the <sos> tokens
        input = ground_truth[0, :]
        predicted = torch.zeros(trg_len, batch_size, device=self.device)
        decoder_outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=self.device)
        
        for t in range(1, trg_len):

            output_decoder, hidden_state, cell_state = self.decoder(input, hidden_state, cell_state)
            decoder_outputs[t] = output_decoder

            # get the tpo1 predicted token from our predictions
            predicted[t, :] = output_decoder.argmax(1)

            # teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            input = ground_truth[t] if teacher_force else predicted[t, :]


        return decoder_outputs, predicted