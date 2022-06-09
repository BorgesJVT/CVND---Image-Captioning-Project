import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size=32, num_layers=1, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        #self.dropout = nn.Dropout(dropout)
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers= self.num_layers,
                            batch_first = True, dropout=dropout)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
    
    def forward(self, features, captions):
        # features shape: (batch_size, embed)
        # captions shape: (batch_size, seq)
        captions = captions[:,:-1]
        embed = self.embedding(captions)
        features = features.view(self.batch_size, 1, -1)
        # combining features and embed (batch_size, seq, embed)
        input_tensor = torch.cat((features, embed), dim=1)
        lstm_outputs, self.hidden = self.lstm(input_tensor, self.hidden)
        
        lstm_outputs_shape = list(lstm_outputs.shape)
        lstm_outputs = lstm_outputs.reshape(lstm_outputs.size()[0]*lstm_outputs.size()[1], -1)
        #get the probability for the next word
        #vocab outputs shape ; (batch_size*seq, vocab_size)
        vocab_outputs = self.linear(lstm_outputs)
        # new vocab outputs shape :(batch_size, seq, vocab_size)
        vocab_outputs = vocab_outputs.reshape(lstm_outputs_shape[0], lstm_outputs_shape[1], -1)
        
        return vocab_outputs
        
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ''' Given a character, predict the next character.
            Used for inference.
        
            Returns the predicted character and the hidden state.
        '''
        predicted_caption = []
        batch_size = inputs.shape[0]
        hidden = (torch.randn(1, 1, 512).to(inputs.device),
              torch.randn(1, 1, 512).to(inputs.device))
        while True:
            lstm_output, hidden = self.lstm(inputs, hidden)
            predicted_caption = self.linear(lstm_output)
            predicted_caption = predicted_caption.squeeze(1)
            _, max_pred_index = torch.max(predicted_caption, dim = 1)
            predicted_caption.append(max_pred_index.cpu().numpy()[0].item())
            if (max_pred_index == 1):
                break
            inputs = self.embedding(max_pred_index)
            inputs = inputs.unsqueeze(1)
        return predicted_caption