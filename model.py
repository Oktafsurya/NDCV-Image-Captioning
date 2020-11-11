import torch
import torch.nn as nn
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embedding_layer = nn.Embedding(num_embeddings = vocab_size, 
                                            embedding_dim = embed_size
                                           )
        
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
        
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            batch_first=True
                           )
        
        self.linear = nn.Linear(in_features = hidden_size,
                                out_features = vocab_size
                               )
        
    def forward(self, features, captions):
        
        # we will fed all the word on the caption to embedding layer from <start> until word-1
        cap_embedding = self.embedding_layer(captions[:,:-1])
        
        # concatenate the features of the encoder (CNN) with the caption embedding results
        # We will using teacher forcer method to train the LSTM 
        # where at t=1 we're using features from encoder, and at t=2,3,4 and so on we'are using word from groundtruth caption
        embeddings_data = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        
        
        lstm_outputs, self.hidden = self.lstm(embeddings_data)
        outputs = self.linear(lstm_outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions_result = []
        for i in range(max_len):
            
            # get the predicted caption from LSTM layer
            predicts, states = self.lstm(inputs, states)
            
            # fed the output LSTM to linear layer
            predicts = self.linear(predicts.squeeze(1))
            
            # get the word with highest probablity 
            target_word = predicts.max(1)[1]
            
            # add word to caption
            captions_result.append(target_word.item())
            
            # Output word from previous timestep is fed to embedding layer 
            # and then used as input LSTM for next timestep. 
            # hidden states from previous timestep also used in next timestep
            inputs = self.embedding_layer(target_word).unsqueeze(1)
            
        return captions_result