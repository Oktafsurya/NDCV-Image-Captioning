# NDCV-Image-Captioning

In this project, we will create a neural network architecture to automatically generate captions from images. 
We used [MS COCO Datasets](https://cocodataset.org/#captions-2015) for captioning task.

## Model Architecture
The model consists of 2 parts: the encoder and decoder.
### 1. Encoder
The encoder that we provide to you uses the pre-trained **ResNet-50** architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. 
The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.

### 2. Decoder
There are 3 layers in our decoder: embedding layer, LSTM layer and linear layer. 
Initially the encoder output features will be fed to the decoder embedding layer then the results from the embedding layer will be fed to the LSTM.
We will use the **teacher forcer** method to train LSTM where at t = 1 we use the features from the encoder, and at t = 2,3,4 
and so on we use the word from the groundtruth caption as input to the LSTM.

<p align="center"> 
<img src=https://github.com/Oktafsurya/NDCV-Image-Captioning/blob/master/images/encoder-decoder.png>
</p>

## Inference Result
<p align="center"> 
<img src=https://github.com/Oktafsurya/NDCV-Image-Captioning/blob/master/images/img_caption_result1.png> <img src=https://github.com/Oktafsurya/NDCV-Image-Captioning/blob/master/images/img_caption_result2.png>
</p>
