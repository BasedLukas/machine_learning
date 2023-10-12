# attention
The purpose of this project is to illustrate how various components of deep learning models work for NLP tasks. I try to write code in the simplest possible manner, often at the expense of performance. This repo is designed to be followed step by step in the following order.  

#### 1. Input data prep
In order to model language, we need to convert it to a numerical representation. This is done by the tokenizer.
Run `data_prep.py` this to see how my basic tokenizer works. The tokens are all individual words or punctuation.  
The dataset used is the TinyStories dataset from the paper: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).


#### 2. simple model
The first model `simple_model.py` does not use attention, and is instead intended to illustrate how a simple next token predictor works. Instead of using the attention mechanism we simply average the embeddings of all the input tokens. This gives the model some contextual information but not much.  
Example output before and after training for 10 000 iterations:
```
iteration 100
Using device: cuda
Mean loss: 9.505645976347083
Learning rate: 0.45
Sampled text:  . 54 . whiskles . leonora . daywas . meshing . excitied diagnosis . chuckie

iteration 9900
Using device: cuda
Mean loss: 6.46035192527023
Learning rate: 1.4756332715326412e-05
Sampled text:  a woosh be had mostly something so always was pasted a feather the saddening .
```
The model works as follows:
* A string of text is tokenized into alist of numbers. 
* If the length of the input vector (tokenized text) is longer than the context window it is truncated.
* If the length is less, it is padded. Thus all input sequences end up with the same length (context window)
* Each token is converted into an embedding vector. The embedding layer is a lookup table of dim (vocab_size, embedding_dim). Thus every word or symbol in our vocab is mapped to a vector (which is initialized randomly). Thus our data now has dim (context_window, embedding_dim).
* In order to combine all the embedding vectors into one, we average them. This is obviously very crude. We combine the information contained in the input text into a single vector, but lose any positional or contextual information. (Attention and positional encodings will fix this). Our data is now of shape (embedding_dim).
* Finally we pass the data through a NN. In this example we use LeakyRelu as the activation fuunction and also add dropout. The output from the model is a vector of dim vocab_size where each index corresponds to a word in our vocab.
* Training follows the standard procedure; using an optimizer (SGD) and a loss function (cross entropy)