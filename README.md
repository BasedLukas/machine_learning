# Language models
The purpose of this project is to illustrate how various components of deep learning models work for NLP tasks. I try to write code in the simplest possible manner, often at the expense of performance. This repo is designed to be followed step by step in the following order.  

## 1. basic_model
#### Tokenizer and dataset
In order to model language, we need to convert it to a numerical representation. This is done by the tokenizer. The tokens are all individual words or punctuation.  
The dataset used is the TinyStories dataset from the paper: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).
This is how the data is fed into the model:
```
x is [185]
y is [19559]
decoded:
x is:  <sos>
y is:  once

x is [185, 19559]
y is [31619]
decoded:
x is:  <sos> once
y is:  upon

x is [185, 19559, 31619]
y is [188]
decoded:
x is:  <sos> once upon
y is:  a

x is [185, 19559, 31619, 188]
y is [29862]
decoded:
x is:  <sos> once upon a
y is:  time
```

The input text is also truncated or padded to be of the correct length:

```
text: hello this is a test of more than 8 words in a sentence
encoded: [13033, 29600, 14535, 188, 29310, 19446, 18327, 29339, 160, 33166, 14062, 188, 25185]
decoded: hello this is a test of more than 8 words in a sentence
Input x is [13033, 29600, 14535, 188, 29310, 19446, 18327, 29339, 160, 33166, 14062, 188, 25185]
After context window, x is [19446, 18327, 29339, 160, 33166, 14062, 188, 25185]
Padding is []
Length of the padding is 0
final x is tensor([19446, 18327, 29339,   160, 33166, 14062,   188, 25185])
embedded_input shape is torch.Size([8, 126])

text: hello this is a test
encoded: [13033, 29600, 14535, 188, 29310]
decoded: hello this is a test
Input x is [13033, 29600, 14535, 188, 29310]
After context window, x is [13033, 29600, 14535, 188, 29310]
Padding is [0, 0, 0]
Length of the padding is 3
final x is tensor([    0,     0,     0, 13033, 29600, 14535,   188, 29310])
embedded_input shape is torch.Size([8, 126])
```
#### Model
The model does not use self attention, and is instead intended to illustrate how a simple next token predictor works. 
The model works as follows:
* A string of text is tokenized into a list of numbers. 
* If the length of the input vector (tokenized text) is longer than the context window it is truncated.
* If the length is less, it is padded. Thus all input sequences end up with the same length (context window)
* Each token is converted into an embedding vector. The embedding layer is a lookup table of dim (vocab_size, embedding_dim). Thus every word or symbol in our vocab is mapped to a vector (which is initialized randomly). Thus our data now has dim (context_window, embedding_dim).
* In order to combine all the embedding vectors into one, we average them. This is obviously very crude. We combine the information contained in the input text into a single vector, but lose any positional or contextual information. (Attention and positional encodings will fix this). Our data is now of shape (embedding_dim).
* Finally we pass the data through a NN. In this example we use LeakyRelu as the activation fuunction and also add dropout. The output from the model is a vector of dim vocab_size where each index corresponds to a word in our vocab.
* Training follows the standard procedure; using an optimizer (SGD) and a loss function (cross entropy)

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


## 2. model_v1 (multiheaded attention model)
This model is more advanced, we use multiheaded attention and the BERT tokenizer. We also use positional encodings. The model is composed of two key parts, the transformer blocks and the wrapper. The wrapper embeds the input passes it through multiple transformer instances and then projects the output to the vocab size dimension.


```
params = {
    'epochs': 7,
    'batch_size': 200,
    'num_transformers': 4, # number of transformer layers
    'seq_len': 45,
    'vocab_size': 30522,
    'embed_size': 256 ,
    'n_heads': 8,
    'output_dim': 30522,
    'hidden_size': 256, # feedforward network hidden size
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'patience': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```
Sample generations:
```
Prompt:
had a sister called lilly. they liked to play with their toys. one day they decided to go to the park. they played on the swings and the slide. then they went home and had dinner. they ate meat and

Completions:
grandpa and said, " yes if we can do it again today! " he felt good. " they grabbed the train and got fun. they did not sorry about it and going to leave the hospital. it's important to be fierce! i

been after the sun, their toys was not tired to play with anymore. he emir as they looked through the gilded at the end. each other and as they passed playing with her microphone and made stories tos their grandma. the sunglasses was stuck to

the rest. the cub praised them and went and put the ball with before all tho. they finally came and laughed until they was found the big children sound. the boy were playing with the loberries against the animals in the sky. he asks timmy
```

## 3. model_v2
This model is similar to v1 and my best model to date.