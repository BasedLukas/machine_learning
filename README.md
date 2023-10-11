# attention
The purpose of this project is to illustrate how the attention mechanism works in a LLM.  

#### data_prep.py
Run this to see how the tokenizer works. The tokens are all individual words or punctuation.  
The dataset used is the TinyStories dataset from the paper: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).

#### simple_model.py
The first model `SimpleModel` does not use attention and is instead intended to illustrate how a simple next token predictor works. Instead of using the attention mechanism we simply average the embeddings of all the input tokens. This gives the model some contextual information but not much.  
Example output after training for 10 000 iterations:
```
iteration 9800
Using device: cuda
Mean loss: 6.653624312550414
Learning rate: 1.639592523925157e-05
Sampled text:  they swiftly she the washi lumber a sweaty said . things can lily a bub


iteration 9900
Using device: cuda
Mean loss: 6.46035192527023
Learning rate: 1.4756332715326412e-05
Sampled text:  a woosh be had mostly something so always was pasted a feather the saddening .
```