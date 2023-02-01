# Reimplementation of kNN-MT
Repository for the trial task of NUS WING - Reimplementation of kNN-MT using hugging face and pytorch. 

Link to paper: https://arxiv.org/abs/2010.00710

If you want to use a pre-trained index for the datastore and directly run inference, you can directly jump to _3. Evaluation with datastore_. 

If you want to start from scratch and experience the every step, you should start from _1. Extracting raw features_.

## 1. Extract raw features 
Before building the datastore, we need to extract all features in the form of (hidden_state<sub>i</sub>, word<sub>i+1</sub>) 
We do so by the following: 

## 2. Construct the datastore index
Although in the paper the datastore consists of key-value pairs, in reality it is a trained FAISS index (after clustering).
This is an essential step. Otherwise, the kNN search becomes infeasible. 
We can train an index by: 

`python datastore.py --feature_dir saved_gen --output_dir datastore_1
`
The --feature_dir is the stored features generated in Step 1, and the output_dir is the path in which you want your datastore to be saved. 

## 3. Evaluation with datastore
This is the key step of the k-NN MT pipeline. As we are using the huggingface transformers library for the pretrained transformers model, we continue using it. 
We modified some part of the library about beam search. 

To run the inference and evaluation on the validation dataset, do: 


The datastore_path parameter should be the datastore path you saved during Step 2. 
The lambda_value parameter determines how much you want to interpolate between the generated score (lambda) and the knn_score (1-lambda). 
The final score for each token is:

final_score = lambda * score<sub>gen</sub> + (1-lambda) * score<sub>kNN</sub>


Please note that you might not see a big difference between BLEU score for the lambda values. This is because we only train the datastore on a very small fraction of the training data (1%), due to computation resources constraint. 
It is too small for any improvement to happen. However, if you have enough resources, you should be able to see the improvement. 