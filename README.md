# Reimplementation of kNN-MT
Repository for the trial task of NUS WING - Reimplementation of kNN-MT using hugging face and pytorch. 

Link to paper: https://arxiv.org/abs/2010.00710

We use the DE-EN translation using Facebook's WMT19 pretrained model as an example. 

If you want to use a pre-trained index for the datastore and directly run inference, you can directly jump to _[3. Evaluation with datastore](https://github.com/tongyao-zhu/knn-mt-reimplement#3-evaluation-with-datastore)_. 

If you want to start from scratch and experience the every step, you should start from _[1. Extracting raw features](https://github.com/tongyao-zhu/knn-mt-reimplement#1-extract-raw-features)_.

## Prerequisites
Please create a suitable python environment (using venv or conda). 

Inside the environment, run `pip install -r requirements.txt` to install the dependencies.

As we modified the transformers library, please install the library by:

```
cd transformers
pip install -e .
```



## 1. Extract raw features 
Before building the datastore, we need to extract all features in the form of (hidden_state<sub>i</sub>, word<sub>i+1</sub>). 
We do so by the following: 

```
python generate_raw_features.py   \
    --model_name_or_path facebook/wmt19-de-en  \ 
    --source_lang de   \
    --target_lang en    \
    --dataset_name wmt19  \
    --dataset_config_name de-en  \ 
    --save_path saved_gen \
    --percentage=10 
```

The `--save_path` specifies where to save the generated features. Please make sure there is enough disk space.

The `--percentage` specifies how many percent of the training data will be loaded and passed through the model. To save time and space for local testing, it is best to use a very small number (e.g. 1) first. 

## 2. Construct the datastore index
Although in the paper the datastore consists of key-value pairs, in reality it is a trained FAISS index (after clustering).

This is an essential step. Otherwise, the kNN search becomes infeasible. 

We can train an index by: 


`python datastore.py --feature_dir saved_gen --output_dir datastore_1
`

The `--feature_dir` is the stored features generated in Step 1, and the `--output_dir` is the path in which you want your datastore to be saved. 

## 3. Evaluation with datastore

This is the key step of the k-NN MT pipeline. As we are using the huggingface transformers library for the pretrained transformers model, we continue using it. 

We modified some part of the library about beam search. 

To run the inference and evaluation on the validation dataset, do: 

```
export PAIR=de-en
export DATA_DIR=data/$PAIR
export SAVE_DIR=data/$PAIR
export BS=8
export NUM_BEAMS=15

python evaluate.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path $SAVE_DIR/test_bleu.json \
    --bs $BS \
    --task translation \
    --num_beams $NUM_BEAMS \
    --datastore_path datastore_1 \
    --lambda_value 0.8 \
    --k 64
```

The `--datastore_path` parameter should be the datastore path you saved during Step 2, or it can be a pretrained index (in this repo, _datastore_1/_ contains the index.)

The `--lambda_value` parameter determines how much you want to interpolate between the generated score (lambda) and the knn_score (1-lambda). 

The `--k` parameter is for kNN search. 

The final score for each token is:

>final_score = lambda * score<sub>gen</sub> + (1-lambda) * score<sub>kNN</sub>


Please note that you might not see a big difference between the BLEU scores for different lambda values. This is because we only train the datastore on a very small fraction of the training data (1%), due to computation resources constraint. 
It is too small for any improvement to happen. However, if you have enough resources, you should be able to see the improvement. 


For the baseline without kNN search, remove the _datastore_path_, _k_, and _lambda_value_. You will get: 

```
python evaluate.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt \
    --reference_path $DATA_DIR/val.target \
    --score_path $SAVE_DIR/test_bleu.json \
    --bs $BS \
    --task translation \
    --num_beams $NUM_BEAMS
```


The baseline result should be:
{'bleu': 41.3159, 'n_obs': 2000, 'runtime': 143, 'seconds_per_sample': 0.0715}
