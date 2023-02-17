### Testing the code correctness

This folder contains the test cases. Currently, we only formally test `datastore.py`, as this is our self-defined class for index search. 
The test cases are created using the [unittest](https://docs.python.org/3/library/unittest.html) framework. 

For the other two files `generate_raw_features.py` and `evaluate.py`, they are adapted based on the [transformers](https://github.com/huggingface/transformers/tree/main/examples/legacy/seq2seq) library. We assume that
there is no issue in their implementation. For our modified/added lines, we inserted assertions in the code to ensure correctness.

To run `test_datastore.py`, go back to the root directory of this project, and do: 

`python -m unittest`

It will automatically run the tests. 