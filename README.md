# IdentifyBlogger
LSTM for multiclass classification on Blog Authorship Corpus

## Usage
### Training
For default training simply execute:
```
python IdentifyBlogger/train.py --data_path path/to/data
``` 
this will train default network from scratch, this means that all preprocessing information such as data split, 
vocabulary, labels encodings will generated anew.

To save preprocessing for futher trainings or for evaluation add `--save_preprocess_info_dir path/to/sve/info/to` to 
training command and for actually using previously generated info use `--preprocess_info_dir` param with your 
preprocessing infodirectory

to save best models (lowest validation loss) pass path to save model to to `--model_path` parmam 

For more training paramaters please see `IdentifyBlogger.train.train`  docstring

### Evaluation
To evaluate model use command:
```
python IdentifyBlogger/evaluate.py --data_path path/to/data --model_path path/to/trained/model --preprocess_info_dir path/to/preprocessing/info 
```

