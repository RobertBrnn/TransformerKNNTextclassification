import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                        TrainingArguments, Trainer
from datasets import Dataset
import transformers
import evaluate
#%% Setting relevant seeds
np.random.seed(42)
transformers.set_seed(42)
#%%
text_col = "cleaned_body_dutch"
train = pd.read_pickle("Train.pkl").loc[:, [text_col, "category"]]
val = pd.read_pickle("Val.pkl").loc[:, [text_col, "category"]]
test = pd.read_pickle("Test.pkl").loc[:, [text_col, "category"]]

train = train.loc[:, [text_col, "category"]].rename(columns = {text_col:"text", "category":"label"})
val = val.loc[:, [text_col, "category"]].rename(columns = {text_col:"text", "category":"label"})
test = test.loc[:, [text_col, "category"]].rename(columns = {text_col:"text", "category":"label"})
#%% Config setup
model_type = "EN"

config = {"NL": {"model_name": "GroNLP/bert-base-dutch-cased",
                 "text_col": "cleaned_body_dutch",
                 "result_folder": "Trainer_NL_BERTje"},
          "NL_EN": {"model_name": "bert-base-multilingual-cased",
                    "text_col": "cleaned_body",
                    "result_folder": "Trainer_NL_EN_mBERT"},
          "NL+EN": {"model_name": "bert-base-multilingual-cased",
                    "text_col": "combined_dutch_english",
                    "result_folder": "Trainer_NL+EN_mBERT"},
          "EN": {"model_name": "bert-base-cased",
                 "text_col":"cleaned_body_english",
                 "result_folder": "Trainer_EN_BERT"}
         }[model_type]
#%% Create Dataset objects
train_ds = Dataset.from_pandas(train, split = "train")
val_ds = Dataset.from_pandas(val, split = "train")
test_ds = Dataset.from_pandas(test, split = "test")

#%% Tokenize data
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast = False)

def tokenize_function(examples):
        return tokenizer(examples["text"], padding = "max_length", truncation = True)
    
tokenized_train_data = train_ds.map(tokenize_function, batched = True)
tokenized_val_data   = val_ds.map(tokenize_function, batched = True)
tokenized_test_data  = test_ds.map(tokenize_function, batched = True)

#%% Evaluation metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = 1)
    return metric.compute(predictions=predictions, references = labels)
#%% Model definition
model = AutoModelForSequenceClassification.from_pretrained(
    config["model_name"], num_labels = 9)

#%% Training setup
training_args = TrainingArguments(output_dir = config["result_folder"],
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  save_total_limit = 2,
                                  load_best_model_at_end = True)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train_data,
    eval_dataset = tokenized_test_data,
    compute_metrics = compute_metrics,
    )
#%% Training
trainer.train()