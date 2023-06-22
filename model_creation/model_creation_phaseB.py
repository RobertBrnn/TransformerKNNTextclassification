import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import transformers
import torch
from datasets import Dataset
import sklearn.neighbors
import sklearn.discriminant_analysis
tqdm.pandas()
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

#%% Specifying model parameters
vect = ["Count", "TFIDF", "BERTje"][0]
n_neighbors  = 30
n_components = 90

save_vectorizer    = True
save_transformer   = True
save_referenceset  = True 
save_model         = True
#%%
from joblib import dump
if vect != "BERTje":
    # Transforming text to vector of values
    if vect == "Count":
        vectorizer = CountVectorizer()
    elif vect == "TFIDF":
        vectorizer = TfidfVectorizer()
   
    train_x = vectorizer.fit_transform(train["text"]).toarray()
    val_x = vectorizer.transform(val["text"]).toarray()
    test_x = vectorizer.transform(test["text"]).toarray() 
    
    if save_vectorizer:
        dump(vectorizer, f"{vect}Vectorizer.joblib")
        
elif vect == "BERTje":
    # Loading of models and preparation of data
    model = transformers.AutoModelForSequenceClassification.from_pretrained("Runs/Trainer_NL_FullRun", output_hidden_states = True)
    tokenizer = transformers.AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased", use_fast = True)

    train_ds = Dataset.from_pandas(train, split = "train")
    val_ds = Dataset.from_pandas(val, split = "eval")
    test_ds = Dataset.from_pandas(test, split = "test")

    train["pred"] = -1
    val["pred"] = -1
    test["pred"] = -1
    train["embedding"] = train.apply(lambda x: [], axis=1)
    val["embedding"] = val.apply(lambda x: [], axis=1)
    test["embedding"] = test.apply(lambda x: [], axis=1)

    # Splitting sentences into tensor of tokens
    def tokenize_function(examples):
            return tokenizer(examples["text"], padding = "max_length", truncation = True, return_tensors = "pt")
        
    tokenized_train_data = train_ds.map(tokenize_function, batched = True)
    tokenized_val_data = val_ds.map(tokenize_function, batched = True)
    tokenized_test_data = test_ds.map(tokenize_function, batched = True)

    tokenized_train_data = torch.tensor(tokenized_train_data['input_ids']).clone().detach()
    tokenized_val_data = torch.tensor(tokenized_val_data['input_ids']).clone().detach()
    tokenized_test_data = torch.tensor(tokenized_test_data['input_ids']).clone().detach()
    
    # Batchwise extraction of document embedding from model
    batch_size = 8
    for i in tqdm(range(0, tokenized_train_data.shape[0]+1, batch_size)):
        train_z = tokenized_train_data[i:i+batch_size, :]
        train_q = model(input_ids = train_z)
        _, train_preds = torch.max(train_q.logits, dim = 1)
        train.iloc[i:i+batch_size, -2] = train_preds.tolist()
        train.iloc[i:i+batch_size, -1] = pd.Series(train_q["hidden_states"][-1][:, 0, :].tolist())
        
    for i in tqdm(range(0, tokenized_val_data.shape[0]+1, batch_size)):
        val_z = tokenized_val_data[i:i+batch_size, :]
        val_q = model(input_ids = val_z)
        _, val_preds = torch.max(val_q.logits, dim = 1)
        val.iloc[i:i+batch_size, -2] = val_preds.tolist()
        val.iloc[i:i+batch_size, -1] = pd.Series(val_q["hidden_states"][-1][:, 0, :].tolist())
    
    for i in tqdm(range(0, tokenized_test_data.shape[0]+1, batch_size)):
        test_z = tokenized_test_data[i:i+batch_size, :]
        test_q = model(input_ids = test_z)
        _, test_preds = torch.max(test_q.logits, dim = 1)
        test.iloc[i:i+batch_size, -2] = test_preds.tolist()
        test.iloc[i:i+batch_size, -1] = pd.Series(test_q["hidden_states"][-1][:, 0, :].tolist())
    
    train_x = np.array(train["embedding"].tolist())
    val_x = np.array(val["embedding"].tolist())
    test_x = np.array(test["embedding"].tolist())
    
#%% Application of NCA
transformer = sklearn.neighbors.NeighborhoodComponentsAnalysis(n_components)

train_x = transformer.fit_transform(train_x, train["label"])
val_x   = transformer.transform(val_x)
test_x  = transformer.transform(test_x)

if save_transformer:
    dump(transformer, f"{vect}_NCA.joblib")

if save_referenceset:
    np.save(f"{vect}_ReferenceSet_X.npy", train_x)
    np.save(f"{vect}_ReferenceSet_Y.npy", train["label"])    
    
#%% Application of KNN
model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, n_jobs = -2)
if save_model:
    dump(model, f"{vect}_Model.joblib")
model.fit(train_x, train["label"])

preds = model.predict(val_x)

acc = np.mean([pred == true for pred, true in zip(preds, val["label"])])

print(f"Num neighbors: {n_neighbors}\tAccuracy: {acc:.5f}")

