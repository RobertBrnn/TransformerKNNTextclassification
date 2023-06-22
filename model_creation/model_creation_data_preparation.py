import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher

tqdm.pandas()
#%%
"""
Omitted: Collection of contactforms from database
"""
#%%
contactforms = contactforms.loc[~contactforms["body"].isna() & contactforms["body"].str.contains("CONTACTFORMULIER", True),:] # Keep only filled CONTACTFORMULIER
contactforms["form_subject"] = contactforms["body"].str.extract("(?<=CONTACTFORMULIER: )(.*?)(?=<\/p>)") # Extract the subject from the body
contactforms["form_parts"] = contactforms["body"].str.findall("(?<=<p>)(.*?)(?=<\/p>)(?s)") # Split the different parts of the message
contactforms["form_parts_len"] = contactforms["form_parts"].apply(lambda x: len(x) if x else 0) # Count the different parts of the message
contactforms["form_body"] = contactforms["form_parts"].apply(lambda x: x[3] if len(x) == 5 else " ".join(x[3:5])) # Extract the text from the message

contactforms = contactforms[~contactforms["form_body"].str.contains(r"[\u4e00-\u9fff]+")] # Remove messages with any CJK text (Chinese, Japanese, Korean)
contactforms["form_body"] = contactforms["form_body"].str.replace("<br />", " ") # Remove break lines
contactforms["form_body"] = contactforms["form_body"].str.replace("Toelichting: ", "") # Remove standard text "Toelichting: "
contactforms["cleaned_body"] = contactforms["form_body"].progress_apply(lambda x: BeautifulSoup(x, "html.parser").get_text()) # Parse message as html to get rid of formatting
contactforms["form_body_length"] = contactforms["cleaned_body"].progress_apply(len)
contactforms.reset_index(drop = True, inplace = True)

#%% Splitting data into batches by number of characters to prevent throttling of google translate api
current_batch_size = 0
batch = {"indices":[], "bodies":[]}
batches = []

for i, row in tqdm(contactforms.iterrows(), total = contactforms.shape[0]):
    body_size = row["form_body_length"]
    
    if len(batch["indices"]) == 0 or (current_batch_size + body_size < 10_000):
        batch["indices"].append(i)
        batch["bodies"].append(row["cleaned_body"])
        current_batch_size += body_size
    else:
        batches.append(batch)
        batch =  {"indices":[i], "bodies":[row["cleaned_body"]]}
        current_batch_size = body_size
batches.append(batch)

#%% Obtain Dutch and English translation of each message in a batch at once (concat using '\n><><\n' -> translate -> split on '\n><><\n')
problem_batches = []
from googletrans import Translator
translator = Translator()

pbar = tqdm(batches)
for batch in pbar:
    try:
        pbar.set_description_str(f"Processing {min(batch['indices']):>6} - {max(batch['indices']):<6}")
        combined_string = "\n><><\n".join(batch["bodies"])
        comb_en = translator.translate(combined_string, src = "nl", dest = "en").text.split("\n><><\n")
        comb_nl = translator.translate(combined_string, src = "en", dest = "nl").text.split("\n><><\n")
    
        contactforms.loc[batch["indices"],"cleaned_body_english"] = comb_en
        contactforms.loc[batch["indices"],"cleaned_body_dutch"] = comb_nl
        
    except Exception as e:
        print(e)
        problem_batches.append(batch)
        

#%% Select relevant categories and obtain category_ids 
label_nums = {"Geldzaken":0,
              "Opmerking nieuwe kamer":1,
              "Afspraak beheerder":2,
              "Sleutels":3,
              "Servicekosten":4,
              "Overlast":5,
              "Acceptatie Nieuwbouw":6,
              "Algemeen Beheer":7,
              "Algemeen Verhuur":8,
}

data = contactforms.loc[contactforms["category"].isin(label_nums.keys()),
                                ["relationId", "original_language", "cleaned_body", "cleaned_body_dutch", "cleaned_body_english", "my_category"]].copy(deep = True)
data["category"] = data["my_category"].progress_apply(lambda x: label_nums[x])

data["combined_dutch_english"] = data.progress_apply(lambda x: f"{x['cleaned_body_dutch']} [SEP] {x['cleaned_body_english']}", axis = 1)

#%%
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, train_size = 0.8, shuffle = True, random_state = 42)
train, val = train_test_split(train, train_size = 0.8, shuffle = True, random_state = 42)

train.to_pickle("data/Train.pkl")
val.to_pickle("data/Val.pkl")
test.to_pickle("data/Test.pkl")
