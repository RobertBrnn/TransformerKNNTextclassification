from transformers import BertTokenizer, BertForSequenceClassification, \
    TextClassificationPipeline, PretrainedConfig
import numpy as np
import joblib
import googletrans

class model_B4():
    
    def __init__(self, referenceset_x, referenceset_y,
                 p_knn = 0.195,
                 nca_model_path = "./BERTje_NCA.joblib",
                 bertje_model_path = "./model_A3",
                 knn_model_path = "./BERTje_Model.joblib"):
        
        self.p_knn = p_knn
        self.translator = googletrans.Translator()
        self.id_to_label = {0: "Geldzaken",
                  1: "Algemeen Beheer",
                  2: "Opmerking nieuwe kamer",
                  3: "Afspraak beheerder",
                  4: "Algemeen Verhuur",
                  5: "Sleutels",
                  6: "Servicekosten",
                  7: "Overlast",
                  8: "Acceptatie Nieuwbouw",
              }        

        self.nca_model = joblib.load(nca_model_path)
        self.knn_model = joblib.load(knn_model_path)
        self.knn_model.fit(referenceset_x, referenceset_y)
        
        self.bertje_model_config = PretrainedConfig().from_json_file(f"{bertje_model_path}/config.json")
        self.bertje_model = BertForSequenceClassification.from_pretrained(bertje_model_path, id2label = self.id_to_label)

        self.tokenizer = BertTokenizer.from_pretrained(self.bertje_model_config._name_or_path, use_fast = True)

        self.pipeline = TextClassificationPipeline(model = self.bertje_model, tokenizer = self.tokenizer, top_k = None)

    def translate(self, message_text):
        #Translate the text to dutch, iif the detected source is not dutch
        translation = self.translator.translate(message_text, dest = "nl")
        
        if translation.src == "nl":
            return message_text
        else:
            return translation.text
    
    def encode(self, message_text):
        input_tokens  = self.tokenizer(message_text, return_tensors="pt")
        outputs = self.bertje_model(**input_tokens, output_hidden_states=True)
        
        embeddings = outputs["hidden_states"][-1][:, 0, :].detach().numpy()

        nca_embedding = self.nca_model.transform(embeddings)[0]
        return nca_embedding
    
    def predict(self, message_text):
        translated_message = self.translate(message_text)
        
        encoded_message = self.encode(translated_message)
        
        knn_probabilities = self.knn_model.predict_proba([encoded_message])[0]
        bertje_probabilities = self.pipeline(translated_message)[0]
        
        combined_probabilities = [{"label":bertje_pred['label'],
                                   "score": self.p_knn * knn_prob + (1 - self.p_knn) * bertje_pred['score']} 
                                  for knn_prob, bertje_pred in zip(knn_probabilities, bertje_probabilities) ]

        sorted_predictions = sorted(combined_probabilities, key = lambda x: x['score'], reverse = True)
        most_likely_label = sorted_predictions[0]["label"]      
            
        return {"most_likely_label":  most_likely_label,
                "sorted_predictions": sorted_predictions,
                "message_encoding":   encoded_message.tolist()}

if __name__ == "__main__":
    referenceset_x = np.load("./BERTje_ReferenceSet_X.npy") # Array with message encodings
    referenceset_y = np.load("./BERTje_ReferenceSet_Y.npy") # Array with message categories

    model = model_B4(referenceset_x, referenceset_y)
    predictions = model.predict("This is an example message")
    print(predictions)
