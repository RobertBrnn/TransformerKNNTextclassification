from transformers import BertTokenizer, BertForSequenceClassification, \
    TextClassificationPipeline, PretrainedConfig
import googletrans

class model_A1:
    
    def __init__(self, model_path):
        
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
        
        self.model_config = PretrainedConfig().from_json_file(f"{model_path}/config.json")
        self.base_model_name = self.model_config._name_or_path
        
        self.tokenizer = BertTokenizer.from_pretrained(self.base_model_name, use_fast = True)
        self.model = BertForSequenceClassification.from_pretrained(model_path, id2label = self.id_to_label)
        
        self.pipeline = TextClassificationPipeline(model = self.model, tokenizer = self.tokenizer, top_k = None)
        
    def translate(self, message_text):
        #Translate the text to dutch, iif the detected source is not dutch
        translation = self.translator.translate(message_text, dest = "nl")
        
        if translation.src == "nl":
            return message_text
        else:
            return translation.text 
        
    def predict(self, message_text):
        translated_message = self.translate(message_text)
        
        predictions = self.pipeline(translated_message)[0]
        
        sorted_predictions = sorted(predictions, key = lambda x: x['score'], reverse = True)
        most_likely_label = sorted_predictions[0]["label"]
        
        return {"most_likely_label":most_likely_label,
                "sorted_predictions": sorted_predictions}
    
if __name__ == "__main__":
    model = model_A1(model_path = "./model_A1")
    predictions = model.predict("This is a example message")
    print(predictions)