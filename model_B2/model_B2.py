import numpy as np
import joblib
import googletrans

class model_B2():
    def __init__(self, referenceset_x, referenceset_y,
                 vectorizer_path = "./TFIDF_Vectorizer.joblib",
                 nca_model_path = "./TFIDF_NCA.joblib",
                 knn_model_path = "./TFIDF_Model.joblib"):
        
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
        
        self.tfidf_vectorizer = joblib.load(vectorizer_path)
        
        self.nca_model = joblib.load(nca_model_path)
        self.knn_model = joblib.load(knn_model_path)
        self.knn_model.fit(referenceset_x, referenceset_y)

    def translate(self, message_text):
        #Translate the text to dutch, iif the detected source is not dutch
        translation = self.translator.translate(message_text, dest = "nl")
        
        if translation.src == "nl":
            return message_text
        else:
            return translation.text
    
    def encode(self, message_text):
        tfidf_scores = self.tfidf_vectorizer.transform([message_text]).toarray()
        nca_message = self.nca_model.transform(tfidf_scores)[0]
        
        return nca_message
        
    def predict(self, message_text):
        translated_message = self.translate(message_text)
        
        encoded_message = self.encode(translated_message)
               
        probabilities = self.knn_model.predict_proba([encoded_message])[0]
        
        labeled_predictions = [{"label":lab, "score":prob}
                               for lab, prob in zip(self.id_to_label.values(), probabilities)] 
        sorted_predictions = sorted(labeled_predictions, key = lambda x: x['score'], reverse = True)
        most_likely_label = sorted_predictions[0]["label"]

        return {"most_likely_label":most_likely_label,
                "sorted_predictions": sorted_predictions,
                "message_encoding": encoded_message.tolist()}
    
    
if __name__ == "__main__":
    referenceset_x = np.load("./TFIDF_ReferenceSet_X.npy") # Array with message encodings
    referenceset_y = np.load("./TFIDF_ReferenceSet_Y.npy") # Array with message categories

    model = model_B2(referenceset_x, referenceset_y)
    predictions = model.predict("This is an example message")
    print(predictions)
