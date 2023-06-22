from flask import Flask, request
import pandas as pd
import numpy as np
app = Flask(__name__)


def get_model(model):
    if model == "A1":
        from model_A1.model_A1 import model_A1
        return model_A1(model_path = "./model_A1/model_A1")
    elif model == "A2":
        from model_A2.model_A2 import model_A2
        return model_A2(model_path = "./model_A2/model_A2")
    elif model == "A3":
        from model_A3.model_A3 import model_A3
        return model_A3(model_path = "./model_A3/model_A3")
    elif model == "A4":
        from model_A4.model_A4 import model_A4
        return model_A4(model_path = "./model_A4/model_A4")
    elif model == "B1":
        from model_B1.model_B1 import model_B1
        return model_B1(referenceset_x = np.load("./model_B1/Count_ReferenceSet_X.npy"),
                        referenceset_y = np.load("./model_B1/Count_ReferenceSet_Y.npy"),
                        vectorizer_path = "./model_B1/Count_Vectorizer.joblib",
                        nca_model_path = "./model_B1/Count_NCA.joblib",
                        knn_model_path = "./model_B1/Count_Model.joblib")
    elif model == "B2":
        from model_B2.model_B2 import model_B2
        return model_B2(referenceset_x = np.load("./model_B2/TFIDF_ReferenceSet_X.npy"),
                        referenceset_y = np.load("./model_B2/TFIDF_ReferenceSet_Y.npy"),
                        vectorizer_path = "./model_B2/TFIDF_Vectorizer.joblib",
                        nca_model_path = "./model_B2/TFIDF_NCA.joblib",
                        knn_model_path = "./model_B2/TFIDF_Model.joblib")
    elif model == "B3":
        from model_B3.model_B3 import model_B3
        return model_B3(referenceset_x = np.load("./model_B3/BERTje_ReferenceSet_X.npy"),
                        referenceset_y = np.load("./model_B3/BERTje_ReferenceSet_Y.npy"),
                        nca_model_path = "./model_B3/BERTje_NCA.joblib",
                        bertje_model_path="./model_B3/model_A3",
                        knn_model_path = "./model_B3/BERTje_Model.joblib")
    elif model == "B4":
        from model_B4.model_B4 import model_B4
        return model_B4(referenceset_x = np.load("./model_B4/BERTje_ReferenceSet_X.npy"),
                        referenceset_y = np.load("./model_B4/BERTje_ReferenceSet_Y.npy"),
                        nca_model_path = "./model_B4/BERTje_NCA.joblib",
                        bertje_model_path="./model_B4/model_A3",
                        knn_model_path = "./model_B4/BERTje_Model.joblib")
    
    
@app.route('/categorize', methods=['GET'])
def predict():
    model_code = request.args.get('model')
    model = get_model(model_code)
    predictions = model.predict(request.args.get("message_text"))

    return predictions

#%%
app.run(debug = True)