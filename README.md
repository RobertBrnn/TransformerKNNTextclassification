# Automatic message classification

This repository contains files which relate to the master thesis of Robert Brandemann for the master Business Analytics (2023).
It contains files for the creation and usage of the models, as created during this study.
 
## Usage

All python model files contain a class for the relevant model, which needs to be instantiated with a certain number of paths to models/model parts.
These all include a method for the categorization of a message:

```python
model = model_A1(model_path = "./model_A1")
predictions = model.predict("This is a example message")

# Example:
predictions = {"most_likely_label": "category_1",
	"sorted_predictions": [{"label": "category_1", "score": 0.9},
						   {"label": "category_4", "score": 0.1}<
						   ...]}
```
All models return the most_likely_label and the sorted_predictions. Models from the second phase require a referenceset of data to compare against.
These consist of encoded messages and their labels. The models that require these, also return the encoding for the input message.
These should be accumulated and added to the existing referenceset.

The folder `model_creation` contains code which can be used to create the relevant models.
`rest_api_interface.py` contains code to compile a REST API for easy access to the models.
