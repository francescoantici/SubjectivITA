# SubjectivITA
Collection of Italian newspaper's articles labeled as Objective or Subjective.
It has been collected 103 articles, divided in sentences. 
Each sentence is labelled as either Subjective or Objective, an amount of 1841 sentences are collected.
## Run evaluations 
The evaluations on the dataset have been performed through a AlBERTo model, provided by huggingface, equipped with a fully connected layer on top, the weights of the global model are not loaded for storage reasons.
To run experiments and show reports, install all the requirements with pip install -r requirements.txt and then run the evaluate.py file.


