# SubjectivITA
Collection of Italian newspaper's articles labeled as Objective or Subjective.
It has been collected 103 articles, divided in sentences. 
Each sentence is labelled as either Subjective or Objective, an amount of 1841 sentences are collected.
https://doi.org/10.1007/978-3-030-85251-1_4
## Run evaluations 
The evaluations on the dataset have been performed through a AlBERTo model, provided by huggingface, equipped with a fully connected layer on top, the weights of the global model are not loaded for storage reasons.
To run experiments and show reports, install all the requirements with pip install -r requirements.txt and then run the evaluate.py file.

If you use this dataset please cite it as:
```
@InProceedings{10.1007/978-3-030-85251-1_4,
author="Antici, Francesco
and Bolognini, Luca
and Inajetovic, Matteo Antonio
and Ivasiuk, Bogdan
and Galassi, Andrea
and Ruggeri, Federico",
title="SubjectivITA: An Italian Corpus for Subjectivity Detection in Newspapers",
booktitle="Experimental IR Meets Multilinguality, Multimodality, and Interaction",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="40--52",
}
```




