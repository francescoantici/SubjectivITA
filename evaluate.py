import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def create_sentences_model():
  input = tf.keras.layers.Input(shape=(), dtype=tf.string)          
  tokenizer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
  tokenized = tokenizer(input)
  encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=True)
  bertOut = encoder(tokenized)["pooled_output"]
  out = tf.keras.layers.Dense(1, activation = 'sigmoid') (tf.keras.layers.Dropout(0.2) (bertOut))
  model = tf.keras.Model(input, out)

  model.compile(optimizer = tf.optimizers.Adam(3e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
  
  return model

def create_articles_model(modelName = 'random-forest'):
  modelSwitcher = {
      "svm" : LinearSVC,
      "logistic" : LogisticRegression,
      "tree" : DecisionTreeClassifier,
      "random-forest": RandomForestClassifier,
      "naive-bayes" : MultinomialNB
  }
  return modelSwitcher[modelName]()

def train_sentences_model(model, Xtrain, ytrain, validation_data = None, save_weights = True):
  try:
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 4, restore_best_weights = True)
    model.fit(Xtrain, ytrain, validation_data = validation_data, batch_size = 16, epochs = 4, callbacks = [callback])
    if save_weights:
      model.save_weights('weights/sentencesModelWeights.h5')
    print("Sentences model trained successfully!")
    return model
  except Exception as e:
    print("Error in training sentences model: {}".format(e))
    return False

def evaluate_sentences_model(model, Xtest, ytest):
  y_pred = model.predict(Xtest)
  ypred = list(map(lambda x: toLabels[x], y_pred))
  ytest = list(map(lambda x: toLabels[x], ytest))
  with open('results/reports_sentences.txt', 'a') as f:
    f.write(classification_report(ytest, y_pred = ypred)+ "\n")

def train_articles_model(model, Xtrain, ytrain):
  try:
    model.fit(X = Xtrain, y = ytrain)
    print("Articles model trained successfully!")
    return model
  except Exception as e:
    print("Error in training articles model: {}".format(e))
    return False

def evaluate_articles_model(model, Xtest, ytest):
  y_pred = model.predict(Xtest)
  ypred = list(map(lambda x: toLabels[x], y_pred))
  ytest = list(map(lambda x: toLabels[x], ytest))
  with open('results/reports_articles.txt', 'a') as f:
    f.write(model+":\n")
    f.write(classification_report(ytest, y_pred = ypred)+ "\n")

def get_articles(split):
  df = pd.read_csv("datasets/articles/articles{}.csv".format(split.capitalize()))
  fonti = list(df['FONTE'].unique())
  tags = {"OGG" : 0, "SOG" : 1}
  X = df.drop(['ID_ARTICOLO', 'TAG_ARTICOLO'], axis = 1)
  X['FONTE'] = X['FONTE'].map(lambda x: fonti.index(x))
  scaler = MinMaxScaler()
  scaler = scaler.fit(X)
  X = scaler.transform(X)
  y = np.array(list(map(lambda x: tags[x], df['TAG_ARTICOLO'].values)))
  return X, y

def get_sentences(split):
  df = pd.read_csv("datasets/sentences/sentences{}.csv".format(split.capitalize()))
  tags = {"OGG" : 0, "SOG" : 1}
  return df['FRASE'].values, np.array(list(map(lambda x: tags[x], df['TAG_FRASE'].values)))
  

def toLabels(data, subT = 0.5):
    ypred = []
    for pred in data:
        if pred > subT:
            ypred.append('SOG')
        else:
            ypred.append('OGG')
    return ypred

def main():
  sentencesModel = create_sentences_model()
  articlesModel = create_articles_model()
  sentencesXtrain, sentencesytrain = get_sentences(split = 'train')
  sentencesXval, sentencesyval = get_sentences(split = 'val')
  sentencesXtest, sentencesytest = get_sentences(split = 'test')
  articlesXtrain, articlesytrain = get_articles(split='train')
  articlesXTest, articlesytest = get_articles(split='test')
  try:
    sentencesModel.load_weights('weights/sentencesModelWeights.h5')
  except:
    sentencesModel = train_sentences_model(sentencesModel, sentencesXtrain, sentencesytrain, 
                                           validation_data = (sentencesXval, sentencesyval))
  articlesModel = train_articles_model(articlesModel, articlesXtrain, articlesytrain)
  evaluate_sentences_model(sentencesModel, sentencesXtest, sentencesytest)
  evaluate_articles_model(articlesModel, articlesXTest, articlesytest)

main()
  
  
  

