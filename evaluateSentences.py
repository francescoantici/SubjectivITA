import tensorflow_text as text
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

maxSentenceLen = 20

def prepare_data(X, y):
    pad = tf.keras.preprocessing.sequence.pad_sequences#(seq, padding = 'post', maxlen = maxlen)
    tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
    dataFields = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "emotions": []
        }
    lbls = {
        'SOG' : [1.0],
        'OGG' : [0.0]
    }
    for i in range(len(X)):
        data = tokenizer(X[i])
        padded = pad([data['input_ids'], data['attention_mask'], data['token_type_ids']], padding = 'post', maxlen = maxSentenceLen)
        dataFields['input_ids'].append(padded[0])
        dataFields['attention_mask'].append(padded[1])
        dataFields['token_type_ids'].append(padded[-1])
        dataFields['emotions'].append(lbls[y[i]])
    
    for key in dataFields:
        dataFields[key] = np.array(dataFields[key])
    
    return [dataFields["input_ids"], dataFields["token_type_ids"], dataFields["attention_mask"]], dataFields["emotions"]

def create_sentences_model():
    input_ids = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(maxSentenceLen,), dtype=tf.int32)
    bertModel = TFBertModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[-1]
    out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(tf.keras.layers.Dropout(0.1)(bertModel))
    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
    model.compile(optimizer = tf.optimizers.Adam(2e-5), loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    return model

def train_sentences_model(model, Xtrain, ytrain, validation_data, save_weights = True):
  try:
    Xtrain, ytrain = prepare_data(Xtrain, ytrain)
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 2, restore_best_weights = True)
    model.fit(Xtrain, ytrain, validation_data = prepare_data(validation_data[0], validation_data[1]), batch_size = 16, epochs = 4, callbacks = [callback])
    if save_weights:
      model.save_weights('weights/sentencesModelWeights.h5')
    print("Sentences model trained successfully!")
    return model
  except Exception as e:
    print("Error in training sentences model: {}".format(e))
    return False

def evaluate_sentences_model(model, Xtest, ytest):
  Xtest, ytest = prepare_data(Xtest, ytest)
  y_pred = model.predict(Xtest)
  ypred = toLabels(y_pred)
  ytest = toLabels(ytest)
  with open('results/reports_sentences.txt', 'w') as f:
    f.write(classification_report(ytest, y_pred = ypred)+ "\n")

def get_sentences(split):
  df = pd.read_csv("datasets/sentences/sentences{}.csv".format(split.capitalize()))
  return df['FRASE'].values, df['TAG_FRASE'].values
  
def toLabels(data, subT = 0.5):
    ypred = []
    for pred in data:
        if pred > subT:
            ypred.append('SOG')
        else:
            ypred.append('OGG')
    return ypred

def main(train = False):
  sentencesModel = create_sentences_model()
  sentencesXtrain, sentencesytrain = get_sentences(split = 'train')
  sentencesXval, sentencesyval = get_sentences(split = 'val')
  sentencesXtest, sentencesytest = get_sentences(split = 'test')
  if train:
    sentencesModel = train_sentences_model(sentencesModel, sentencesXtrain, sentencesytrain, validation_data = (sentencesXval, sentencesyval))
  else:
    try:
      sentencesModel.load_weights('weights/sentencesModelWeights.h5')
    except:
      print("No weights found!")
  evaluate_sentences_model(sentencesModel, sentencesXtest, sentencesytest)

main()
  
  
  

