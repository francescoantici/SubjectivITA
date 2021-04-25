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
from evaluateSentences import toLabels

def create_articles_model(modelName = 'random-forest'):
  modelSwitcher = {
      "svm" : LinearSVC,
      "logistic" : LogisticRegression,
      "decision-tree" : DecisionTreeClassifier,
      "random-forest": RandomForestClassifier,
      "naive-bayes" : MultinomialNB
  }
  return modelSwitcher[modelName]()

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
  ypred = toLabels(y_pred)
  ytest = toLabels(ytest)
  with open('results/reports_articles.txt', 'a') as f:
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

def main():
    articlesXtrain, articlesytrain = get_articles(split='train')
    articlesXTest, articlesytest = get_articles(split='test')
    for model in ["svm", "logistic", "random-forest", "naive-bayes", "decision-tree"]:
      articlesModel = create_articles_model(model)
      articlesModel = train_articles_model(articlesModel, articlesXtrain, articlesytrain)
      evaluate_articles_model(articlesModel, articlesXTest, articlesytest)

main()