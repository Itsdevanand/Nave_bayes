import pandas as pd

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\9.Naive Bayes\Disaster_tweets_NB.csv')

X = df['text']

Y = df.target


import re
from nltk.corpus import stopwords

stop_words = stopwords.words('English')
X1 = []



# Cleaning Text data
for i in X:
    text = re.sub("[^A-Za-z" "]+"," ", i).lower()
    text = re.sub("[0-9" "]+"," ", text)
    words = text.split(' ')
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    X1.append(text)
  
from sklearn.model_selection import train_test_split

#splitting the data

x, x_test, y, y_test = train_test_split(X1, Y, test_size=0.2 )


    
from sklearn.feature_extraction.text import CountVectorizer

vect  = CountVectorizer()    

x = vect.fit_transform(x).toarray()

x_test = vect.transform(x_test).toarray()


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(x, y)

model.score(x, y) 

pred = model.predict(x_test) #predicted values

train_pred = model.predict(x) 

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)   #Test accuracy
accuracy_score(y, train_pred)  #Train accuracy

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred)

