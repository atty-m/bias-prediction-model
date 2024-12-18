import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def data_loader(directory:str="no directory"):
    json_list = pd.read_csv(directory,sep="\t")
    data = []
    for ID in json_list["ID"]:
        with open(f"data/jsons/{ID}.json","r") as json_file:
            data.append(json.load(json_file))
    return pd.DataFrame.from_dict(data)


def predict_bias(text:str):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction

model = SVC(kernel="linear")
vectorizer = TfidfVectorizer(stop_words="english",max_df=0.7)

df = data_loader("data/splits/random/test.tsv")

df.to_csv("data.csv")
X = vectorizer.fit_transform(df["content"])
y = df["bias"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)

print(accuracy)
print(report)

print(predict_bias(input("Text to predict")))