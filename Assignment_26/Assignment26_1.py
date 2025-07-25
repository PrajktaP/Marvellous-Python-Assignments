import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def play_predictor_knn(datafilepath):
    line = "*"*100

    df = pd.read_csv(datafilepath)

    print(f"PlayPredictor Dataset: {df.head()}\n\n{line}\n")
    print(f"Dimensions of Dataset: {df.shape}\n\n{line}\n")

    df.drop(columns = ['Unnamed: 0'], inplace=True) # with inplace it updates the same df and returns

    le_weather = LabelEncoder()
    le_temp = LabelEncoder()
    le_play = LabelEncoder()

    df["Whether"] = le_weather.fit_transform(df["Whether"])
    df["Temperature"] = le_temp.fit_transform(df["Temperature"])
    df["Play"] = le_play.fit_transform(df["Play"])

    x = df[["Whether", "Temperature"]] # independent variable
    y = df["Play"] # dependent variable

    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(line)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(line)

    accuracy_scores = []
    k_range = range(1, 16)

    best_k = 0
    best_accuracy = 0

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors = k)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

        print(f"Accuracy when k = {k} is: {accuracy}")

        accuracy_scores.append(accuracy)

    print(line)
    print(accuracy_scores)
    print(line)

    print(f"Best value of k: {best_k}")
    print(f"Best accuracy score: {best_accuracy}")

def main():
    play_predictor_knn("PlayPredictor.csv")

if __name__ == "__main__":
    main()