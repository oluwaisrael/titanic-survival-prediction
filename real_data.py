import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.columns.tolist())   # see all column names
print(df.isnull().sum())     # check for missing values
print(df["Survived"].value_counts())  # how many survived vs died
# Fill missing ages with the median agesp
df["Age"] = df["Age"].fillna(df["Age"].median())
print(df.isnull().sum())
# Drop Cabin (too many missing values - 687 out of 891)
df = df.drop(columns=["Cabin"])

# Fill missing Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert Sex to numbers (ML models need numbers, not text)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Check we're clean
print(df.isnull().sum())
print(df["Sex"].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Select features and target
X = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]]
y = df["Survived"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
# Try Random Forest - usually more powerful than Logistic Regression
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

forest_predictions = forest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, forest_predictions))
import pandas as pd

feature_importance = pd.Series(
    forest.feature_importances_,
    index=["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]
).sort_values(ascending=False)
print(feature_importance)