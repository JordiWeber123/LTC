import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the Excel file with titles and categories
input_file = "e_commerce_input.xlsx"

# Load the dataset from the Excel file
df = pd.read_excel(input_file)

# Split the original data into only the first category in "Category"
for i, row in enumerate(df["Category"]):
     df["Category"][i] = str(row).split(" |")[0]

# Delete the "nan" rows
df = df[df.Category != "nan"]

# Avoid oversampling with Toys & Games
df = df.drop(df[df["Category"] == "Toys & Games"].sample(frac=.50).index)


X = df['Product Name']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Adjust the number of features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))