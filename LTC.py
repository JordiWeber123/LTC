import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Set the input csv file to processed Amazon listings
input_file = 'combined_data.csv'

# Create the pandas dataframe from the given csv file
df = pd.read_csv(input_file)

# x is the NAME / PRODUCT / TITLE / LISTING (x is assumed to be the first column)
x = df['name']
# y is the CATEGORY of the corresponding product (y is assumed to be the second column)
y = df['main_category']

# Trains with half the data, and uses the rest to validate and test
# Used for demonstration purposes, otherwise, you can allow the model to train with the whole csv file.
x_train, x_test, y_train, y_test = train_test_split(

    x, y, test_size=0.5, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

classifier = MultinomialNB()
classifier.fit(x_train_tfidf, y_train)

# Predicts the test cases, and runs a classification report to demo results
y_pred = classifier.predict(x_test_tfidf)
print(classification_report(y_test, y_pred))

# Logic for allowing the user to categorize their own products / titles
while True:
    user_input = input('Enter a product title (or type "exit" to quit): ')

    if user_input.lower() == 'exit':
        break

    # Preprocess the user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Predict category and display confidence score
    predicted_category = classifier.predict(user_input_tfidf)[0]
    confidence_score = classifier.predict_proba(user_input_tfidf).max()

    # Print to console
    print(f'Predicted Category: {predicted_category}')
    print(f'Confidence Score: {confidence_score:.2f}\n')
