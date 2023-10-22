import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Sample categories for classification
categories = ["Toys & Games", "Home & Kitchen", "Clothing, Shoes & Jewelry", "Sports & Outdoors", "Baby Products", "Arts, Crafts & Sewing", "Office Products", "Hobbies", "Industrial & Scientific", "Health & Household", "Remote & App Controlled Vehicle Parts", "Tools & Home Improvement", "Remote & App Controlled Vehicles & Parts", "Pet Supplies", "Patio, Lawn & Garden", "Grocery & Gourmet Food", "Beauty & Personal Care", "Automotive", "Electronics", "Video Games", "Musical Instruments", "Movies & TV", "Cell Phones & Accessories"]

# Load the Excel file with titles and categories
input_file = "e_commerce_input.xlsx"
output_file = "e_commerce_output.xlsx"

# Load the dataset from the Excel file
df = pd.read_excel(input_file)

# Split the original data into only the first category in "Category"
for i, row in enumerate(df["Category"]):
     df["Category"][i] = str(row).split(" |")[0]

# Delete the "nan" rows
df = df[df.Category != "nan"]

# Avoid oversampling with Toys & Games
df = df.drop(df[df["Category"] == "Toys & Games"].sample(frac=.80).index)

# Creating a dataframe with 50%
# values of original dataframe
part_50 = df.sample(frac = 0.5)
 
# Creating dataframe with 
# rest of the 50% values
rest_part_50 = df.drop(part_50.index)

df = part_50

titles = df['Product Name'].values
category_labels = df['Category'].apply(categories.index).values

# Create a tokenizer and preprocess the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(titles)
X = tokenizer.texts_to_sequences(titles)
X = pad_sequences(X, maxlen=10)  # Pad sequences to a fixed length

# Convert categories to one-hot encoding
y = keras.utils.to_categorical(category_labels, num_classes=len(categories))

# Build and compile the neural network model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=10),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(categories), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)

df = rest_part_50

# Predict categories and confidence scores
predictions = model.predict(X)
predicted_categories = [categories[np.argmax(pred)] for pred in predictions]
confidence_scores = [round(max(pred), 2) for pred in predictions]

# Add the predicted categories and confidence scores to the DataFrame
df['Predicted_Category'] = predicted_categories
df['Confidence'] = confidence_scores

# Save the DataFrame with predictions to a new Excel file
df.to_excel(output_file, index=False)

print(f"Predictions saved to {output_file}")

match = len(df[df["Category"] == df["Predicted_Category"]])
count = len(df["Category"])
correctRatio = (match / count)
     
print("Total Correct:", match)
print("Correct Percentage:", correctRatio)