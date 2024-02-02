import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load CSV data
df = pd.read_csv('./data/Testing.csv')

# Preprocess data
# ... (handle missing values, encode categorical variables, etc.)
print(df.columns.array)

# Separate features and target variable
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text data
vectorizer = CountVectorizer(max_features=1000)  # Adjust the number based on your needs
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Convert sparse matrix to dense array
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Check shapes
print("X_train_dense shape:", X_train_dense.shape)
print("y_train shape:", y_train.shape)
print("X_test_dense shape:", X_test_dense.shape)
print("y_test shape:", y_test.shape)

# Build a simple neural network
model = Sequential([
    Dense(64, input_dim=X_train_dense.shape[1], activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Check shapes before fitting
print("X_train_dense shape before fitting:", X_train_dense.shape)
print("y_train shape before fitting:", y_train.shape)

model.fit(X_train_dense, y_train, epochs=10, batch_size=32, validation_data=(X_test_dense, y_test))
