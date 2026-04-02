import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from preprocess import clean_data # Import from our sibling file

# 1. Get processed data
X, y = clean_data('../data/Student_Performance.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1) # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Train
print("Training Neural Network...")
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# 4. Save to the models/ directory
model.save('../models/student_model.h5')
print("Model saved to models/student_model.h5")