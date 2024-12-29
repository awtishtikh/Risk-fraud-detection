import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from keras import Sequential
# from tensorflow.keras.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.optimizers import Adam
import matplotlib.pyplot as plt
from data_processing import cleaned_df

# Step 1: Load the data into a DataFrame
data = pd.read_csv("credit_risk_dataset.csv")
df = pd.DataFrame(data)
# cleaned_df = df.dropna()

# Step 2: Preprocessing
# Encode categorical variables
categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
# df_encoded = pd.get_dummies(cleaned_df, columns=categorical_columns, drop_first=True, dtype=int)

# Separate features and target
X = cleaned_df.iloc[:, 1: -1].values
print(X)
y = cleaned_df["loan_status"]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
# print(X_train[:10])
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# print(X_train[:10])
# Step 4: Build the Neural Network model
model = Sequential([
    Dense(45, input_shape=(X_train.shape[1],), activation='tanh'),
    Dropout(0.3),
    Dense(55, activation='linear'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='crossentropy', metrics=['accuracy', 'mse'])

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, verbose=1)

# Step 6: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
