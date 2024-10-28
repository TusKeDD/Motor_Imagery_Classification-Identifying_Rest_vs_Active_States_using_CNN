import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

# Load the first dataset
dataset_1 = np.load('1.npy')

# Load and append the second dataset
dataset_2 = np.load('10.npy')
dataset_combined = np.vstack((dataset_1, dataset_2))
print('Shape of combined dataset:', dataset_combined.shape)

# Split dataset into features (X) and labels (y)
X = dataset_combined[:, :-1]
y = dataset_combined[:, -1]

# Normalize the features (X) using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Group classes 0-9 into class 0, and keep class 10 as class 1
y_grouped = np.where(y == 10, 1, 0)

# Define windowing parameters and preprocessing
window_size = 24
step_size = 12

def create_windows(X, window_size, step_size):
    num_windows = (X.shape[0] - window_size) // step_size + 1
    X_windows = np.array([X[i:i + window_size] for i in range(0, X.shape[0] - window_size + 1, step_size)])
    return X_windows.reshape(num_windows, window_size, 64, 1)

X_windows = create_windows(X_normalized, window_size, step_size)

# Function to compute average label for each window
def assign_window_labels_avg(y, window_size, step_size):
    num_windows = (len(y) - window_size) // step_size + 1
    y_windows = []

    for i in range(0, len(y) - window_size + 1, step_size):
        # Extract the labels for the current window
        window_labels = y[i:i + window_size]

        # Calculate the average label in the window
        avg_label = np.mean(window_labels)

        # Round the average to the nearest integer if you need integer labels
        y_windows.append(round(avg_label))

    return np.array(y_windows)

# Use this function to assign labels based on the average of each window
y_windows_avg = assign_window_labels_avg(y_grouped, window_size, step_size)

# Display shape and other details after windowing
print("Shape of windowed feature data (X_windows):", X_windows.shape)
print("Shape of windowed label data (y_windows_avg):", y_windows_avg.shape)

# Check for any discrepancies or unique values in labels
print("\nUnique values in y_windows_avg after windowing (label distribution):", np.unique(y_windows_avg, return_counts=True))

# Display a few samples for verification
print("\nFirst 5 samples of X_windows (flattened):")
print(X_windows[:5].reshape(5, -1))

print("\nFirst 5 labels after windowing (y_windows_avg):", y_windows_avg[:5])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows_avg, test_size=0.1, random_state=42, stratify=y_windows_avg)

# Build the Keras Model based on the PyTorch architecture
model = Sequential()

# Adapted Layers
model.add(Input(shape=(24, 64, 1)))
model.add(Conv2D(16, kernel_size=(2, 4), strides=1, padding='same'))  # Conv1 layer
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 4)))

model.add(Conv2D(32, kernel_size=(2, 2), strides=1, padding='same'))  # Conv2 layer
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# Output Layer for binary classification
model.add(Dense(2, activation='softmax'))

# Compile the model with categorical crossentropy for binary softmax output
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=40, batch_size=128, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {test_acc:.4f}')

# Get Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Display results
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Class 0-9', 'Class 10']))
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

y_pred_train = np.argmax(model.predict(X_train), axis=1)

print("\nTraining Classification Report:\n", classification_report(y_train, y_pred_train, target_names=['Class 0-9', 'Class 10']))
train_conf_matrix = confusion_matrix(y_train, y_pred_train)
print("\nTraining Confusion Matrix:\n", train_conf_matrix)

# Plot training & validation accuracy and loss
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
