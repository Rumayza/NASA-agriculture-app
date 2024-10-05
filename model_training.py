
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_crop_yield_model():
    """
    Train a Random Forest model to predict crop yield using processed data.
    """
    # Load cleaned data
    data = pd.read_csv('cleaned_data/weather_data_cleaned.csv')
    # Define input features and target variable
    X = data[['T2M', 'TS', 'PRECTOTCORR']]
    y = data['Yield']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    
    # Save trained model to disk
    with open('models/crop_yield_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def train_vegetation_health_model():
    """
    Train a Convolutional Neural Network (CNN) model to predict vegetation health using satellite images.
    """
    # Placeholder for image data loading (e.g., NDVI images)
    # Assuming X_train, y_train are image data and labels
    X_train = ...  # Load image data
    y_train = ...  # Load labels

    # Define CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    
    # Save trained model to disk
    model.save('models/vegetation_health_model.h5')

if __name__ == "__main__":
    train_crop_yield_model()
    train_vegetation_health_model()
