from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the model
model = load_model('Epohe2000inputNurons512_outputcsv.h5')

# Assuming your input data is a list of lists
# where each sublist is a separate PPG signal
ppg_signals = [[-8001.620335906972, -10535.183932898373, ... ]] # Fill this with your actual data
correct_ages = [30, 37, 58] # Fill this with the actual ages

# If your model was trained on padded sequences,
# make sure to pad your input data in the same way
ppg_signals = pad_sequences(ppg_signals, maxlen=599, truncating='post', padding='post')

# Predict the age for each PPG signal
ages_predicted = model.predict(ppg_signals)

# Calculate error metrics
mae = mean_absolute_error(correct_ages, ages_predicted)
mse = mean_squared_error(correct_ages, ages_predicted)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# If you also want to print each individual prediction alongside the actual age
for i in range(len(correct_ages)):
    print(f"Zadata starost: {correct_ages[i]}, Predviđena starost:{ages_predicted[i][0]}")
