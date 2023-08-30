import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Load the CSV file into a DataFrame
df = pd.read_csv('SredjenExcel.csv')

# Function to extract modified coefficients for each peak
def extract_coefficients(data):
    coefficients = np.array([float(value) for value in data.strip('[]').split(', ')])
    peaks, _ = find_peaks(coefficients, height=0)  # Adjust parameters as needed
    
    modified_coefficients = []
    for peak_index in peaks:
        start_index = max(0, peak_index - 20)
        end_index = min(len(coefficients), peak_index + 31)  # 20 values before + 30 values after
        
        modified_coefficients.extend(coefficients[start_index:end_index])
    
    return modified_coefficients

# Apply the function to each row and create a new DataFrame
output_data = df['data'].apply(extract_coefficients)
output_df = pd.DataFrame({
    'id': df['id'],
    'age': df['age'],
    'data': output_data
})

# Save the new DataFrame to output.csv
output_df.to_csv('output.csv', index=False)
