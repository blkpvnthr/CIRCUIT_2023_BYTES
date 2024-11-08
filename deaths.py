import pandas as pd
import csv

# Replace 'your_data.csv' with the path to your CSV file
data_path = 'deaths-raw.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(data_path)

# Select the columns you want to keep
selected_columns = ['Lat', 'Long_', 'Combined_Key', 'Population']

# Filter the data for rows where the total deaths column is greater than 0
filtered_df = df[df.iloc[:, 12:].sum(axis=1) > 0]

# Calculate the 'Total Deaths' for each combined key using the 'DailyDeaths' columns
filtered_df['Total Deaths'] = filtered_df.iloc[:, 12:].sum(axis=1)

# Keep only the selected columns
filtered_df = filtered_df[selected_columns + ['Total Deaths']]

# Print the filtered DataFrame
print(filtered_df)

# Save the DataFrame to a CSV file
filtered_df.to_csv('deaths.csv', index=False)

print("Filtered DataFrame has been saved to 'deaths.csv'.")