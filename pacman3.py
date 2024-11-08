import pandas as pd

# Function for data preprocessing
def preprocess_data(dataset, drop_missing=True):
    # Drop any unnecessary columns that are not relevant for the analysis
    date_columns = ['POSTCODE', 'STEMERG', 'STEMERGEND', 'STAYHOME', 'END_STHM', 'FM_ALL', 'FM_END', 'QR_ALLST', 'QR_END', 'PUBDATE']
    columns_to_drop = dataset.columns.difference(date_columns)
    dataset = dataset.drop(columns=columns_to_drop)

    # Handle missing values
    if drop_missing:
        dataset = dataset.dropna()
    else:
        dataset = dataset.fillna('N/A')  # Fill missing values with 'N/A'

    return dataset

# Main function
def main():
    try:
        # Load the dataset
        dataset = pd.read_excel('policies.xlsx')

        # Check for missing values
        if dataset.isnull().values.any():
            print("Dataset contains missing values.")
            drop_missing = False  # Set drop_missing to False for filling missing values with 'N/A'
            # Preprocess the data by dropping or filling missing values
            preprocessed_data = preprocess_data(dataset, drop_missing)
        else:
            print("No missing values found in the dataset.")
            preprocessed_data = dataset

        # Write the preprocessed DataFrame to a file
        preprocessed_data.to_csv('policy.csv', index=False)

        # Print a success message
        print("Preprocessed data written to 'policy.csv'.")

    except FileNotFoundError:
        print("File 'policies.xlsx' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
