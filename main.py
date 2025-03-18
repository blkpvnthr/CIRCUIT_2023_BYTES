import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class CovidAnalysis:
    def __init__(self):
        """
        Initialize any shared variables or configurations needed
        across the analysis methods.
        """
        # Move the dictionary to an instance variable.
        self.state_map = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY"
            # Add DC, PR, GU, etc. if needed
        }

    def process_deaths(self, input_path='deaths-raw.csv', output_path='deaths.csv'):
        """
        1) Reads the raw deaths data (deaths-raw.csv)
        2) Filters out rows with zero or missing population
        3) Calculates 'Total Deaths' if needed and saves to an output CSV
        """
        df = pd.read_csv(input_path)

        # Filter out invalid population rows
        df = df[df['Population'] > 0].copy()

        # If 'Total Deaths' doesn't exist, create it by summing from col 12 onward
        if 'Total Deaths' not in df.columns:
            df['Total Deaths'] = df.iloc[:, 12:].sum(axis=1)

        print(df.head(5))
        df.to_csv(output_path, index=False)
        print(f"Processed deaths data saved to '{output_path}'.")

    def cluster_data(self, input_path='deaths.csv', num_clusters=5, output_path='deaths_clustered.csv'):
        """
        Performs K-Means clustering on the processed data.
        """
        df = pd.read_csv(input_path)

        if 'Deaths_per_100k' not in df.columns:
            df['Deaths_per_100k'] = (df['Total Deaths'] / df['Population']) * 100000

        needed_cols = ['Lat', 'Long_', 'Deaths_per_100k']
        if not all(col in df.columns for col in needed_cols):
            print("Error: 'Lat', 'Long_', or 'Deaths_per_100k' missing.")
            return

        X = df[needed_cols]

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        
        df['Cluster'] = kmeans.labels_
        df.to_csv(output_path, index=False)
        print(f"K-Means clustering complete. Clustered data saved to '{output_path}'.")

    def visualize_data(self, 
                       shapefile_path='map/cb_2018_us_state_500k.shp',
                       clustering_csv='deaths_clustered.csv',
                       map_key_column='NAME',
                       metric_column='Total Deaths'):
        """
        Generates a choropleth map using GeoPandas based on a chosen metric.
        """
        # Debug check
        df_deaths = pd.read_csv("deaths.csv")
        print("Deaths CSV columns:\n", df_deaths.columns)

        # Make a geodataframe out of raw deaths for point plotting
        raw_df = pd.read_csv("deaths-raw.csv")
        gdf_points = geopandas.GeoDataFrame(
            raw_df, 
            geometry=geopandas.points_from_xy(raw_df['Long_'], raw_df['Lat']), 
            crs="EPSG:4326"
        )

        df2 = pd.read_csv("deaths.csv")
        print("Preview of 'deaths.csv':\n", df2.head())

        # Load the shapefile
        gdf_map = geopandas.read_file(shapefile_path)
        merged = gdf_map.merge(df2, how='left', left_on='NAME', right_on='Province_State')

        ax = merged.plot(
            cmap="OrRd",
            edgecolor="black",
            legend=True,
            column=metric_column,
            figsize=(10,6)
        )

        # Plot data points
        # e.g. if you want to color by "9/30/20"
        if "9/30/20" in gdf_points.columns:
            gdf_points.plot(ax=ax, column="9/30/20", legend=True, markersize=5)
        else:
            gdf_points.plot(ax=ax, color="blue", markersize=5)

        plt.title(f"{metric_column} Choropleth + Points Overlay")
        plt.show()

    def pacman3(
        self,
        excel_file='policies.xlsx',
        deaths_csv='deaths.csv',
        output_policy_csv='policy.csv',
        output_numeric_csv='policy_numeric.csv',
        drop_missing=True,
        policy_state_col='POSTCODE',
        numeric_date_cols=None,
        baseline_date='2020-01-01'
    ):
        """
        All-in-one policy data workflow that:

        1) Reads an Excel file of policies (e.g., 'policies.xlsx'),
        2) Keeps relevant columns and optionally drops/fills missing values,
        3) Saves to 'policy.csv',
        4) Converts any date columns to numeric (#days since `baseline_date`),
        5) Merges the numeric policy data with 'deaths.csv' (mapping spelled-out 
            state names -> two-letter codes),
        6) Computes correlation between numeric policy columns and 'Total Deaths'.

        :param excel_file: Path to the policy Excel file (default 'policies.xlsx')
        :param deaths_csv: Path to the processed deaths CSV (default 'deaths.csv'), 
                        must contain 'Province_State' + 'Total Deaths'.
        :param output_policy_csv: Where to save the initial cleaned policy CSV 
                                (default 'policy.csv').
        :param output_numeric_csv: Where to save the numeric date version 
                                (default 'policy_numeric.csv').
        :param drop_missing: If True, drops rows with missing cells. If False, 
                            fills them with 'N/A'.
        :param policy_state_col: Column in the policy data to hold 2-letter codes 
                                (default 'POSTCODE').
        :param numeric_date_cols: List of date columns to parse to numeric, 
                                or None to use a default set.
        :param baseline_date: The baseline used for calculating day offsets 
                            (default '2020-01-01').

        :return: A pandas DataFrame containing the merged data, plus a printed
                correlation matrix in the console.
        """
        import pandas as pd

        # 1) Read Excel
        try:
            data = pd.read_excel(excel_file)
        except FileNotFoundError:
            print(f"File '{excel_file}' not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading '{excel_file}': {e}")
            return None

        # We want to keep only these relevant columns from the Excel
        default_date_columns = [
            'POSTCODE', 'STEMERG', 'STEMERGEND', 'STAYHOME', 'END_STHM',
            'FM_ALL', 'FM_END', 'QR_ALLST', 'QR_END', 'PUBDATE'
        ]
        if numeric_date_cols is None:
            numeric_date_cols = default_date_columns[1:]  # exclude 'POSTCODE'

        # 2) Drop or fill missing
        if drop_missing:
            data = data.dropna()
        else:
            data = data.fillna('N/A')

        # 3) Keep only our default_date_columns if they exist in the data
        keep_cols = [col for col in default_date_columns if col in data.columns]
        data = data[keep_cols]

        # 4) Save the initial cleaned policy CSV
        data.to_csv(output_policy_csv, index=False)
        print(f"Initial policy data processed & saved to '{output_policy_csv}'.\nPreview:\n", data.head())

        # 5) Convert date columns to numeric
        df = pd.read_csv(output_policy_csv)
        baseline = pd.to_datetime(baseline_date)

        for col in numeric_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col + "_days"] = (df[col] - baseline).dt.days

        # Save the numeric version
        df.to_csv(output_numeric_csv, index=False)
        print(f"Numeric policy data saved to '{output_numeric_csv}'.\nPreview:\n", df.head())

        # 6) Merge with deaths.csv
        # Use the self.state_map for spelled-out names -> abbreviations
        policy_df = pd.read_csv(output_numeric_csv)
        deaths_df = pd.read_csv(deaths_csv)

        # Create a new col 'POSTCODE' in the deaths DataFrame if not already present
        # by mapping from spelled-out 'Province_State' (like 'Alabama' -> 'AL').
        # If you have a self.state_map stored in your class:
        if not hasattr(self, 'state_map'):
            print("Warning: no self.state_map found. Please define the dictionary in __init__.")
            self.state_map = {}  # or define an empty dict

        # If the user wants to unify them, let's do so:
        deaths_df['POSTCODE'] = deaths_df['Province_State'].map(self.state_map)

        merged = pd.merge(
            policy_df,
            deaths_df,
            how='left',
            on=policy_state_col  # merges on 'POSTCODE'
        )
        print("\nPreview of merged policy & deaths:\n", merged.head())

        if 'Total Deaths' not in merged.columns:
            print("Error: 'Total Deaths' missing in merged data. Make sure you processed the deaths data first.")
            return merged

        # 7) Correlate numeric policy columns with 'Total Deaths'
        numeric_cols = [c for c in merged.columns if c.endswith('_days')]
        numeric_cols.append('Total Deaths')
        corr_matrix = merged[numeric_cols].corr()
        print("\nCorrelation matrix between policy date offsets & Total Deaths:\n", corr_matrix)

        return merged


    def parse_policy_dates_to_numeric(self, policy_csv='policy.csv', output_csv='policy_numeric.csv'):
        """
        Reads policy CSV, converts date columns to numeric (#days since 2020-01-01).
        """
        df = pd.read_csv(policy_csv)
        baseline = pd.to_datetime('2020-01-01')

        date_columns = [
            'STEMERG', 'STEMERGEND', 'STAYHOME', 'END_STHM',
            'FM_ALL', 'FM_END', 'QR_ALLST', 'QR_END', 'PUBDATE'
        ]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col + "_days"] = (df[col] - baseline).dt.days

        print("Policy data after date-to-numeric conversion:\n", df.head())
        df.to_csv(output_csv, index=False)
        print(f"Numeric policy data saved to '{output_csv}'.")

    def merge_policy_and_deaths(self, policy_csv='policy.csv', deaths_csv='deaths.csv'):
        """
        Reads policy CSV, reads deaths CSV, merges them on shared 2-letter code.
        'POSTCODE' in policy => 'Province_State' is spelled out in deaths => map using self.state_map.
        """
        policy_df = pd.read_csv(policy_csv)
        deaths_df = pd.read_csv(deaths_csv)

        # Create a new col 'POSTCODE' in deaths by mapping spelled-out names to two-letter codes
        deaths_df['POSTCODE'] = deaths_df['Province_State'].map(self.state_map)

        # Merge
        merged = pd.merge(policy_df, deaths_df, how='left', on='POSTCODE')
        print("Preview of merged policy & deaths:\n", merged.head())
        return merged

    def correlate_policy_dates_to_deaths(self,
                                         policy_csv='policy.csv',
                                         deaths_csv='deaths.csv',
                                         baseline_date='2020-01-01'):
        """
        1) Reads a policy CSV that has:
           - 'POSTCODE' (two-letter code like "AL","AK"...)
           - date columns (e.g. 'STEMERG','STAYHOME') in string format
        2) Converts those date columns into numeric "days since `baseline_date`".
        3) Loads 'deaths.csv' which has 'Province_State' (e.g. "Alabama") and 'Total Deaths'.
        4) Maps spelled-out states in 'deaths.csv' -> 2-letter codes using `self.state_map`.
        5) Merges the two DataFrames on 'POSTCODE'.
        6) Computes a correlation matrix among the numeric date columns and 'Total Deaths'.

        :param policy_csv: Path to the policy CSV that has a single row per state, with date columns.
        :param deaths_csv: Path to the CSV that has a single row per state, with 'Province_State' and 'Total Deaths'.
        :param baseline_date: Baseline for calculating day offsets (default '2020-01-01').

        :return: A correlation matrix (pandas DataFrame).
        """

        # 1) Load policy data
        policy_df = pd.read_csv(policy_csv)

        # Identify date columns from the policy
        # Adjust these to match your actual date column names
        date_columns = [
            'STEMERG',      # e.g., State of emergency start
            'STEMERGEND',   # e.g., State of emergency end
            'STAYHOME',     # e.g., Stay-at-home start
            'END_STHM',     # e.g., Stay-at-home end
            'FM_ALL',       # e.g., Face mask start
            'FM_END',       # e.g., Face mask end
            'QR_ALLST',     # e.g., Quarantine start
            'QR_END',       # e.g., Quarantine end
            'PUBDATE'       # e.g., Public vax eligibility
        ]

        # 2) Convert policy date columns to numeric days
        baseline = pd.to_datetime(baseline_date)
        for col in date_columns:
            if col in policy_df.columns:
                # Convert text => datetime => numeric offset
                policy_df[col] = pd.to_datetime(policy_df[col], errors='coerce')
                policy_df[col + "_days"] = (policy_df[col] - baseline).dt.days

        # 3) Load deaths data
        deaths_df = pd.read_csv(deaths_csv)

        # 4) Map spelled-out states (like "Alabama") to 2-letter codes (like "AL")
        deaths_df['POSTCODE'] = deaths_df['Province_State'].map(self.state_map)

        # 5) Merge on 'POSTCODE'
        merged = pd.merge(
            policy_df,
            deaths_df,
            how='left',
            on='POSTCODE'
        )

        if 'Total Deaths' not in merged.columns:
            print("Error: 'Total Deaths' not found in merged data. Ensure your deaths CSV has that column.")
            return None
        
        # Gather numeric date columns (those that end in "_days")
        numeric_cols = [c for c in merged.columns if c.endswith("_days")]
        # Add 'Total Deaths' to compute correlation
        numeric_cols.append('Total Deaths')

        # 6) Compute correlation among date offsets vs. total deaths
        corr_matrix = merged[numeric_cols].corr()
        print("\nCorrelation matrix between policy effective dates & Total Deaths:\n", corr_matrix)
        return corr_matrix

if __name__ == '__main__':
    # Instantiate the analysis class
    analysis = CovidAnalysis()

    # ------------------ (1) Process Deaths ------------------ #
    analysis.process_deaths(
        input_path='deaths-raw.csv',
        output_path='deaths.csv'
    )

    # ------------------ (2) Cluster Deaths ------------------ #
    analysis.cluster_data(
        input_path='deaths.csv',
        num_clusters=5,
        output_path='deaths_clustered.csv'
    )

    # ------------------ (3) Visualize (Optional) ------------------ #
    # (Requires a shapefile + consistent column in both shapefile and CSV.)
    analysis.visualize_data(
        shapefile_path='map/cb_2018_us_state_500k.shp',
        clustering_csv='deaths_clustered.csv',
        map_key_column='Province_State',
        metric_column='Total Deaths'
    )

    # ------------------ (4) Preprocess Policy Data ------------------ #
    # e.g. remove extra columns, drop/fill missing values, etc.
    analysis.pacman3(
        excel_file='policies.xlsx',
        output='policy.csv',
        drop_missing=True
    )

    # ------------------ (5) Convert Policy Dates -> Numeric ------------------ #
    analysis.parse_policy_dates_to_numeric(
        policy_csv='policy.csv',
        output_csv='policy_numeric.csv'
    )

    # ------------------ (6) Merge Policy & Deaths Data on Abbreviations ------------------ #
    # Make sure this method includes or references a dictionary from spelled-out states
    # to two-letter abbreviations, or vice versa.
    merged_df = analysis.merge_policy_and_deaths(
        policy_csv='policy_numeric.csv', 
        deaths_csv='deaths.csv'
    )

    # Quick preview
    print("\nPreview of merged policy & deaths DataFrame:\n", merged_df.head())

    # ------------------ (7) Correlate Numeric Policy Columns w/ Deaths ------------------ #
    correlation_matrix = analysis.correlate_policy_to_deaths(
        policy_csv='policy_numeric.csv',
        deaths_csv='deaths.csv',
        policy_state_col='POSTCODE',
        deaths_state_col='STUSPS'
    )
    print("\nCorrelation matrix:\n", correlation_matrix)
