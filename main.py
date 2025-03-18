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

    def process_deaths(self, input_path='deaths-raw.csv', output_path='deaths2.csv'):
        """
        1) Reads the raw deaths data (deaths-raw.csv)
        2) Filters out rows with zero or missing population
        3) Calculates 'Total Deaths' if needed and saves to an output CSV
        """
        df = pd.read_csv(input_path)
        policy = pd.read_csv('policy.csv')
        baseline='2020-01-01'
        drop_missing=True
        # B: Keep relevant columns for policy date offsets
        # Adjust or add columns as needed
        date_cols = ['STEMERG', 'STAYHOME', 'FM_ALL']  # example policy date fields
        keep_cols = ['POSTCODE'] + date_cols
        # If the raw Excel definitely has them, we can filter:
        # but if not, we skip or do partial
        policy = policy[keep_cols] if all(c in policy.columns for c in keep_cols) else policy
        
        # C: Drop or fill missing
        if drop_missing:
            policy = policy.dropna()
        else:
            policy = policy.iloc[3:]

        # D: Save the cleaned policy to CSV)
        print(f"Cleaned policy data: {policy.head()}")

        # E: Convert date columns to numeric offsets
        print(policy.head())

        baseline_date = pd.to_datetime(baseline)
        for col in date_cols:
            if col in policy.columns:
                policy[col] = pd.to_datetime(policy[col], errors='coerce')
                policy[col + "_days"] = (policy[col] - baseline_date).dt.days

        print(policy.head())
        # Filter out invalid population rows
        sub_df = df.iloc[:, 12:]  # columns from 13th to the end

        df = pd.read_csv("deaths-raw.csv")  # or whichever file has these columns
        # For instance, we see columns up to index 11, so columns 12+ are the daily counts
        sub_df = df.iloc[:, 12:]  # columns 12 onward

        # We'll store the sum of unique values for each row in a new column
        df["Total Deaths"] = 0

        for idx, row in sub_df.iterrows():
            # row is a Series of the daily count columns for this specific row
            unique_vals = set(row.values)  # get distinct values in that row
            sum_of_unique = sum(unique_vals)  # add them up
            df.loc[idx, "Total Deaths"] = sum_of_unique

        # Now df has a new column with the sum of distinct daily values from cols 12+
        print(df[["UID", "Total Deaths"]].head())


        df = df[df['Population'] > 0].copy()
        
        # Select the columns you want to keep
        selected_columns = ['Province_State', 'Population']
        # Keep only the selected columns
        df = df[selected_columns + ['Total Deaths']]
        df.to_csv("deaths3.csv")
        df3 = pd.read_csv("deaths3.csv")
        df3['POSTCODE'] = policy['POSTCODE'].map(self.state_map)
        print(df3)
        df4 = df.groupby('Province_State', as_index=False)['Population'].sum()
        print(df.head())
        print(df4.head())
        df4['Total Deaths'] = df['Total Deaths']
        print(df4.head())
        df4.to_csv("deaths4.csv")
        # If 'Total Deaths' doesn't exist, create it by summing from col 12 onward
        #if 'Total Deaths' not in df.columns:
        #df3.merge(df, how='left', left_on='POSTCODE', right_on=' Province_State')

        #df3['State_Deaths'] = df3[''].groupby('POSTCODE', as_index=False)['Total Deaths'].sum()
        #df['State_Deaths'] = df.groupby('POSTCODE', as_index=False)['Total Deaths'].sum()
       
        print("Filtered DataFrame has been saved to 'deaths4.csv'.")

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

        df2 = pd.read_csv("deaths4.csv")
        print("Preview of 'deaths4.csv':\n", df2.head())

        # Load the shapefile
        gdf_map = geopandas.read_file(shapefile_path)
        correlation = gdf_map.merge(df2, how='left', left_on='NAME', right_on='Province_State')

        ax = correlation.plot(
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
        policy_excel='policies.xlsx',
        deaths_csv='death-raw.csv',
        pop = 'deaths4.csv',
        output_policy_csv='policy.csv',
        drop_missing=True,
        baseline='2020-01-01'
    ):
        """
        1) Reads 'policies.xlsx' with columns for policy effective dates.
        2) Cleans it, saving to 'policy.csv'.
        3) Converts policy date columns to numeric offsets from `baseline` (e.g. '2020-01-01').
        4) Loads 'deaths.csv' with spelled-out states and 'Total Deaths', maps them to 2-letter codes (POSTCODE).
        5) Merges policy & deaths on 'POSTCODE'.
        6) Computes correlation between each policy date offset and 'Total Deaths'.
        7) MELTs the result into a multi-index: (POSTCODE, PolicyType) => [EffectiveOffset, Total Deaths].

        The final melted DataFrame is returned at the end, with correlation matrix also printed to console.

        :param policy_excel:   Path to the raw policy Excel file.
        :param deaths_csv:     Path to the CSV with state-level 'Total Deaths'.
        :param output_policy_csv: Where to save the cleaned policy CSV.
        :param drop_missing:   If True, drop rows with missing data in policy. If False, fill with 'N/A'.
        :param baseline:       Baseline date string for computing day offsets. Default '2020-01-01'.
        :return:               A multi-index DataFrame with index=[POSTCODE, PolicyType],
                            columns=['Total Deaths','EffectiveOffset'].
        """
        
        # Step A: Read raw policy data
        try:
            policy_df = pd.read_excel(policy_excel)
        except FileNotFoundError:
            print(f"File '{policy_excel}' not found.")
            return None
        except Exception as e:
            print(f"Error reading '{policy_excel}': {e}")
            return None

        # B: Keep relevant columns for policy date offsets
        # Adjust or add columns as needed
        date_cols = ['STEMERG', 'STAYHOME', 'FM_ALL']  # example policy date fields
        keep_cols = ['POSTCODE'] + date_cols
        # If the raw Excel definitely has them, we can filter:
        # but if not, we skip or do partial
        policy_df = policy_df[keep_cols] if all(c in policy_df.columns for c in keep_cols) else policy_df
        
        # C: Drop or fill missing
        if drop_missing:
            policy_df = policy_df.dropna()
        else:
            policy_df = policy_df.fillna('N/A')

        # D: Save the cleaned policy to CSV
        
        policy_df.iloc[: 3:]
        policy_df.to_csv(output_policy_csv, index=False)
        print(f"Cleaned policy data saved to '{output_policy_csv}'.\nPreview:\n", policy_df.head())

        # E: Convert date columns to numeric offsets
        df_policy = pd.read_csv(output_policy_csv)
        baseline_date = pd.to_datetime(baseline)

        for col in date_cols:
            if col in df_policy.columns:
                df_policy[col] = pd.to_datetime(df_policy[col], errors='coerce')
                df_policy[col + "_days"] = (df_policy[col] - baseline_date).dt.days

        # Overwrite or use a different CSV name
        df_policy.iloc[:, 3:]
        df_policy.to_csv(output_policy_csv, index=False)
        print("\nNumeric offset columns added. Policy data now:\n", df_policy[3:].head())
        pop_df = pd.read_csv(pop)
        # F: Load deaths data, map spelled-out 'Province_State' -> 'POSTCODE')
        if not hasattr(self, 'state_map'):
            print("Error: self.state_map is missing. Define it in __init__ for name->abbrev mapping.")
            return df_policy

        # Create a 'POSTCODE' col from spelled-out states
        print(pop_df.head())
        policy_df = df_policy[3:]
        policy_df['Population'] = pop_df['Population']
        policy_df['Total Deaths'] = pop_df['Total Deaths']
        print(f"\ncorrelation policy + deaths saved to 'policies-to-deaths.csv'")
        print("\nPreview of merged policy + deaths:\n", policy_df.head())
        policy_df.to_csv("policies-to-deaths.csv")
        correlation = pd.read_csv("policies-to-deaths.csv")
        print(correlation)
        # H: Correlate policy date offsets w/ 'Total Deaths'
        offset_cols = [c for c in correlation.columns if c.endswith('_days')]
        offset_cols.append('Total Deaths')
        corr_matrix = correlation[offset_cols].corr()
        print("\nCorrelation matrix: policy effective date offsets vs. Total Deaths\n", corr_matrix)

        # I: MELT into a multi-index DataFrame:
        #    One row per (POSTCODE, policy), with [EffectiveOffset, Total Deaths].
        #    We'll use the date offset columns for the melt.
        policy_offset_cols = [c for c in offset_cols if c != 'Total Deaths']  # e.g. 'STEMERG_days','STAYHOME_days'
        df_melt = correlation.melt(
            id_vars=['POSTCODE','STAYHOME_days', 'FM_ALL_days', 'Population', 'Total Deaths'],
            value_vars=policy_offset_cols,
            var_name='PolicyType',
            value_name='EffectiveOffset'
        )
        # Now we have columns: [POSTCODE, Total Deaths, PolicyType, EffectiveOffset]
        df_melt['pct_of_pop_dead'] = (
        df_melt['Total Deaths'] / df_melt['Population']) * 100
        # Set a MultiIndex
        df_melt.set_index(['POSTCODE','PolicyType'], inplace=True)
        print("\nMulti-index DataFrame saved to 'correlation.csv'\n", df_melt.head(10))
        df_melt.to_csv("correlation.csv")
        # The final structure: index=[(AL, 'STEMERG_days'), (AL, 'STAYHOME_days'), ... ],
        # columns=['Total Deaths','EffectiveOffset'].

        return df_melt


# ------------------------ USAGE EXAMPLE --------------------------------- #
if __name__ == '__main__':
    # Instantiate the analysis class
    analysis = CovidAnalysis()

    # 1. Process the raw COVID-19 deaths data
    analysis.process_deaths(
        input_path='deaths-raw.csv',   # The raw CSV you uploaded
        output_path='deaths2.csv'
    )

    # 2. Perform clustering analysis 
    analysis.cluster_data(
        input_path='deaths2.csv', 
        num_clusters=5, 
        output_path='deaths_clustered.csv'
    )

    # 3. (Optional) Visualize the data on a map
    #    Note: You must have a compatible shapefile 
    #          and a matching key column in your CSV (e.g., state/county names).
    analysis.visualize_data(
        shapefile_path='map/cb_2018_us_state_500k.shp',
        clustering_csv='deaths_clustered.csv',
        map_key_column='Province_State',       
        metric_column='Total Deaths'
    )

    # 4. (Optional) Preprocess your policy data from the policies.xlsx file 
    analysis.pacman3(
       policy_excel='policies.xlsx',
        deaths_csv='deaths.csv',
        output_policy_csv='policy.csv',
        drop_missing=True,
        baseline='2020-01-01'
    )
    