import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from geodatasets import get_path


clusteringCSV = "deaths-raw.csv"
regressionCSV = "deaths.cvs"

#translates the csv with the deaths points into a geopandas dataframe
df = pd.read_csv(clusteringCSV)
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Long_, df.Lat), crs="EPSG:4326"
)


#Setting up and plotting the heat map
#NOTE make sure territories are removed or it will result in error
df2 = pd.read_csv(regressionCSV)
USA = geopandas.read_file("map/cb_2018_us_state_500k.shp")
merged = USA.merge(df2, on='NAME')
ax = merged.plot(cmap="OrRd", edgecolor="black", legend = True, column = "Deaths")

# plots data points over map
gdf.plot(ax=ax, column = "9/30/20", legend = True)

plt.show()