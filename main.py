from load_data import load_data
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import binom
from split_data import split
from pandas.plotting import scatter_matrix

housing= load_data()

print(housing.head())
print(housing.info())
print(housing.describe())



IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


    # extra cgit config --global user.emailode – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
# plt.show()


train_set, test_set= split(housing)

housing= train_set.copy()
print(housing.head())

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
s=housing["population"] / 100, label="population",
c="median_house_value", cmap="jet", colorbar=True,
legend=True, sharex=False, figsize=(10, 7))
save_fig("population_density")
# plt.show()

#find and plot corr
corr_matrix= housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1, grid=True)
# plt.show()

# feature combination
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
corr_matrix= housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#seprate train and test set
housing=train_set.drop("median_house_value", axis=1)
housing_labels=train_set["median_house_value"].copy()   


