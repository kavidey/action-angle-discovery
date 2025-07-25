import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from pathlib import Path
import re

# load the model predicted results
ngb_df = pd.read_csv("tables_for_analysis/NGBooster_result.csv")
ngb_df["pred_e"] = ngb_df["pred_e"] + ngb_df["e"]
ngb_df["pred_sini"] = ngb_df["pred_inc"] + np.sin(ngb_df['Incl.']*np.pi/180)

merged_df = pd.read_csv("merged_elements.csv")
merged_df_copy = merged_df
test_name_list = ngb_df["Des'n"].to_list()

ngb_subset = ngb_df[["Des'n", "pred_e", "error_e", "pred_inc", "error_inc", "pred_sini"]]
merged_df = pd.merge(merged_df, ngb_subset, on="Des'n", how="inner")

propa = merged_df["propa"]
prope = merged_df["prope"]
propsini = merged_df["propsini"]

def histogram_generation(d_cutoff, family_df):
    def convert_id(val):
        val_str = str(val)
        if val_str.isdigit():
            return int(val_str)
        return val_str
    family_df["PackedName"] = family_df["PackedName"].apply(convert_id)
    name_list = family_df["PackedName"]
    family_df = merged_df_copy[merged_df_copy["Des'n"].isin(name_list)]
    family_pred_df = ngb_df[ngb_df["Des'n"].isin(name_list)]
    # Calculate the percentage of machine learning prediction still classifies asteroids into the family
    family_pred_df = ngb_df[ngb_df["Des'n"].isin(name_list)]
    def calculate_d(a_p, delta_a_p, delta_e_p, delta_sin_i_p):
        numerator = 3e4  # 3 × 10^4 m/s
        denominator = math.sqrt(a_p)
        term1 = (delta_a_p / a_p) ** 2
        term2 = 2 * (delta_e_p ** 2)
        term3 = 2 * (delta_sin_i_p ** 2)
        inside_sqrt = (5 / 4) * term1 + term2 + term3
        d = (numerator / denominator) * math.sqrt(inside_sqrt)
        return d
    
    # create a slab enclosing all family members
    def slab_d_calc(family_df_copy, family_pred_df_copy, merged_df, command):
        if command == "osculating":
            columns_bound = {"a": "a", "e": "e", "sini": "Incl."}
            columns = {"a": "a", "e": "e", "sini": "Incl."}
        elif command == "proper":
            columns_bound = {"a": "propa", "e": "prope", "sini": "propsini"}
            columns = {"a": "propa", "e": "prope", "sini": "propsini"}
        elif command == "pred":
            # family_df_copy = family_pred_df_copy
            columns_bound = {"a": "propa", "e": "prope", "sini": "propsini"}
            columns = {"a": "propa", "e": "pred_e", "sini": "pred_sini"}

        adds = 0

        a_adds = (family_df_copy[columns_bound["a"]].max() - family_df_copy[columns_bound["a"]].min())/2
        e_adds = (family_df_copy[columns_bound["e"]].max() - family_df_copy[columns_bound["e"]].min())/2
        sini_adds = (family_df_copy[columns_bound["sini"]].max() - family_df_copy[columns_bound["sini"]].min())/2

        a_adds = 0
        e_adds = 0
        sini_adds = 0

        a_min, a_max = family_df_copy[columns_bound["a"]].min() - a_adds, family_df_copy[columns_bound["a"]].max() + a_adds
        e_min, e_max = family_df_copy[columns_bound["e"]].min() - e_adds, family_df_copy[columns_bound["e"]].max() + e_adds
        sini_min, sini_max = family_df_copy[columns_bound["sini"]].min() - sini_adds, family_df_copy[columns_bound["sini"]].max() + sini_adds

        slab_df = merged_df[
            (merged_df[columns["a"]] >= a_min) & (merged_df[columns["a"]] <= a_max) &
            (merged_df[columns["e"]] >= e_min) & (merged_df[columns["e"]] <= e_max) &
            (merged_df[columns["sini"]] >= sini_min) & (merged_df[columns["sini"]] <= sini_max)
        ]

        a_family, e_family, sini_family, names = family_df_copy[columns_bound["a"]].values, family_df_copy[columns_bound["e"]].values, family_df_copy[columns_bound["sini"]].values, family_df_copy["Des'n"].values
        
        d_results = []
        for idx, row in slab_df.iterrows():
            a = row[columns["a"]]
            e = row[columns["e"]]
            sini = row[columns["sini"]]
            name = row["Des'n"]
            
            for a_f, e_f, sini_f, name_f in zip(a_family, e_family, sini_family, names):
                if name == name_f:
                    continue
                else:
                    da = a_f - a
                    de = e_f - e
                    dsini = sini_f - sini
                    d = calculate_d(a, da, de, dsini)
                    d_results.append({
                        "name_asteroid": name,
                        "name_family_asteroid": name_f,
                        "d": d
                    })
        d_df = pd.DataFrame(d_results)

        return d_df
        
    d_df = slab_d_calc(family_df, family_pred_df, merged_df, "pred")
    d_df_single = d_df.drop_duplicates(subset = ["name_asteroid"])
    num_family_slab = len(d_df_single[d_df_single["name_asteroid"].isin(name_list)])
    filtered_df = d_df[d_df["d"] < float(d_cutoff.iloc[0])]
    filtered_df = filtered_df.drop_duplicates(subset = ["name_asteroid"])
    family_slab_df = filtered_df[filtered_df["name_asteroid"].isin(name_list)]

    return len(family_slab_df), num_family_slab, len(filtered_df)

df = pd.read_csv("asteroid_families_csv.txt")
dataset_path = Path('family_tables')
filenames = list(dataset_path.glob('*.csv'))
name_list = []
total_family_list = []
true_family_list = []
detected_family_list = []

def change_asteroid_name(name):
    if isinstance(name, str):
        first_part = name.split(' ')[0]
        match = re.match(r'^(?P<year>\d{4})(?P<letters>[a-zA-Z]+)(?P<numbers>\d+)$', name)
        if first_part.isdigit() or name.replace('.', '').isdigit():
            return name
        if match:
            year = match.group('year')
            letters = match.group('letters').upper() # Capitalize all letters
            numbers = match.group('numbers')
            return f"{year} {letters}{numbers}"
        else:
            return name.title()
    return name

dataset_path = Path('family_tables')
filenames = list(dataset_path.glob('*.csv'))
column_names = ['propa', 'prope', 'propsini', 'g', 's', 'H', 'NumOpps', 'PackedName', 'UnpackedName']
for filename in filenames:
    try:
        df_family = pd.read_csv(str(filename), header = None, names = column_names)
        family_name = str(filename).split("/")[-1].split(".")[0].split("_")[-2]
        family_name = change_asteroid_name(family_name)
        d = df[df["Name"] == family_name]["HCM Cut (m s^-1)"]
        number = df[df["Name"] == family_name]["Number of Members"]
        if number.iloc[0] > 50:
            true_family, total_family, detected_family = histogram_generation(d, df_family)
            name_list.append(filename)
            true_family_list.append(true_family)
            total_family_list.append(total_family)
            detected_family_list.append(detected_family)
        else: 
            continue
    except Exception as e:
        # This is a general catch-all for any other unexpected errors
        print(f"An unexpected error occurred while processing '{filename}': {e}. Skipping.")

plt.hist(np.array(true_family_list)/np.array(detected_family_list) * 100)
plt.xlabel("Precision of the test sample", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("Precision_plot")
plt.cla()

plt.hist(np.array(true_family_list)/np.array(total_family_list) * 100)
plt.xlabel("Percent of families detected", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("Percent_detected_plot")

print(len(true_family_list)/len(filenames))

precision = np.array(true_family_list)/np.array(detected_family_list) * 100
indices = [i for i, val in enumerate(precision) if val < 50]
for i in indices:
    print(name_list[i])
