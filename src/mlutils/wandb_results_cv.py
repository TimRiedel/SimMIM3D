import pandas as pd
import wandb
from collections import defaultdict
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="adni-brats-finetune", choices=["adni-brats-finetune", "adni-brats-finetune-reconall", "brats-brats-finetune"])
args = parser.parse_args()

# Set your WandB API key
wandb.login()

# Define the mask ratios and train fractions
mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
train_fractions = [0.1, 0.25, 0.5, 1.0]

# Define the WandB entity and project
entity = "timriedel"
project = args.project

print(f"Getting runs for project {project}...")

# Define the save directory
save_dir = "/dhc/home/tim.riedel/bachelor-thesis/jobs/results/"

# Get all runs for the specified entity and project
api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# Create a dictionary to store the runs for each train fraction and mask ratio
runs_dict = defaultdict(lambda: defaultdict(list))

def save_df(df, filename):
    df_supervised = df[df['Mask Ratio'] == 'Baseline']
    df = df[df['Mask Ratio'] != 'Baseline']
    df = df.sort_values(by='Mask Ratio', ascending=False)
    df = pd.concat([df, df_supervised])

    filename = f"train_fraction_{train_fraction}.csv"
    df.to_csv(os.path.join(save_dir, project, filename), index=False)
    print(f"Saved {filename}.")
    

# Get all runs and assign them a train fraction and cv fold
for run in runs:
    name = run.name
    cv = 0
    if "_cv" in name:  # check if the run is a cv run and extract cv number
        cv = int(name.split("_cv")[1])

    if "Brats_FT" in name:  # finetune runs
        for train_fraction in train_fractions:
            for mask_ratio in mask_ratios:
                if f"MR:{mask_ratio}_TF:{train_fraction}" in name or f"MR:{mask_ratio}_TF:{train_fraction}_cv" in name:
                    runs_dict[train_fraction][cv].append(run)
    elif "Brats_SV" in name:  # supervised runs
        for train_fraction in train_fractions:
            if f"TF:{train_fraction}" in name or f"TF:{train_fraction}_cv" in name:
                runs_dict[train_fraction][cv].append(run)

# Iterate over train fractions and cv values and create CSV files
for train_fraction, cv_runs in runs_dict.items():
    tf_all_dfs = []
    for cv, runs in cv_runs.items():
        csv_dict = {"Mask Ratio": [], "Average": [], "TC": [], "ET": [], "WT": []}
        for run in runs:
            name = run.name
            try:
                mask_ratio = "Baseline" if "Brats_SV_TF" in name else name.split(":")[1].split("_")[0] # Mask ratio or supervised

                csv_dict["Average"].append(round(run.history(keys=["validation/dice"], x_axis="epoch").tail(1)["validation/dice"].values[0], 4))
                csv_dict["TC"].append(round(run.history(keys=["validation/dice_tc"], x_axis="epoch").tail(1)["validation/dice_tc"].values[0], 4))
                csv_dict["ET"].append(round(run.history(keys=["validation/dice_et"], x_axis="epoch").tail(1)["validation/dice_et"].values[0], 4))
                csv_dict["WT"].append(round(run.history(keys=["validation/dice_wt"], x_axis="epoch").tail(1)["validation/dice_wt"].values[0], 4))
                csv_dict["Mask Ratio"].append(mask_ratio)
            except Exception as e:
                print(f"Error processing run {name}: {e}")
        
        if csv_dict["Average"] and csv_dict["TC"] and csv_dict["ET"] and csv_dict["WT"]:  # only save CSV if there are valid entries
            cv_df = pd.DataFrame(csv_dict)
            tf_all_dfs.append(cv_df)
            
    combined_df = pd.concat(tf_all_dfs)
    tf_df = combined_df.groupby('Mask Ratio').mean().round(4).reset_index()
    save_df(tf_df, f"train_fraction_{train_fraction}.csv")
print("Results saved.")

