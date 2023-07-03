import pandas as pd
import numpy as np
from pathlib import Path

root_dir = "/dhc/home/tim.riedel/bachelor-thesis/data/ADNI"

data_dir = Path(f"{root_dir}/adni_128_int")
data_info_path = Path(f"{root_dir}/dataset.csv")
seed = 42
train_frac = 0.6
val_frac = 0.2

pt_dxs = ['CN', 'AD', 'MCI', 'EMCI', 'LMCI','SMC']
pt_only_dxs = ['MCI', 'EMCI', 'LMCI', 'SMC']
ft_dxs = ['CN', 'AD']

df = pd.read_csv(data_info_path)

def preprocess_dataframe(df):
    npy_files = list(data_dir.glob('*'))
    img_uid = [path.name.split('.')[0].split('_')[-1] for path in npy_files]
    df = df[df.IMAGEUID.isin(img_uid)]

    pt_df = df[df.DX.isin(pt_dxs)]
    ft_df = df[df.DX.isin(ft_dxs)]
    return pt_df, ft_df


pt_all_df, ft_df = preprocess_dataframe(df)

print("---------------------------------- Finetuning Split -----------------------------------")
print()
ft_patients = ft_df.PTID.unique()
print(f"Number of all finetuning patients: \t\t{len(ft_patients)} \t{ft_dxs}")
print()

random_generator = np.random.RandomState(seed)
num_ft_patients = len(ft_patients)
permutation_ft = random_generator.permutation(num_ft_patients)

train_ft_last_idx = int(num_ft_patients * train_frac)
val_ft_last_idx = int(num_ft_patients * (train_frac + val_frac))

ft_train_patients = ft_patients[permutation_ft[:train_ft_last_idx]]
ft_val_patients = ft_patients[permutation_ft[train_ft_last_idx:val_ft_last_idx]]
ft_test_patients = ft_patients[permutation_ft[val_ft_last_idx:]]

print(f"Number of patients finetuning training: \t{train_ft_last_idx}")
print(f"Number of patients finetuning validation: \t{val_ft_last_idx - train_ft_last_idx}")
print(f"Number of patients finetuning test: \t\t{num_ft_patients - val_ft_last_idx}")
print()


print("----------------------------------- Pretraining Split ----------------------------------")
print()
pt_all_patients = pt_all_df.PTID.unique()
print(f"Number of all pretraining patients : \t\t{len(pt_all_df.PTID.unique())} \t{pt_dxs}")

pt_only_df = pt_all_df[~pt_all_df.PTID.isin(ft_df.PTID.unique())]
pt_only_patients = pt_only_df.PTID.unique()
print(f"Number of only pretraining patients : \t\t{len(pt_only_patients)} \t{pt_only_dxs}")
print()

print(f"Number of patients finetune training: \t\t{len(ft_train_patients)} \t(included in pretraining)")
ft_val_test_patients = np.concatenate((ft_val_patients, ft_test_patients))
print(f"Number of patients finetune validation + test: \t{len(ft_val_test_patients)} \t(excluded from pretraining)")

pt_patients = np.setdiff1d(pt_all_patients, np.concatenate((ft_val_patients, ft_test_patients)))
pt_df_unique = pt_all_df[pt_all_df.PTID.isin(pt_patients)]
print(f"Number of patients pretraining: \t\t{len(pt_df_unique.PTID.unique())}")
print()


# Exists a patient which is both in [CN, AD] and [MCI, EMCI, LMCI, SMC]?
intersection = set(ft_df.PTID).intersection(set(pt_only_df.PTID))
if intersection:
    print("There exists a patient with both [CN, AD] and [MCI, EMCI, LMCI, SMC] diagnoses.")
else:
    print("There exitsts no patient with both [CN, AD] and [MCI, EMCI, LMCI, SMC] diagnoses.")




