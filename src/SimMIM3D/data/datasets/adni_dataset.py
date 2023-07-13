from pathlib import Path
import numpy as np
import pandas as pd
from monai.data import CacheDataset

valid_dxs = ['CN', 'MCI', 'EMCI', 'LMCI', 'AD', 'SMC']

class AdniDataset(CacheDataset):
    def __init__(
        self,
        root_dir = None,
        pretrain_dxs = valid_dxs,
        finetune_dxs = ['CN', 'AD'],
        transform = None,
        section = "training",
        seed = 42,
        train_frac = 0.6,
        val_frac = 0.2,
        is_pretrain = True,
        cache_rate = 1.0,
        num_workers = 1
    ):
        self.data_dir = Path(f"{root_dir}/adni_128_int")
        self.data_info_path = Path(f"{root_dir}/dataset.csv")
        
        self.pt_classes = self.validate_dxs(pretrain_dxs)
        self.ft_classes = self.validate_dxs(finetune_dxs)

        df = pd.read_csv(self.data_info_path)
        pt_all_df, ft_df = self.preprocess_dataframe(df)
        self.split_ft_patients(ft_df, seed, train_frac, val_frac)

        if section in ["train", "training"]:
            if is_pretrain:
                pt_all_patients = pt_all_df.PTID.unique()
                # Patients from the validation and test sets of finetuning must not be included
                # in the pretraining train set
                self.pt_patients = np.setdiff1d(pt_all_patients, self.ft_val_test_patients)
                self.df = pt_all_df[pt_all_df.PTID.isin(self.pt_patients)]
            else:
                self.df = ft_df[ft_df.PTID.isin(self.ft_train_patients)]
        elif section in ["val", "validation"]:
            self.df = ft_df[ft_df.PTID.isin(self.ft_val_patients)]
        elif section == "test":
            self.df = ft_df[ft_df.PTID.isin(self.ft_test_patients)]
        else:
            raise ValueError(f"split {section} not a valid option")
        
        phase = "pretraining" if is_pretrain else "finetuning"
        print(f"Number of {phase} - {section} images: {len(self.df)}")
        print(f"Number of {phase} - {section} patients: {len(self.df.PTID.unique())}")
        print()

        self.idx_to_class = {i: j for i, j in enumerate(self.pt_classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

        super().__init__(self.get_data(), transform, cache_rate=cache_rate, num_workers=num_workers)

        if self.transform is not None:
            self.transform.set_random_state(seed)

    
    def validate_dxs(self, dxs):
        # Check for duplicates and valid classes
        if len(set(dxs)) != len(dxs) or not all(dx in valid_dxs for dx in dxs):
            raise ValueError(f"Diagnoses {dxs} is not a valid option.")
        return dxs


    def preprocess_dataframe(self, df):
        npy_files = list(self.data_dir.glob('*'))
        img_uid = [path.name.split('.')[0].split('_')[-1] for path in npy_files]
        df = df[df.IMAGEUID.isin(img_uid)]

        pt_df = df[df.DX.isin(self.pt_classes)]
        ft_df = df[df.DX.isin(self.ft_classes)]
        return pt_df, ft_df


    def split_ft_patients(self, ft_df, seed, train_frac, val_frac):
        ft_all_patients = ft_df.PTID.unique()
        ft_num_patients = len(ft_all_patients)
        random_generator = np.random.RandomState(seed)
        permutation_ft = random_generator.permutation(ft_num_patients)

        train_ft_last_idx = int(ft_num_patients * train_frac)
        val_ft_last_idx = int(ft_num_patients * (train_frac + val_frac))

        self.ft_train_patients = ft_all_patients[permutation_ft[:train_ft_last_idx]]
        self.ft_val_patients = ft_all_patients[permutation_ft[train_ft_last_idx:val_ft_last_idx]]
        self.ft_test_patients = ft_all_patients[permutation_ft[val_ft_last_idx:]]
        self.ft_val_test_patients = np.concatenate((self.ft_val_patients, self.ft_test_patients))


    def get_data(self):
        data = []
        for index in range(len(self.df)):
            img_uid = self.df.iloc[index]['IMAGEUID']
            diagnosis = self.df.iloc[index]['DX']
            image_path = list(self.data_dir.glob(f'file_{img_uid}.npy*'))[0]

            label = self.class_to_idx[diagnosis]

            item = {"image": image_path, "label": label, "diagnosis": diagnosis}
            data.append(item)
        return data
