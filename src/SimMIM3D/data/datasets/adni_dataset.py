from pathlib import Path
import numpy as np
import pandas as pd
from monai.data import NumpyReader
from monai.data import Dataset

class AdniDataset(Dataset):
    def __init__(
        self,
        root_dir = None,
        classes = None,
        transform = None,
        section = "training",
        seed = 42,
        train_frac = 0.7,
        val_frac = 0.2,
    ):
        self.data_dir = Path(f"{root_dir}/adni_128_int")
        self.data_info_path = Path(f"{root_dir}/dataset.csv")
        
        valid_classes = ['CN', 'MCI', 'AD']
        if classes is None:
            self.classes = valid_classes 
        elif all(elem in classes for elem in valid_classes):
            self.classes = classes
        else:
            raise ValueError(f"Classes {self.classes} not a valid option.")

        df = pd.read_csv(self.data_info_path)
        self.df = self.preprocess_dataframe(df)

        # split of patients
        self.patients = self.df.PTID.unique()
        random_generator = np.random.RandomState(seed)
        num_patients = len(self.patients)
        permutation = random_generator.permutation(num_patients)

        train_last_idx = int(num_patients * train_frac)
        val_last_idx = int(num_patients * (train_frac + val_frac))

        if section == "training":
            train_patients = self.patients[permutation[:train_last_idx]]
            self.df = self.df[self.df.PTID.isin(train_patients)]
        elif section in ["val", "validation"]:
            valid_patients = self.patients[permutation[train_last_idx:val_last_idx]]
            self.df = self.df[self.df.PTID.isin(valid_patients)]
        elif section == "test":
            test_patients = self.patients[permutation[val_last_idx:]]
            self.df = self.df[self.df.PTID.isin(test_patients)]
        else:
            raise ValueError(f"split {section} not a valid option")

        self.idx_to_class = {i: j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

        self.transform = transform
        if self.transform is not None:
            self.transform.set_random_state(seed)

    def preprocess_dataframe(self, df):
        npy_files = list(self.data_dir.glob('*'))
        img_uid = [path.name.split('.')[0].split('_')[-1] for path in npy_files]
        df = df[df.IMAGEUID.isin(img_uid)]
        df = df[df.DX.isin(self.classes)]
        return df


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index: int):
        img_uid = self.df.iloc[index]['IMAGEUID']
        diagnosis = self.df.iloc[index]['DX']

        image_path = list(self.data_dir.glob(f'file_{img_uid}.npy*'))[0]

        label = self.class_to_idx[diagnosis]

        data = {"image": image_path, "label": label, "diagnosis": diagnosis}
        if self.transform is not None:
            return self.transform(data)
        return data
