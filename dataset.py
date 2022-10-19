from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index]
