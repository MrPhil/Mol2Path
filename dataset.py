from pathlib import Path
import re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _cell_id(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Cannot parse cell ID from {value!r}")
    return int(match.group(1))

def var_mean(m1, m2, m3, v1, v2, v3):
    n1, n2, n3 = 50, 50, 50
    total_mean = (n1 * m1 + n2 * m2 + n3 * m3) / (n1 + n2 + n3)
    within = (n1 - 1) * v1 + (n2 - 1) * v2 + (n3 - 1) * v3
    between = (
        n1 * (m1 - total_mean) ** 2
        + n2 * (m2 - total_mean) ** 2
        + n3 * (m3 - total_mean) ** 2
    )
    return (within + between) / (n1 + n2 + n3 - 1)

def load_imgdataset(dataset, cid_list):
    img_root = Path(dataset) / "image" / "data_224"
    cid_image_dic = {}
    for cid in cid_list:
        name = img_root / (str(cid) + ".png")
        with Image.open(name) as img:
            cid_image_dic[cid] = IMAGE_TRANSFORM(img.convert("RGB"))
    return cid_image_dic

class SynergyEncoderDataset(Dataset):
    def __init__(self, drug_row, drug_col, cell, Y_row, Y_col, Y,
                 matrix_drugExp, matrix_cellExp, device, embedding_file=None):
        embedding_file = Path(embedding_file) if embedding_file is not None else (
            Path(__file__).resolve().parent.parent
            / "Mol2PathVC-dataset"
            / "cell_LLM"
            / "cell_embeddings.npz"
        )
        data = np.load(embedding_file)
        cell_ids = [_cell_id(i) for i in data["cell_ids"]]
        cell_embeddings = data["cell_embeddings"]
        data.close()
        self.drug_row = drug_row
        self.drug_col = drug_col
        self.Y_row = Y_row
        self.Y_col = Y_col
        self.cell = cell
        self.c2emb = {cell_ids[k]: cell_embeddings[k] for k in range(len(cell_ids))}
        self.Y = Y
        self.device = device
        self.len = len(self.Y)
        self.drugExp = matrix_drugExp.apply(lambda row: row.to_list(), axis=1).to_dict()
        self.cellExp = matrix_cellExp.apply(lambda row: row.to_list(), axis=1).to_dict()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        drugR_exp = self.drugExp[self.drug_row[index]]
        drugC_exp = self.drugExp[self.drug_col[index]]
        cell_exp = self.cellExp[self.cell[index]]
        cell_emb = self.c2emb[self.cell[index]]
        return [
            torch.FloatTensor(drugR_exp),
            torch.FloatTensor(drugC_exp),
            torch.FloatTensor(cell_exp),
            torch.FloatTensor(cell_emb),
            torch.FloatTensor([self.Y[index]]),
            torch.FloatTensor([self.Y_row[index]]),
            torch.FloatTensor([self.Y_col[index]]),
            [self.drug_row[index], self.drug_col[index]],
        ]

class ImageDataset(Dataset):
    def __init__(self, drug_row, dataset_path, device):
        self.drug_row = drug_row
        self.device = device
        self.dataset_path = dataset_path
        self.len = len(self.drug_row)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        drugR_img = load_imgdataset(
            self.dataset_path, [self.drug_row[index]]
        )[self.drug_row[index]]
        return [torch.FloatTensor(drugR_img)]
