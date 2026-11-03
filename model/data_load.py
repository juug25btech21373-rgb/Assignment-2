import os

from PIL import Image
from torch.utils import data


class Dataset_all(data.Dataset):
    def __init__(
        self, x_txt_input_ids, x_txt_attention_masks, x_img,
        x_kg1, x_kg2, x_kg_sim, y, transform, pathset
    ):
        self.data_txt_input_ids = x_txt_input_ids
        self.data_txt_attention_masks = x_txt_attention_masks
        self.data_img = x_img
        self.data_kg1 = x_kg1
        self.data_kg2 = x_kg2
        self.data_kg_sim = x_kg_sim
        self.label = y
        self.transform = transform
        self.pathset = pathset

    def __getitem__(self, index):
        image_id = str(self.data_img[index]).strip()
        image_path = os.path.join(
            self.pathset.path_img_data, image_id + ".jpg"
        )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                "Image is missing for CSV record {}: {}".format(index, image_path)
            )

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return (
            self.data_txt_input_ids[index],
            self.data_txt_attention_masks[index],
            image,
            self.data_kg1[index],
            self.data_kg2[index],
            self.data_kg_sim[index],
            self.label[index],
        )

    def __len__(self):
        return len(self.data_txt_input_ids)
