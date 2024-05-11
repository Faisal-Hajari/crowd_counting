import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob 
from PIL import Image 

__all__ = [
    "ObjectCountingFolder"
    ]
class ObjectCountingFolder(Dataset):
    """Implements a PyTorch dataset class that accepts a path (data_path) that points to a directory following this structure:

        data_path:
        ├── images/
        │   ├── name_1.png
        │   └── ...
        ├── labels/
        │   ├── name_1.txt
        │   └── ...

        Where name_1.txt correspond to name_1.png and is a text file where each line contains a single head point in the format 'x{sep}y'.

        Args:
            data_path (str): The path to the dataset.
            transforms (callable, optional): Transformation to be applied to both the target and the image.
            sep: the string used to seperate x and y in the label files. defult to " ".
            
        .note:
        When passing transforms, use the ones provided in the dataset.transforms module as they preserve the labels of the dataset. 
        Finally, there's no need to pass ToTensor in the transforms as this is the default behavior of the module.
        """

    def __init__(self, data_path, transforms, sep:str = " ") -> None:
        super().__init__()
        self._check_data_validty(data_path)
        self.transforms = transforms
        self.data_path = data_path
        self.images = sorted(glob(os.path.join(data_path, "images")+"/*"))
        self.sep = sep


    def _check_data_validty(self, data_path) -> None:
        """Given a data_path return True if the dataset are valid False otherwise"""
        images = glob(os.path.join(data_path, "images", "*"))
        labels = glob(os.path.join(data_path, "labels","*.txt"))
        assert len(labels) == len(images), "number of labels and images don't match" 
        assert len(labels) > 0, "No data was detected"

        label_names = [os.path.splitext(os.path.basename(name))[0] for name in labels]
        image_names = [os.path.splitext(os.path.basename(name))[0] for name in images]
        unique_to_labels= [item for item in label_names if item not in image_names]
        unique_to_images = [item for item in image_names if item not in label_names]
        assert len(unique_to_labels) == len(unique_to_images) == 0, f"there is {len(unique_to_labels)} without images and {len(unique_to_images)} images without labels"

    def _get_labels(self, index):
        label = self.images[index].replace("images", "labels")
        label = os.path.splitext(label)[0] +".txt"
        with open(label, "r") as file: 
            points = file.readlines()
        points = [[float(point.split(self.sep)[0]), float(point.split(self.sep)[1])] for point in points]
        points = torch.tensor(points)
        return points 


    def _get_image(self, index): 
        image = self.images[index]
        image = Image.open(image)
        image = transforms.PILToTensor()(image)
        return image 


    def __getitem__(self, index:int) -> dict[torch.Tensor, torch.Tensor]:
        image = self._get_image(index)
        labels = self._get_labels(index)
        data = {"image":image, "label":labels}
        data = self.transforms(data) if self.transforms else data 
        return data 
    

    def __len__(self):
        return len(self.images)