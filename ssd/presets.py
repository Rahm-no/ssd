import transforms as T
import time
import datetime

class DetectionPresetTrain:
    def __init__(self, data_augmentation, csv_file_path, hflip_prob=0.5, mean=(123., 117., 104.)):



        if data_augmentation == 'hflip':
            print("rnouaj: hflip")

            self.transforms = T.Compose([
                T.RandomHorizontalFlip(csv_file_path,p=hflip_prob),
                T.ToTensor(csv_file_path),
            ])
        elif data_augmentation == 'ssd':
            print("rnouaj: 5 transformations")
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(csv_file_path),
                T.RandomZoomOut(csv_file_path,fill=list(mean)),
                T.RandomIoUCrop(csv_file_path),
                T.RandomHorizontalFlip(csv_file_path,p=hflip_prob),
                T.ToTensor(csv_file_path),
            ])
           
        elif data_augmentation == 'ssdlite':
            print("rnouaj: 3 transformations")

            self.transforms = T.Compose([
                T.RandomIoUCrop(csv_file_path),
                T.RandomHorizontalFlip(csv_file_path,p=hflip_prob),
                T.ToTensor(csv_file_path),
            ])
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self,csv_file_path):

        print("is the code going here?")

        self.transforms = T.ToTensor(csv_file_path)

    def __call__(self, img, target):
        return self.transforms(img, target)

