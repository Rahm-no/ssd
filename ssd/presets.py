import transforms as T
import time
import datetime

class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, mean=(123., 117., 104.)):



        if data_augmentation == 'hflip':
            print("rnouaj: hflip")

            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
        elif data_augmentation == 'ssd':
            print("rnouaj: 5 transformations")
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=list(mean)),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
           
        elif data_augmentation == 'ssdlite':
            print("rnouaj: 3 transformations")

            self.transforms = T.Compose([
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)

