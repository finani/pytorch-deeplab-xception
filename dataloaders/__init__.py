from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensor
# %matplotlib inline

dataset_path = '/home/weebee/recyclables/baseline/input'
anns_file_path = os.path.join(dataset_path, 'train.json')

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1

print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)

# Count annotations
cat_histogram = np.zeros(nr_cats,dtype=int)
for ann in anns:
    cat_histogram[ann['category_id']] += 1

# Initialize the matplotlib figure
# f, ax = plt.subplots(figsize=(5,5))

# Convert to DataFrame
df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', 0, False)

# Plot the histogram
# plt.title("category distribution of train set ")
# plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df, label="Total", color="b")

sorted_temp_df = df.sort_index()
sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
print(sorted_df.info())
print(sorted_df)

category_names = list(sorted_df.Categories)

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        # Get the image_info using coco library
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # Load the image using opencv
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            # print("image_infos['id'] : {}".format(image_infos['id']) )
            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks_size : height x width
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Background = 0, Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # We can use Albumentations for image & mask transformation(or augmentation)
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
                masks =  masks.squeeze()

            return images, masks, image_infos

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return images, image_infos


    def __len__(self) -> int:
        return len(self.coco.getImgIds())



def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'recyclables':
        num_class = 12

        train_path = os.path.join(dataset_path, 'train.json')
        val_path = os.path.join(dataset_path, 'val.json')
        test_path = os.path.join(dataset_path, 'test.json')

        # collate_fn needs for batch
        def collate_fn(batch):
            return tuple(zip(*batch))

        train_transform = A.Compose([
            ToTensor(),
        ])

        val_transform = A.Compose([
            ToTensor(),
        ])

        test_transform = A.Compose([
            ToTensor(),
        ])

        # create own Dataset 2
        # train dataset
        train_dataset = CustomDataset(data_dir=train_path, mode='train', transform=train_transform)

        # validation dataset
        val_dataset = CustomDataset(data_dir=val_path, mode='val', transform=val_transform)

        # test dataset
        test_dataset = CustomDataset(data_dir=test_path, mode='test', transform=test_transform)


        # DataLoader
        train_loader = DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=1,
                                                collate_fn=collate_fn,
                                                drop_last=True)

        val_loader = DataLoader(dataset=val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=1,
                                                collate_fn=collate_fn,
                                                drop_last=True)

        test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=1,
                                                collate_fn=collate_fn,
                                                drop_last=True)
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
