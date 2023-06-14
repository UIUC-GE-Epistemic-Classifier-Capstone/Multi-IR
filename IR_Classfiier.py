import torch
import os
from torch.utils import data
from PIL import Image
import torchvision
from torchvision import transforms
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Create a new torch dataset
class FLIRDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, length=-1):
        desired_samples = length
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))[:desired_samples]

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Create img_id Tensor
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Remake Annotation
        new_annotation = {"boxes": boxes, "labels": labels, "image_id": img_id, "area": areas, "iscrowd": iscrowd}
        ImgtoTensor = transforms.toTensor()
        img = ImgtoTensor(img)

        return img, new_annotation

    def __len__(self):
        return len(self.ids)


# path to your own data and coco file
train_data_dir = r'G:\School Stuff\Capstone\Epistemic\FLIR_ADAS_v2\images_thermal_train'
train_coco = r'G:\School Stuff\Capstone\Epistemic\FLIR_ADAS_v2\images_thermal_train\coco.json'
val_data_dir = r'G:\School Stuff\Capstone\Epistemic\FLIR_ADAS_v2\images_thermal_val'
val_coco = r'G:\School Stuff\Capstone\Epistemic\FLIR_ADAS_v2\images_thermal_val\coco.json'

# create own Dataset
my_dataset = FLIRDataset(root=train_data_dir,
                         annotation=train_coco,
                         )


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    # Batch size
    train_batch_size = 4

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 16
    num_epochs = 10
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            boxes = [len(bb.get('boxes', None)) for bb in annotations]
            if not all(boxes):
                continue
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
    torch.save(model, "IR Model")


if __name__ == "__main__":
    main()
