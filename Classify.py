import torch
from IR_Classfiier import myOwnDataset, get_transform, collate_fn
import torchvision
import matplotlib as plt
import numpy as np

# path to your own data and coco file
test_data_dir = r'G:\School Stuff\Capstone\Epistemic\FLIR_ADAS_v2\video_thermal_test'
test_coco = r'G:\School Stuff\Capstone\Epistemic\FLIR_ADAS_v2\video_thermal_test\coco.json'

# create own Dataset
num_samples = 10
my_dataset = myOwnDataset(root=test_data_dir,
                          annotation=test_coco,
                          transforms=get_transform(),
                          length=num_samples
                          )
batch_size = 4
data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

def main():
    model = torch.load("IR Model", map_location=device)
    i = 0

    # move model to the right device
    model.eval()
    len_dataloader = len(data_loader)
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        boxes = [len(bb.get('boxes', None)) for bb in annotations]
        if not all(boxes):
            continue
        predictions = model(imgs, annotations)
        # losses = 0
        for loss_dict in predictions:
            losses = sum(loss for loss in loss_dict['scores']).item()
        for idx, annotation in enumerate(annotations):
            # img = (imgs[idx]*255).detach().cpu().numpy().astype(np.uint8)
            img = (imgs[idx]*255).to(torch.uint8)
            boxes = annotation['boxes']
            labels = annotation['labels'].reshape(-1, 1)
            image = torchvision.utils.draw_bounding_boxes(img, boxes, labels)
            plt.imshow(image.permute(1, 2, 0))

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')


if __name__ == "__main__":
    main()
