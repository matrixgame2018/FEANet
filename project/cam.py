import os, argparse, time, sys,  torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from grad_cam.pytorch_grad_cam import GradCAM
import cv2
from grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image
from model.EAEFNet import EAEFNet
#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='your model name')
parser.add_argument('--weight_name', '-w', type=str, default='your model name')
parser.add_argument('--file_name', '-f', type=str, default='your.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
args = parser.parse_args()
#############################################################################################

class SemanticSegmentationTarget:
    def __init__(self, mask):
        print(np.unique(mask))
        self.number = np.unique(mask)
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
    def __call__(self, model_output):
        model_output = model_output.view(9,480,640)
        self.mask = self.mask.reshape(480,640)
        model_out = (model_output[2,:,:] + model_output[3,:,:] + model_output[4,:,:])
        return ((model_out) * self.mask).sum()

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    model_dir = os.path.join('./runs/', args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    conf_total = np.zeros((args.n_class, args.n_class))

    model = FAENet_p_p.FATNet_pp(args.n_class)

    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()

    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1  # do not change this parameter!
    test_dataset = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    ave_time_cost = 0.0

    model.eval()
    for it, (images, labels, names) in enumerate(test_loader):
        print(names[0])
        th_img = cv2.imread(args.data_dir + "images_th/" + names[0] + "_th.png", 1)
        rgb_img = cv2.imread(args.data_dir + "images_rgb/" + names[0] + "_rgb.png", 1)
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        logit, logits = model(images)
        start_time = time.time()
        output = logits + logit
        sem_classes = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        car_mask = output.argmax(1).cpu().numpy()
        car_mask_float = np.float32(car_mask)
        #TODO choose the layer you want to visualize the cam map
        target_layers = [model.FA_encoder.encoder_rgb_layer4]
        #######################################################################################################
        targets = [SemanticSegmentationTarget(car_mask_float)]
        with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
            ####################################################################################################
            grayscale_cam = cam(input_tensor=images, targets=targets,aug_smooth=True, eigen_smooth=False)[0, :]
            cam_image = show_cam_on_image(rgb_img.astype(np.float32) / 255, grayscale_cam, use_rgb=False)
            ####################################################################################################
        cv2.imwrite("./RUN/" + names[0] + ".png", cam_image)




