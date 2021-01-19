# coding:utf-8
import torch
import numpy as np  
import models
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_pre_train_ignore_name(net, pre_train):
    if pre_train == '':
        print('the pre_train is null, skip')
        return
    else:
        print('the pre_train is %s' % pre_train)
        new_dict = {}
        pretrained_model = torch.load(pre_train, map_location=torch.device('cpu'))

        pre_keys = pretrained_model.keys()
        net_keys = net.state_dict().keys()
        print('net keys len:%d, pretrain keys len:%d' % (len(net_keys), len(pre_keys)))
        if len(net_keys) != len(pre_keys):
            print(
                'key lens not same, maybe the pytorch version for pretrain and net are difficent; use name load')
            for key_net in net_keys:
                strip_key_net = key_net.replace('module.', '')
                if strip_key_net not in pre_keys:
                    print('op: %s not exist in pretrain, ignore' % (key_net))
                    new_dict[key_net] = net.state_dict()[key_net]
                    continue
                else:
                    net_shape = str(net.state_dict()[key_net].shape).replace('torch.Size', '')
                    pre_shape = str(pretrained_model[strip_key_net].shape).replace('torch.Size', '')
                    if net.state_dict()[key_net].shape != pretrained_model[strip_key_net].shape:
                        print('op: %s exist in pretrain but shape difficenet(%s:%s), ignore' % (
                        key_net, net_shape, pre_shape))
                        new_dict[key_net] = net.state_dict()[key_net]
                    else:
                        print(
                            'op: %s exist in pretrain and shape same(%s:%s), load' % (key_net, net_shape, pre_shape))
                        new_dict[key_net] = pretrained_model[strip_key_net]

        else:
            for key_pre, key_net in zip(pretrained_model.keys(), net.state_dict().keys()):
                if net.state_dict()[key_net].shape == pretrained_model[key_pre].shape:
                    new_dict[key_net] = pretrained_model[key_pre]
                    print('op: %s shape same, load weights' % (key_net))
                else:
                    new_dict[key_net] = net.state_dict()[key_net]
                    print('op: %s:%s shape diffient(%s:%s), ignore weights' %
                                 (key_net, key_pre,
                                  str(net.state_dict()[key_net].shape).replace('torch.Size', ''),
                                  str(pretrained_model[key_pre].shape).replace('torch.Size', '')))

        net.load_state_dict(new_dict, strict=False)

class PredModel:
    def __init__(self, model_name, model_args, pre_train, resize=[224,224], mean=[128.,128.,128.], std=[1.,1.,1.]):
        # create the model
        arg_dict = {}
        arg_dict['class_num'] = 1000
        arg_dict['width_mult'] = 1.0
        self.model = models.__dict__[model_name](model_args)
        load_pre_train_ignore_name(self.model, pre_train)
        self.model.eval()
        self.model.cuda()

        self.resize = resize
        self.mean = mean
        self.std = std

        self.calss_map = models.imagenet_class_map

    def pred_img(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, self.resize)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = ((img_rgb - self.mean) / self.std).astype(np.float32)
        img = img.transpose(2, 0, 1)
        img_t = torch.from_numpy(img).cuda().unsqueeze(0)
        preds = self.model(img_t).squeeze()
        preds = torch.nn.functional.softmax(preds, dim=0)

        values, indices = preds.topk(3, dim=0, largest=True, sorted=True)
        cls_names = []
        cls_probs = []
        for cls, prob in zip(indices, values):
            cls_name = self.calss_map[str(cls.item())][0]
            cls_prob = prob.item()
            cls_names.append(cls_name)
            cls_probs.append(cls_prob)
        return img_rgb, cls_names, cls_probs


if __name__ == '__main__':
    model_name = 'ShuffleNetV2'
    model_arg = {'class_num': 1000, 'channel_ratio': 1.0}
    resize = (224, 224)
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    pred_model = PredModel('ShuffleNetV2', model_arg, 'models/shufflenetv2_x1-5666bf0f80.pth', resize, mean, std)
    img, cls_names, cls_probs = pred_model.pred_img('static/1.jpg')

    plt.subplot(2,1,1)
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    cls_names.reverse()
    cls_probs.reverse()
    prob_max = np.array(cls_probs).max()
    print(prob_max)
    b = plt.barh(range(len(cls_names)), cls_probs, color='#6699CC')
    plt.yticks([])
    for x, y in enumerate(cls_probs):
        plt.text(0 + prob_max / 40, x, '%s' % cls_names[x])
    plt.savefig('./temp.jpg')