

import os
import torch
import numpy as np
import json
import shap
import torch.optim as opt
import torchvision as tv
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_score
from torch.utils.data import DataLoader
import ssl
import time
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context
base_dir = "data/fps"

transforms_dict = {
    "train": tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224),
        tv.transforms.RandomRotation(degrees=10),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.ColorJitter(brightness=0.15, contrast=0.15),
        tv.transforms.ToTensor()
    ]),
    "val": tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()
    ])
}


def dta_load(cla, batch_size, base_dir=base_dir):
    dta_path = os.path.join(base_dir, cla)
    dataset = tv.datasets.ImageFolder(
        root=dta_path,
        transform=transforms_dict[cla]
    )
    dta_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

    return dta_loader, len(dataset), dataset.classes



class PpiResNet(torch.nn.Module):

    def __init__(self, finetune=False):
        super(PpiResNet, self).__init__()

        # 冻结Resnet50参数用于特征提取
        self.resnet50 = tv.models.resnet50(pretrained=True)
        if not finetune:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        fc_inputs = self.resnet50.fc.in_features

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_inputs, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):

        self.resnet50.fc = self.fc
        return self.resnet50(x)


class PpiVgg(torch.nn.Module):

    def __init__(self, finetune=False):
        super(PpiVgg, self).__init__()

        # 冻结Resnet50参数用于特征提取
        self.vgg = tv.models.vgg11(pretrained=True)
        if not finetune:
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1000, 2)
        )

    def forward(self, x):
        out_vgg = self.vgg(x)

        return self.fc(out_vgg)


class PpiAlexNet(torch.nn.Module):

    def __init__(self, finetune=False):
        super(PpiAlexNet, self).__init__()

        # 冻结Resnet50参数用于特征提取
        self.alexnet = tv.models.alexnet(pretrained=True)
        if not finetune:
            for param in self.alexnet.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1000, 2)
        )

    def forward(self, x):
        out_alex = self.alexnet(x)

        return self.fc(out_alex)


class PpiGoogleNet(torch.nn.Module):

    def __init__(self, finetune=False):
        super(PpiGoogleNet, self).__init__()

        self.googlenet = tv.models.googlenet(pretrained=True)
        if not finetune:
            for param in self.googlenet.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1000, 2)
        )

    def forward(self, x):
        out_google = self.googlenet(x)

        return self.fc(out_google)


class PpiDenseNet(torch.nn.Module):

    def __init__(self, finetune=False):
        super(PpiDenseNet, self).__init__()

        self.densenet = tv.models.densenet201(pretrained=True)
        if not finetune:
            for param in self.densenet.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1000, 2)
        )

    def forward(self, x):
        out_dense = self.densenet(x)
        return self.fc(out_dense)


if __name__ == "__main__":
    model_path = 'model/PerspectiveIde_wts_0.935114072582358_0.8801161103047895_0.7266187050359713_0.8278688524590164_0.7739463601532568_20_2020-01-17-22%3A51%3A25_True_PpiResNet.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #PpiGoogleNet PpiResNet PpiDenseNet PpiAlexNet PpiVgg
    model = PpiResNet(finetune=True)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    data_loaders = {}
    dataset_size = {}
    batch_size = 128
    batch_size2 = 1
    data_loaders["train"], dataset_size["train"], class_name = dta_load("train", batch_size)
    data_loaders["val"], dataset_size["val"], _ = dta_load("val", batch_size2)
    _input = []
    _label = []
    i = 0
    for inputs, labels in data_loaders["train"]:
        i+=1
        _input.append(inputs)
        _label.append(_label)
        if i >=3:
            break
    bg = torch.cat(_input,dim=0)
    print(bg.shape)
    e = shap.GradientExplainer(model, bg)
    print('build explainer')
    j = 0
    for inputs, labels in data_loaders["val"]:
        j += 1
        start = time.time()
        shap_values = e.shap_values(inputs, ranked_outputs=1)[0][0]
        end = time.time()
        print(end-start)
        inputs = inputs.numpy()
        inputs = np.transpose(inputs, [0, 2, 3, 1])
        shap_values = np.transpose(shap_values, [0,2,3,1])
        shap.image_plot(shap_values, inputs, show=False, hspace=0.01, wspace=0.05)
        plt.savefig('resnet1/' + str(j) + '.pdf', dpi=100, bbox_inches='tight')
        plt.close('all')

    # for inputs, labels in data_loaders["val"]:
    #     if j>250:
    #         break
    #     j += 1
    #     outputs = model(inputs)
    #     _, preds = torch.max(outputs,1)
    #     print('labels:',labels)
    #     print('preds:',preds)
    #     if labels != preds:
    #     # start = time.time()
    #         shap_values = e.shap_values(inputs, ranked_outputs=1)[0][0]
    #         # end = time.time()
    #         # print(end-start)
    #         # print(labels)
    #         # print(shap_values)
    #         # print(len(shap_values))
    #         # print(shap_values.shape)
    #         inputs = inputs.numpy()
    #         inputs = np.transpose(inputs, [0, 2, 3, 1])
    #         # inputs = inputs.view(2, 224, 224, 3)
    #         shap_values = np.transpose(shap_values, [0,2,3,1])
    #         # print(shap_values.shape)
    #         # shap_values.reshape((2, 224, 224, 3))
    #         # shap_values[1] = shap_values[1].reshape((1, 224, 224, 3))
    #         # shap.image_plot(shap_values, inputs, show=False)
    #         shap.image_plot(shap_values, inputs, show=False, hspace=0.01, wspace=0.05)
    #         plt.savefig('label0pred1/' + str(j) + '.pdf', dpi=100, bbox_inches='tight')
    #         plt.close('all')

