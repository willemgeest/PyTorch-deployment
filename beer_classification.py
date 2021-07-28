import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt


def get_classes():
    return ['Amstel', 'Bavaria', 'Desperados', 'Grolsch', 'Heineken', 'Hertog Jan', 'Jupiler']


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        class_names = get_classes()
        model_name = "beerchallenge_resnet18_7brands_rotated.pth"

        # define the resnet 50
        #torch.hub.set_dir('.')
        self.resnet = resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, len(class_names))
        self.resnet.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1,
                                      self.resnet.layer2,
                                      self.resnet.layer3,
                                      self.resnet.layer4)

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # extract the features
        x = self.features(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x

def beer_classification(img, class_int=None):
    # get classes
    class_names = get_classes()
    # init the resnet
    resnet = ResNet()
    # set the evaluation mode
    _ = resnet.eval()

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])  # normalize images for R, G, B (both mean and SD)

    img_transformed = test_transforms(img)
    # add 1 dimension to tensor
    img_transformed = img_transformed.unsqueeze(0)
    # forward pass
    pred = resnet(img_transformed)

    # tranforms tensors with results to probabilities
    sm = torch.nn.Softmax(dim=1)  # use softmax to convert tensor values to probs (dim = columns (0) or rows (1) have to sum up to 1?)
    probabilities = sm(pred)

    probabilities = probabilities[0]

    # get the gradient of the output with respect to the parameters of the model
    if class_int==None:
        pred[:, pred.argmax()].backward() # heatmap of class with highest prob
    else:
        pred[:, class_int].backward()

    # pull the gradients out of the model
    gradients = resnet.get_gradient()
#    gradients.shape
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = resnet.get_activations(img_transformed).detach()
    # len(activations[0])
    activations.shape
    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap (values between 0 and 1
    heatmap /= torch.max(heatmap)

    # make the heatmap to be a numpy array
    heatmap = heatmap.numpy()
    # interpolate the heatmap, resize to original size
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize((img.size[0], img.size[1]))
    #Image.fromarray(np.uint8(255 * heatmap)).show()
    # Get the color map by name:
    cm = plt.get_cmap('jet')

    heatmap = np.asarray(heatmap)/255
    # Apply the colormap like a function to any array:
    heatmap = cm(heatmap)
    #heatmap.shape  (648, 180, 4)
    heatmap = np.delete(heatmap, 3, 2)
    # heatmap.shape  (648, 180, 3)
    heatmap = heatmap * 255
    mix = (0.8 * np.asarray(img) + 0.5 * heatmap) # (80% of original picture + 50% of heatmap )
    #clip checks for values outside 0 and 255. This is unnessecary because we've limited between 0 and 1 earlier
    mix = np.clip(mix, 0, 255).astype(np.uint8)
    # save heatmap
    #Image.fromarray(mix).save(heatmap_location)

    return class_names[pred.argmax()], probabilities, Image.fromarray(mix)

