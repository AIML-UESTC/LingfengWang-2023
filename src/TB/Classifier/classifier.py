import torch
import math
from src.TB.Classifier.se_resnet import se_resnet_18

def get_classifier_model():
    
    classification_model = se_resnet_18(num_classes=6)
    checkpoint = torch.load(r'./ckpts/TB/classifier.pth.tar')
    print('loaded')
    base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
    classification_model.cuda()
    classification_model.load_state_dict(base_dict)
    '''
    classification_model = torch.load(r'./ckpts/TB/classifier.pth.tar')
    classification_model.train(False)
    '''
    return classification_model