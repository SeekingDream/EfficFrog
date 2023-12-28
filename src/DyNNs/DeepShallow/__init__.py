
from .utils import extend_lists, cnn_to_sdn
from .utils import create_mobilenet, create_vgg16bn, create_resnet56
from .train_networks import train_sdn, train

from .train_networks import load_model as shallowdeep_load_model

from .architectures.SDNs.VGG_SDN import VGG_SDN
from .architectures.SDNs.MobileNet_SDN import MobileNet_SDN
from .architectures.SDNs.ResNet_SDN import ResNet_SDN
