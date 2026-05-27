from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tracker.nn.backbone.mobile_v3 import mobilenetv3_small, mobilenetv3_small_v3

BACKBONES = {
              'mobilenetv3_small': mobilenetv3_small,
              'mobilenetv3_small_v3': mobilenetv3_small_v3,
            }

def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs) 
