import torch
import timm
import pytorch_lightning as pl


def efficient_net(model_name='b3'):
    if model_name == 'b8':
        efficient_net_model = timm.create_model('tf_efficientnet_b8',
                                                    pretrained=True, num_classes=0)
    elif model_name == 'b3':
        efficient_net_model = timm.create_model('tf_efficientnet_b3',
                                                    pretrained=True, num_classes=0
        )
        
    data_config = timm.data.resolve_model_data_config(efficient_net_model)
    transforms = timm.data.create_transform(**data_config, is_training=True)
    val_transforms = timm.data.create_transform(**data_config, is_training=False)

    return efficient_net_model, transforms, val_transforms