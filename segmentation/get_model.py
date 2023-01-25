# download https://github.com/czczup/ViT-Adapter/releases/download/v0.2.3/mask2former_beit_adapter_large_896_80k_cityscapes.zip into segmentation/checkpoints

import os 

url = 'https://github.com/czczup/ViT-Adapter/releases/download/v0.2.3/mask2former_beit_adapter_large_896_80k_cityscapes.zip'
output = 'checkpoints/mask2former_beit_adapter_large_896_80k_cityscapes.zip'

if not os.path.exists(output):
    print('Downloading model...')
    os.system(f'wget {url} -O {output}')

    print('Unzipping model...')
    os.system(f'unzip {output} -d checkpoints')
    os.system(f'rm {output}')