'''
extract ReID features from testing data.
'''
import os
import argparse
import os.path as osp
import numpy as np
import torch
import time
import torchvision.transforms as T
from PIL import Image
import sys
from utils import FeatureExtractor
import torchreid

def make_parser():
    parser = argparse.ArgumentParser("reid")
    parser.add_argument("root_path", type=str, default=None)
    return parser

if __name__ == "__main__":

    args = make_parser().parse_args()
    data_root = args.root_path

    test_list = ['S001']
    dataset = 'test'
    sys.path.append(data_root+'/deep-person-reid')

    img_dir = os.path.join(data_root,'data/{}'.format(dataset))
    det_dir = os.path.join(data_root,'data/{}_det'.format(dataset))
    out_dir = os.path.join(data_root,'data/{}_emb'.format(dataset))

    models = {
              'osnet_x1_0':data_root+'/deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar',
              'osnet_ibn_x1_0':data_root+'/deep-person-reid/checkpoints/osnet_ibn_ms_m_c.pth.tar',
              'osnet_ain_x1_0':data_root+'/deep-person-reid/checkpoints/osnet_ain_ms_m_c.pth.tar',
              'osnet_x1_0_market':data_root+'/deep-person-reid/checkpoints/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
              'osnet_x1_0_msmt17':data_root+'/deep-person-reid/checkpoints/osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
             }
    
    
    model_names = ['osnet_x1_0','osnet_ibn_x1_0','osnet_ain_x1_0','osnet_x1_0','osnet_x1_0']
    

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for model_idx,name in enumerate(models):
        
        model_p = models[name]
        model_name = model_names[model_idx]

        print('Using model {}'.format(name))

        extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_p,
            device='cuda'
        )   

        scene = 'S001'
        out_path = os.path.join(out_dir,scene+'_{}.npy'.format(name))
        det_path = os.path.join(det_dir,scene+'.txt')
        dets = np.genfromtxt(det_path,dtype=str,delimiter=',')
        cur_frame = 0
        emb = np.array([None]*len(dets))
        start = time.time()
        print('processing scene {} with {} detections'.format(scene,len(dets)))
        for idx,(cam,frame,_,x1,y1,x2,y2,_) in enumerate(dets):
            x1,y1,x2,y2 = map(float,[x1,y1,x2,y2])
            if idx%1000 == 0:
                end = time.time()
                print('processing time :',end-start)
                start = time.time()
                print('process {}/{}'.format(idx,len(dets)))
            
            if cur_frame != int(frame):
                cur_frame = int(frame)
                img_path = os.path.join(img_dir,scene,cam,'frame',frame.zfill(5)+'.jpg')
                img = Image.open(img_path)
        
            img_crop = img.crop((x1,y1,x2,y2))
            img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
            feature = extractor(img_crop).cpu().detach().numpy()[0]
    
            # feature = feature/np.linalg.norm(feature)
            emb[idx] = feature
            
        np.save(out_path,emb)
