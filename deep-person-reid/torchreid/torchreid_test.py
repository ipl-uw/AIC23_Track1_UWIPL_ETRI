import os
import os.path as osp
#from torch.backends import cudnn
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
#from config import Config
from scipy.spatial import distance
import glob
import sys
sys.path.append('/home/u1436961/cycyang/deep-person-reid')
from utils import FeatureExtractor
import torchreid

#from processor import do_inference
#from utils.logger import setup_logger
def check_sport(seq,sports_dic):
    for sport in sports_dic:
        if seq in sports_dic[sport]:
            return sport

if __name__ == "__main__":
    #cfg = Config()
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    sports = ['basketball','football','volleyball']
    video_dic = {}
    splits_dir = '/work/u1436961/hsiangwei/dataset/splits_txt'
    for sport in sports: 
        file_path = osp.join(splits_dir,sport)
        my_file = open(file_path+".txt", "r")
        content = my_file.read()
        video_dic[sport] = content.split("\n")
        my_file.close()
    
    #model = make_model(cfg, 500)
    #model.load_param(cfg.TEST_WEIGHT)
    #device = "cuda"
    #model.to(device)
    #model.eval()
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='/home/u1436961/cycyang/deep-person-reid/log/resnet50/model/model.pth.tar-10',
        device='cuda'
    )   
    
    seqs = os.listdir('/home/u1436961/hsiangwei/sportsmot/OC_SORT/result/test/')
    for s_id,seq in enumerate(seqs):
        print(seq)
        seq = seq.replace('.txt','')
        print('processing ', s_id+1)
        sport_name = check_sport(seq,video_dic)
        # if sport_name != 'football':
        #     continue
        imgs = sorted(glob.glob('/work/u1436961/hsiangwei/dataset/test/{}/img1/*'.format(seq)))
        txt = '/home/u1436961/hsiangwei/sportsmot/OC_SORT/result/test/{}.txt'.format(seq)
        labels = np.genfromtxt(txt, delimiter=',', dtype=None)
        labels = np.sort(labels,axis=0)
        emb = np.array([None]*len(labels))
        for idx,label in enumerate(labels):
            if idx%1000==0:
                print(idx,'/',len(labels))
            frame_id = label[0]-1
            a,b,c,d = label[2],label[3],label[2]+label[4],label[3]+label[5]
            img = Image.open(imgs[frame_id]).crop((a,b,c,d))
            img = val_transforms(img.convert('RGB')).unsqueeze(0)
            #with torch.no_grad():
            #    img = img.to(device)
            #    feat = model(img).cpu().detach().numpy()[0]
            #emb[idx] = feat
        
            features = extractor(img)
            emb[idx] = features.cpu().detach().numpy()

        np.save('/home/u1436961/hsiangwei/sportsmot/embedding/emb_new/{}.npy'.format(seq),emb)
