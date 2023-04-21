'''
Generation track1.txt for final submission.
'''
import os
import argparse
import numpy as np
import collections

def make_parser():
    parser = argparse.ArgumentParser("generate submission")
    parser.add_argument("root_path", default="/home/hsiangwei/Desktop/AICITY2023/data", type=str)
    return parser

args = make_parser().parse_args()
root_path = args.root_path
output_path = os.path.join(root_path,'submission.txt')

results = []

dataset = 'test'
synthetic = os.path.join(root_path,'final_n=15_dist200_pk_filter_margin_2')
real = os.path.join(root_path,'0324_new_offset_fixed_reassignment_iteratively_250_65_200_70_150_75_interpolation')

seqs_syn = sorted([seq for seq in os.listdir(synthetic) if seq.endswith('.txt')])
seqs_real = sorted([seq for seq in os.listdir(real) if seq.endswith('.txt')])

scenes = ['S001','S003','S009','S014','S018','S021','S022']

print('generating submission file for {} set'.format(dataset))

cams = set()

scene_set=set()
seq_ids=set()

for seq in seqs_real:
    
    ids = collections.defaultdict(list)    
    seq = seq.replace('.txt','')
    scene,cam = seq.split('_')
    scene_id = int(scene[1:])
    sct = np.genfromtxt(os.path.join(real,seq+'.txt'), delimiter=',', dtype=None)
        
    print('processing scene {} cam {}'.format(scene,cam))
    print('start {} end {}'.format(sct[0][0],sct[-1][0]))
    if scene not in scene_set:
        scene_set.add(scene)
        seq_ids=set()
    
    if scene not in scenes:
        continue
    
    cams.add(cam)
    
    cam = int(cam[1:])
    
    for frame_id,trk_id,x,y,w,h,_,_,_,_ in sct:
        global_id = scene_id*1000+trk_id
        seq_ids.add(global_id)
        results.append(
                        f"{cam},{global_id},{frame_id},{int(x)},{int(y)},{int(w)},{int(h)},-1,-1\n"
                    )
     
    print('scene {} cam {} id_set:{}'.format(scene,cam,seq_ids))
    print('total detections {}'.format(len(sct)))
    print('==========================')

for seq in seqs_syn:
    
    ids = collections.defaultdict(list)    
    seq = seq.replace('.txt','')
    scene,cam = seq.split('_')
    scene_id = int(scene[1:])
    sct = np.genfromtxt(os.path.join(synthetic,seq+'.txt'), delimiter=',', dtype=None)
        
    print('processing scene {} cam {}'.format(scene,cam))
    
    if scene not in scene_set:
        scene_set.add(scene)
        seq_ids=set()
    
    if scene not in scenes:
        continue
    
    cams.add(cam)
    
    cam = int(cam[1:])

    print('start {} end {}'.format(sct[0][0],sct[-1][0]))
    
    for frame_id,trk_id,x,y,w,h,_,_,_,_ in sct:
        global_id = scene_id*1000+trk_id
        seq_ids.add(global_id)
        results.append(
                        f"{cam},{global_id},{frame_id},{int(x)},{int(y)},{int(w)},{int(h)},-1,-1\n"
                    )
     
    print('scene {} cam {} id_set:{}'.format(scene,cam,seq_ids))
    print('total detections {}'.format(len(sct)))
    print('==========================')

assert len(cams) == 43

with open(output_path, 'w') as f:
    f.writelines(results)    