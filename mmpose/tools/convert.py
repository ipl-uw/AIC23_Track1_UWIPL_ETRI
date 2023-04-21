import json
import tqdm


cams = ['c00%d' % i for i in range(1, 8)]
for cam in cams:
    print(cam)
    with open('S001_%s_feet.txt' % cam, 'w') as f:
        keypoints = json.load(open('%s.json' % cam))
        tracking_results = open('tracking_result/S001_%s.txt' % cam)
        pid = 0
        prev_fid = -1
        for lid, line in enumerate(tqdm.tqdm(tracking_results)):
            f.write(line.rstrip())
            fid = line.rstrip().split(',')[0]
            if int(fid) != prev_fid:
                prev_fid = int(fid)
                pid = 0
            if fid in keypoints:
                kp = keypoints[fid][pid]['keypoints']
                f.write(',%f,%f,%f,%f,%f,%f\n' % (kp[15][0], kp[15][1], kp[15][2], kp[16][0], kp[16][1], kp[16][2]))
                pid += 1
            else:
                f.write(',0,0,0,0,0,0\n')
    print()
