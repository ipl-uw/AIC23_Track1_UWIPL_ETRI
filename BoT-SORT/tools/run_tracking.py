import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser("run tracking")
    parser.add_argument("root_path", default="/home/hsiangwei/Desktop/AICITY2023", type=str)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    root_path = os.path.join(args.root_path,'data')

    # single camera tracking
    os.system("python tools/aic_track.py {}".format(root_path))

    # clustering for synthetic data
    os.system("python tools/aic_hungarian_cluster.py {}".format(root_path))
    
    # clustering for real data
    os.system("python tools/aic_hungarian_cluster_S001.py {}".format(root_path))
