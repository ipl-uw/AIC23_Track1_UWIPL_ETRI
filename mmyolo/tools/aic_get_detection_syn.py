# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import show_data_classes

dataset = 'test'
input = '/data/test'

testing_img_list = []


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--scene', default='None', help='Demo scene')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    parser.add_argument(
        '--to-labelme',
        action='store_true',
        help='Output labelme style label file')
    parser.add_argument(
        '--store-det', type=bool, default=True, help='save detection')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if args.deploy:
        switch_to_deploy(model)

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    # files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # ready for labelme format if it is needed
    to_label_format = LabelmeFormat(classes=dataset_classes)

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            show_data_classes(dataset_classes)
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    # start detector inference
    scenes = sorted(os.listdir(input))
    for scene in scenes:
        if args.scene and args.scene != scene: continue  # only demo 1 scene
        print('processing scene {}'.format(scene))
        cams = os.listdir(os.path.join(input, scene))
        cams = [cam for cam in cams if not cam.endswith('png')]
        detection = []
        for cam in cams:
            images = sorted(list_full_paths(os.path.join(input, scene, cam, 'img')))
            progress_bar = ProgressBar(len(images))
            for file in images:
                result = inference_detector(model, file)
                # img = mmcv.imread(file)
                # img = mmcv.imconvert(img, 'bgr', 'rgb')
                filename = os.path.basename(file)
                out_file = None if args.show else os.path.join(args.out_dir, filename)

                progress_bar.update()

                # Get candidate predict info with score threshold
                pred_instances = result.pred_instances[
                    result.pred_instances.scores > args.score_thr]

                if args.to_labelme:
                    # save result to labelme files
                    out_file = out_file.replace(
                        os.path.splitext(out_file)[-1], '.json')
                    to_label_format(pred_instances, result.metainfo, out_file,
                                    args.class_name)
                    continue

                if args.store_det:
                    # detection = []
                    image_path = os.path.abspath(result.metainfo['img_path'])
                    info = image_path.split('/')
                    image_name = info[-1].replace('.jpg', '')
                    cam = info[-3]
                    scene = info[-4]
                    for pred_instance in pred_instances:
                        pred_bbox = pred_instance.bboxes.cpu().numpy().tolist()[0]
                        score = pred_instance.scores.cpu().numpy().tolist()[0]
                        points = [cam, image_name, 1, pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3], score]
                        detection.append(points)

        output_path = 'data/{}_det/{}.txt'.format(dataset, scene)

        with open(output_path, 'a') as f:
            for cam, img_name, cls, x1, y1, x2, y2, score in detection:
                f.write('{},{},{},{},{},{},{},{}\n'.format(cam, int(img_name), cls, x1, y1, x2, y2, score))

    if not args.show and not args.to_labelme:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')

    elif args.to_labelme:
        print_log('\nLabelme format label files '
                  f'had all been saved in {args.out_dir}')


if __name__ == '__main__':
    main()
