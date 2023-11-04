import argparse
import logging

import matplotlib.pyplot as plt
import torch.utils.data
from utils.data import get_dataset
from utils.dataset_processing.grasp import detect_grasps,GraspRectangles
from models.common import post_process_output
from opts import parse_args_test
import cv2
import matplotlib

matplotlib.use("TkAgg")

if __name__ == '__main__':
    args = parse_args_test()
    print(args.network)
    print(args.use_rgb,args.use_depth)
    net = torch.load(args.network)
    device = torch.device("cuda:0")
    Dataset = get_dataset(args.dataset)

    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }
    ld = len(val_data)
    with torch.no_grad():
        batch_idx = 0
        fig = plt.figure(figsize=(20, 10))
        # ax = fig.add_subplot(5, 5, 1)
        # while batch_idx < 100:
        for id,(x, y, didx, rot, zoom_factor) in enumerate( val_data):
                # batch_idx += 1
                if id>24:
                    break
                print(id)
                print(x.shape)
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])
                gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=1)
                rgb_img=val_dataset.get_rgb(didx, rot, zoom_factor, normalise=False)
                # print(rgb_img)
                ax = fig.add_subplot(5, 5, id+1)
                ax.imshow(rgb_img)
                ax.axis('off')
                for g in gs_1:
                    g.plot(ax)
        plt.show()