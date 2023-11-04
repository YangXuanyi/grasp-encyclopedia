import datetime
import os
import sys
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from traning import train, validate
from utils.data import get_dataset
from opts import parse_args_train
from net_init import get_net
logging.basicConfig(level=logging.INFO)

def run():
    args = parse_args_train()
    
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    save_folder = os.path.join(args.outdir, net_desc) + args.method
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=False,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    net = get_net(args)
    #net = SwinTransformerSys(in_chans=input_channels, embed_dim=48, num_heads=[1, 2, 4, 8])
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    listy = [x * 2 for x in range(1,1000,5)]
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=listy,gamma=0.5)
    logging.info('Done')
    summary(net, (input_channels, 224, 224))
    f = open(os.path.join(save_folder, 'net.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()
    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        print("current lr:",optimizer.state_dict()['param_groups'][0]['lr'])
        # for i in range(5000):
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # schedule.step()
        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if epoch%1==0 or iou>best_iou:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
        best_iou = iou


if __name__ == '__main__':
    run()
