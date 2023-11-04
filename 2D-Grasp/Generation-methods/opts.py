import argparse

def parse_args_train():
    parser = argparse.ArgumentParser(description='2D_Grasp')
    ####################################
    # General settings for all method 
    ####################################
    # grasp method
    parser.add_argument('--method', type=str, default="GRCNN", help='the grasp network method you need,'
                        'You can choose: TF-Grasp, GRCNN')
    # grasp dataset
    parser.add_argument('--dataset', type=str,default="cornell", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str,default="F:/NEU_Project_Code/GitHub/grasp-encyclopedia/Datasets/Cornell" ,help='Path to dataset')
    parser.add_argument('--outdir', type=str, default='F:/NEU_Project_Code/GitHub/grasp-encyclopedia/2D-Grasp/Generation-methods/output/model', help='Training Output Directory')
    args = parser.parse_args()
    
    ####################################
    # Different settings for different methods
    ####################################
    if args.method=='TF-Grasp':
        # Network
        # Dataset & Data & Training
        parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
        parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
        parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
        parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
        parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
        parser.add_argument('--vis', type=bool, default=False, help='vis')
        parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
        parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
        parser.add_argument('--val-batches', type=int, default=32, help='Validation Batches')
        # Logging etc.
        parser.add_argument('--description', type=str, default='', help='Training description')

    elif args.method=='GRCNN':
        # Network
        parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
        parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
        parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
        parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
        parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
        parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
        parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
        parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
        # Datasets
        parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
        parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
        parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
        parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')
        # Training
        parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
        parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
        parser.add_argument('--batches-per-epoch', type=int, default=2,
                        help='Batches per Epoch')
        parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
        parser.add_argument('--val-batches', type=int, default=2, help='Validation Batches')
        # Logging etc.
        
        parser.add_argument('--description', type=str, default='',
                        help='Training description')
        parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
        parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
        parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
        parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    
    args = parser.parse_args()
    return args



def parse_args_test():
    parser = argparse.ArgumentParser(description='Evaluate 2D-Grasp')
    # Network
    parser.add_argument('--network', type=str,default="F:/NEU_Project_Code/GitHub/grasp-encyclopedia/2D-Grasp/Generation-methods/output/model/231104_1634_GRCNN/epoch_00_iou_0.00",
                        help='Path to saved network to evaluate')
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str,default="cornell", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str,default="F:/NEU_Project_Code/GitHub/grasp-encyclopedia/Datasets/Cornell" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--vis', type=bool, default=False, help='vis')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=32, help='Validation Batches')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='F:/NEU_Project_Code/GitHub/grasp-encyclopedia/2D-Grasp/Generation-methods/output/visual', help='Training Output Directory')

    args = parser.parse_args()
    return args