from models.TF_network.swin import SwinTransformerSys

def get_GRCNN_network(network_name):
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from models.GRCNN_network.grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from models.GRCNN_network.grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from models.GRCNN_network.grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from models.GRCNN_network.grconvnet4 import GenerativeResnet
        return GenerativeResnet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))

def get_net(args):
    if args.method == 'TF-Grasp':
        net = SwinTransformerSys(in_chans=1*args.use_depth + 3*args.use_rgb, embed_dim=48, num_heads=[1, 2, 4, 8])
    elif args.method == 'GRCNN':
        GRCNN = get_GRCNN_network(args.network)
        net = GRCNN(
         input_channels=1*args.use_depth + 3*args.use_rgb,
         dropout=args.use_dropout,
         prob=args.dropout_prob,
         channel_size=args.channel_size
        )
    
    return net
    