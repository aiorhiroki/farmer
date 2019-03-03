import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation',
        usage='python main.py',
        description='This module demonstrates image segmentation.',
        add_help=True
    )

    parser.add_argument('-t', '--task', type=str, default='classification', help='Task name')
    parser.add_argument('-e', '--epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('-w', '--width', type=int, default=71, help='Input width')
    parser.add_argument('-ht', '--height', type=int, default=71, help='Input height')
    parser.add_argument('-b', '--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('-c', '--classes', type=int, default=10, help='Number of classes')
    parser.add_argument('-bb', '--backbone', type=str, default='none', help='Model backbone')
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', help='Model optimizer')
    parser.add_argument('-m', '--model', type=str, default='Xception', help='Model name')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')

    return parser
