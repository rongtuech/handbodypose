import argparse
import torch
from model.mini_model import OpenPoseLightning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, required=True)
    parser.add_argument("-o", '--output_name', type=str, required=True)
    args = parser.parse_args()

    net = OpenPoseLightning()
    net.load_state_dict(torch.load(args.weight))

    input = torch.randn(1, 3, 320, 320)
    input_names = ['data']
    output_names = ['pafs', 'heatmaps']

    torch.onnx.export(net, input, args.output_name,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)