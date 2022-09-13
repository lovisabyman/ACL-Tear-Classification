import argparse
from evaluate import evaluate_for_cam
from tensorflow.keras.utils import plot_model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--use_gpu')
    parser.add_argument('--plane')
    return parser

def cam(split, model_path, diagnosis, plane, use_gpu):
  model, preds, labels = evaluate_for_cam(split, model_path, diagnosis, plane, use_gpu)
  # model.summary()
  plot_model(model)


if __name__ == '__main__':
    args = get_parser().parse_args()
    cam(args.split, args.model_path, args.diagnosis, args.plane, False)
