import sys
import torch
from classification.data_loader import get_predict_loader
from classification.models.cnn_model import ConvolutionalClassifier
import pandas as pd


def load(model_fn, device):
    d = torch.load(model_fn, map_location=device)
    return d['config'], d['model']


if __name__ == "__main__":
    idx_for_adjust = {0: 2, 1: 0, 2: 1}
    idx_for_win = {0: 2, 1: 0, 2: 1}
    test_imgs_path = sys.argv[-1]
    model_fn = "./trained_models/imgSize.64.split.augmented.vanillaCnn.bs.128.epochs.20.pth"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config, state_dict = load(model_fn, device)
    model = ConvolutionalClassifier(3)
    model.load_state_dict(state_dict)
    predict_loader = get_predict_loader(config, 64, "./test_imgs")
    result = []
    model.eval()
    with torch.no_grad():
        for file_names, x in predict_loader:
            x = x.to(device)
            y_hat = model(x)
            prediction = torch.argmax(y_hat, dim=-1)
            for i in range(len(y_hat)):
                idx_adjusted = idx_for_adjust[int(prediction[i])]
                result.append([file_names[i], idx_for_win[idx_adjusted]])
    df = pd.DataFrame(result)
    df.to_csv("./output.txt", sep="\t", index=False, header=False)