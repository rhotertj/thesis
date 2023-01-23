from data import MultiModalHblDataset
from video_models import make_kinetics_mvit
from utils import array2gif, has_action
from torchvision import transforms

def main():
    model = make_kinetics_mvit("models/mvit_b_16x4.pt", 2)

    model.eval()
    data = MultiModalHblDataset(
        "/nfs/home/rhotertj/datasets/hbl/meta3d.csv",
        seq_len=16,
        sampling_rate=4,
        label_mapping=has_action,
        transforms=transforms.Compose([transforms.CenterCrop(224)])
    )
    x = data[187]

    x = x["frames"].unsqueeze(0)
    print(x.shape)
    y = model(x)
    array2gif(x.detach().cpu().numpy(), "img/inference.gif", 10)
    print(y.shape, y.argmax())

if "__main__" == __name__:
    main()