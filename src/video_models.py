import torch
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
from pytorchvideo.models.head import create_vit_basic_head

def make_kinetics_mvit(ckpt_path : str, num_classes : int):
    spatial_size = 224
    temporal_size = 16
    embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
    atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
    pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
    pool_kv_stride_adaptive = [1, 8, 8]
    pool_kvq_kernel = [3, 3, 3]

    model = create_multiscale_vision_transformers(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        embed_dim_mul=embed_dim_mul,
        atten_head_mul=atten_head_mul,
        pool_q_stride_size=pool_q_stride_size,
        pool_kv_stride_adaptive=pool_kv_stride_adaptive,
        pool_kvq_kernel=pool_kvq_kernel,
        head=None
    )

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"], strict=False) # allow different head

    zeros = torch.zeros(1, 3, 16, 224, 224)
    y = model(zeros)

    out_dim = y.shape[-1]

    new_head = create_vit_basic_head(
            in_features=out_dim,
            out_features=num_classes,
            seq_pool_type="cls",
            dropout_rate=0.5,
            activation=None,
        )

    model.head = new_head

    return model


if __name__ == "__main__":
    from data import MultiModalHblDataset
    import torch
    import numpy as np
    from torchvision.transforms.functional import center_crop

    model = make_kinetics_mvit("models/mvit_b_16x4.pt", 2)

    model.eval()
    data = MultiModalHblDataset("/nfs/home/rhotertj/datasets/hbl/meta3d.csv", 16, sampling_rate=4)
    x = data[15525]

    x = x["frames"]
    x = np.transpose(x, (3, 0, 1, 2))
    x = torch.Tensor(x).unsqueeze(0)
    x = center_crop(x, 224)

    y = model(x)
    print(y.shape, y.argmax())


