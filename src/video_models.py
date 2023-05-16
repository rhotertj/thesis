import torch
import torch.nn.functional as F
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
from pytorchvideo.models.head import SequencePool
from heads import create_mvit_twin_head, create_vit_vlad_head
from pooling import NetVLAD

def make_kinetics_mvit(pretrained_path : str, num_classes : int, head_type : str, batch_size : int, netvlad_clusters : int):
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

    ckpt = torch.load(pretrained_path)
    model.load_state_dict(ckpt["model_state"], strict=False) # allow different head

    
    zeros = torch.zeros(1, 3, 16, 224, 224)
    y = model(zeros)
    out_dim = y.shape[-1]

    if head_type == "classify":
        new_head = create_vit_vlad_head(
            in_features=out_dim,
            n_clusters=netvlad_clusters,
            out_features=num_classes,
            seq_pool_type="cls",
            dropout_rate=0.5,
            activation=torch.nn.Softmax,
        )
    elif head_type == "pool":
        new_head = SequencePool("cls")
        
        if netvlad_clusters > 0:
            netvlad = NetVLAD(
                dim=768,
                num_clusters=netvlad_clusters
            )
            new_head = torch.nn.Sequential(new_head, netvlad)

    elif head_type == "twin":
        new_head = create_mvit_twin_head(
            dim_in=768,
            num_classes=num_classes,
            n_clusters=netvlad_clusters,
            activation=F.relu,
            dropout=0.5
        )
    else:
        raise NotImplementedError(f"Unknown head type {head_type}")
    
    model.head = new_head

    return model


if __name__ == "__main__":
    from data.datasets import MultiModalHblDataset
    import torch
    import numpy as np
    from torchvision.transforms.functional import center_crop
    from lit_data import collate_function_builder
    from torchvision import transforms as t
    import video_transforms as vt

    model = make_kinetics_mvit("models/mvit_b_16x4.pt", num_classes=3, batch_size=8, netvlad_clusters=4, head_type="classify")

    model.eval()
    basic_transforms = t.Compose([
            vt.FrameSequenceToTensor(),
            t.Resize((224,224))
            ])

    collate_fn = collate_function_builder(
        epsilon=7,
        load_frames=True,
        mix_video=None,
        position_format="flattened",
        relative_positions=False
        )

    dataset = MultiModalHblDataset("/nfs/home/rhotertj/datasets/hbl/meta3d.csv", 16, sampling_rate=2, transforms=basic_transforms, overlap=False)

    instances = []
    for i in range(8):
        x = dataset[i*25]
        instances.append(x)

    batch = collate_fn(instances)

    x = batch["frames"]
    y = model(x)
    if isinstance(y, tuple):
        print(y[0].shape, y[1].shape)
    else:
        print(y.shape)

