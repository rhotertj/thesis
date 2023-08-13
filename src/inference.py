from data import MultiModalHblDataset, LabelDecoder
import multimodal_transforms as mmt
from lit_data import collate_function_builder

from video_models import make_kinetics_mvit
from graph_models import GAT, GCN, GIN, PositionTransformer
from torchvision import transforms
import torch
import numpy as np 

def main():
    batch_size = 1

    # setup models

    # model = make_kinetics_mvit("models/mvit_b_16x4.pt", 3, "twin", batch_size)
    # model = PositionTransformer(dim_in=49, dim_h=256, num_classes=3, input_operation="linear", batch_size=batch_size)
    # model = GAT(
    #     dim_in=49,
    #     dim_h=256,
    #     num_layers=2,
    #     readout="mean",
    #     input_operation="linear",
    #     num_heads=8,
    #     num_classes=3,
    #     batch_size=batch_size
    # )

    # model = GIN(
    #     dim_in=49,
    #     dim_h=256,
    #     input_operation="linear",
    #     num_layers=2,
    #     learn_eps=False,
    #     batch_size=batch_size,
    #     num_classes=3
    # )

    model = GCN(
        dim_in=4,
        dim_h=64,
        readout="mean",
        input_operation="linear",
        num_heads=8,
        temporal_pooling=True,
        batch_size=batch_size,
        num_classes=3
    )

    model.eval()
    # setup data and desired batching
    data = MultiModalHblDataset(
        meta_path="/nfs/home/rhotertj/datasets/hbl/meta3d.csv",
        seq_len=16,
        sampling_rate=2,
        transforms=transforms.Compose([mmt.FrameSequenceToTensor(), mmt.Resize(size=(224,224))]),
        label_mapping=LabelDecoder(3),
        overlap=False            
    )
    val_collate = collate_function_builder(7, True, position_format="graph_per_timestep", relative_positions=False, team_indicator=True)

    instances = []
    for i in range(batch_size):
        instances.append(data[100 * i])

    batch = val_collate(instances)

    # trainable(!) model params
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # move to gpu
    device = torch.device("cuda")
    model = model.to(device)
    input_g = batch["positions"].to(device)
    input_x = batch["positions"].ndata["positions"].to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions,1))

    # warm up
    for _ in range(10):
        _ = model(input_g, input_x)

    # measurements
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(input_g, input_x)
            ender.record()
            # wait for gpu
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn, std_syn)

if "__main__" == __name__:
    main()