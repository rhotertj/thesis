import torch

from video_models import make_kinetics_mvit
from graph_models import GAT
from heads import create_default_head

class MultiModalModel(torch.nn.Module):

    def __init__(
        self,
        video_model_name: str,
        video_model_params : dict,
        graph_model_name : str,
        graph_model_params : dict,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.video_model = eval(video_model_name)(
            **video_model_params,
            num_classes=num_classes,
            head_type="pool",
        )
        self.graph_model = eval(graph_model_name)(
            num_classes=num_classes,
            use_head=False,
            **graph_model_params
        )
        self.head = create_default_head(
            input_dim=graph_model_params["dim_h"] + 768,
            output_dim=num_classes,
            activation=torch.nn.ReLU(),
            dropout=0.3
        )

        

    def forward(self, batch):
        graph = batch["positions"]
        frames = batch["frames"]

        g_repr = self.graph_model(graph, graph.ndata["positions"])
        vid_repr = self.video_model(frames)
        mm_repr = torch.concatenate([g_repr, vid_repr], dim=1)
        return self.head(mm_repr)



if "__main__" == __name__:
    from data.datasets import MultiModalHblDataset, ResampledHblDataset
    from lit_data import collate_function_builder
    from torchvision import transforms as t
    import video_transforms as vt
    import pytorchvideo.transforms as ptvt

    basic_transforms = t.Compose([
            vt.FrameSequenceToTensor(),
            t.Resize((224,224))
            ])

    collate_fn = collate_function_builder(7, True, None)

    dataset = MultiModalHblDataset("/nfs/home/rhotertj/datasets/hbl/meta3d.csv", 16, sampling_rate=4, transforms=basic_transforms, overlap=False)

    instances = []
    for i in range(8):
        x = dataset[i*25]
        instances.append(x)

    batch = collate_fn(instances)

    mm_model = MultiModalModel(
        graph_model_name="GAT",
        graph_model_params={
            "dim_in": 49,
            "dim_h": 128,
            "readout": "mean",
            "input_embedding": True,
            "num_heads": 8,
        },
        video_model_name="make_kinetics_mvit",
        video_model_params={
            "pretrained_path": "models/mvit_b_16x4.pt",
        },
        num_classes=3
    )
    mm_model(batch)

    