import torch

from video_models import make_kinetics_mvit
from graph_models import GAT, GIN, PositionTransformer
from heads import create_default_head, BasicTwinHead

class MultiModalModel(torch.nn.Module):

    def __init__(
        self,
        video_model_name: str,
        video_model_params : dict,
        graph_model_name : str,
        graph_model_params : dict,
        num_classes: int,
        batch_size: int,
        head_type: str,
        train_head_only: bool,
        graph_model_ckpt: str = "",
        video_model_ckpt: str = "",
    ) -> None:
        super().__init__()
        self.video_model = eval(video_model_name)(
            **video_model_params,
            num_classes=num_classes,
            batch_size=batch_size,
            head_type="pool",
        )
        self.graph_model = eval(graph_model_name)(
            num_classes=num_classes,
            batch_size=batch_size,
            head_type="pool",
            **graph_model_params
        )

        if head_type == "basic":
            self.head = create_default_head(
                input_dim=graph_model_params["dim_h"] + 768, # mvit hidden dim
                output_dim=num_classes,
                activation=torch.nn.ReLU(),
                dropout=0.3
            )
        elif head_type == "twin":
            self.head = BasicTwinHead(
                dim_in=graph_model_params["dim_h"] + 768,
                num_classes=num_classes,
                activation=torch.nn.ReLU(),
                dropout=0.3
            )

        if not graph_model_ckpt == "":
            self.load_graph_model_from_ckpt(graph_model_ckpt)

        if not video_model_ckpt == "":
            self.load_video_model_from_ckpt(video_model_ckpt)

        if train_head_only:
            for p in self.video_model.parameters():
                p.requires_grad = False
            for p in self.graph_model.parameters():
                p.requires_grad = False

    def load_graph_model_from_ckpt(self, ckpt_path : str):
        """Pytorch Lightning saves layer weights by prepending "model" to layer names.
        If we do not want to load checkpoints for Lightning Modules but for plain torch models, we have to shorten dict keys. 

        Args:
            ckpt_path (str): Path to checkpoint file.
        """        
        ckpt = torch.load(ckpt_path)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.removeprefix("model.")] = v
        self.graph_model.load_state_dict(state_dict, strict=False)

    def load_video_model_from_ckpt(self, ckpt_path : str):
        """Pytorch Lightning saves layer weights by prepending "model" to layer names.
        If we do not want to load checkpoints for Lightning Modules but for plain torch models, we have to shorten dict keys. 

        Args:
            ckpt_path (str): Path to checkpoint file.
        """        
        ckpt = torch.load(ckpt_path)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.removeprefix("model.")] = v
        # do not load strict since we dont use the head
        self.video_model.load_state_dict(state_dict, strict=False)


    def forward(self, batch):
        graph = batch["positions"]
        frames = batch["frames"]

        if isinstance(self.graph_model, (GIN, GAT)):
            g_repr = self.graph_model(graph, graph.ndata["positions"])
        elif isinstance(self.graph_model, PositionTransformer):
            g_repr = self.graph_model(graph)
        else:
            raise NotImplementedError("Unknown model type", type(self.graph_model))

        vid_repr = self.video_model(frames)
        mm_repr = torch.concatenate([g_repr, vid_repr], dim=1)
        return self.head(mm_repr)



if "__main__" == __name__:
    from data.datasets import MultiModalHblDataset, ResampledHblDataset
    from lit_data import collate_function_builder
    from torchvision import transforms as t
    import multimodal_transforms as mmt
    import pytorchvideo.transforms as ptvt

    basic_transforms = t.Compose([
            vt.FrameSequenceToTensor(),
            mmt.Resize(size=(224,224))
            ])

    collate_fn = collate_function_builder(
        epsilon=7,
        load_frames=True,
        mix_video=None,
        position_format="flattened",
        relative_positions=False
        )

    dataset = MultiModalHblDataset("/nfs/home/rhotertj/datasets/hbl/meta3d.csv", 16, sampling_rate=4, transforms=basic_transforms, overlap=False)

    instances = []
    for i in range(8):
        x = dataset[i*25]
        instances.append(x)

    batch = collate_fn(instances)

    mm_model = MultiModalModel(
        graph_model_name="PositionTransformer",
        graph_model_params={
            "dim_in": 49,
            "dim_h": 256,
            # "readout": "mean",
            "input_operation": "linear",
            "num_heads": 8,
            "batch_size": 8
        },
        video_model_name="make_kinetics_mvit",
        video_model_params={
            "pretrained_path": "models/mvit_b_16x4.pt",
            "batch_size": 8
        },
        num_classes=3
    )
    mm_model.load_graph_model_from_ckpt("/nfs/home/rhotertj/Code/thesis/experiments/gat/train/comic-waterfall-163/epoch=99-step=474600.ckpt")
    mm_model.load_video_model_from_ckpt("/nfs/home/rhotertj/Code/thesis/experiments/mvit/train/silver-blaze-237/epoch=29-step=113910.ckpt")
    print("Done")
    mm_model(batch)

    