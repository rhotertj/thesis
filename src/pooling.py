import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD(nn.Module):
    """
    NetVLAD: CNN architecture for weakly supervised place recognition
    https://arxiv.org/abs/1511.07247

    Adapted from unofficial implementation
    https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    
    """

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def get_output_size(self):
        return self.dim * self.num_clusters

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        B, D = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(B, self.num_clusters, -1) # [b, c, 1]
        soft_assign = F.softmax(soft_assign, dim=1) # membership of cluster for each feature vector

        x_flatten = x.view(B, D, -1) # flattens feature vector, probably irrelevant for us 
        
        # calculate residuals to each clusters x_i(j) - c_k(j)
        x_expanded = x_flatten.expand(self.num_clusters, -1, -1, -1)
        x_expanded = x_expanded.permute(1, 0, 2, 3) # [b, c, d, 1]

        # expand centroids to x shape
        centroids = self.centroids.expand(x_flatten.size(-1), -1, -1)
        centroids = centroids.permute(1, 2, 0).unsqueeze(0)

        # [b, c, dim, 1] = [b, c, dim, 1] - [1, c, dim, 1]
        residual = x_expanded - centroids
        # filter residuals by cluster membership
        residual *= soft_assign.unsqueeze(2) # [b, c, dim, 1] = [b, c, dim, 1] * [b, c, 1, 1]
        vlad = residual.sum(dim=-1) # sum over last dim (which is 1?) -> [b,c,d]

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten [b, c*d]
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class MeanPool(nn.Module):

    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)
    
if __name__ == "__main__":
    torch.manual_seed(0)
    num_clusters = 2
    dim = 8
    batch_size = 3
    x = torch.rand(batch_size, dim, 1, 1)
    print("x:", x)
    model = NetVLAD(
        num_clusters=num_clusters,
        dim=dim
    )
    y = model(x)

    print(y.shape) # [n, n_cluster * dim]

