import torch
from encoderDecoder import TransformerEncoder, TransformerDecoder
import torch.nn as nn
from typing import List, Tuple

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x

class DenoMAE(nn.Module):
    def __init__(self, num_modalities: int, img_size: int = 224, patch_size: int = 16, 
                 in_chans: int = 3, mask_ratio: float = 0.75, embed_dim: int = 768, encoder_depth: int = 12, 
                 decoder_depth: int = 4, num_heads: int = 12):
        """
        Initialize the DenoMAE model.

        Args:
            num_modalities (int): Number of input modalities.
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of the patches to be extracted from the image.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the token embeddings.
            encoder_depth (int): Number of transformer encoder layers.
            decoder_depth (int): Number of transformer decoder layers.
            num_heads (int): Number of attention heads in transformer layers.
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embeds = nn.ModuleList([
            PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
            for _ in range(num_modalities)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        self.encoder = TransformerEncoder(embed_dim, encoder_depth, num_heads)
        
        self.decoders = nn.ModuleList([
            TransformerDecoder(embed_dim, decoder_depth, num_heads)
            for _ in range(num_modalities)
        ])
        
        self.modality_heads = nn.ModuleList([
            nn.Linear(embed_dim, patch_size**2 * in_chans)
            for _ in range(num_modalities)
        ])

    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).
            mask_ratio (float): Ratio of patches to be masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Masked input tensor
                - Binary mask (1 for masked, 0 for unmasked)
                - Indices for restoring the original order
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of the DynamicMultiMAE model.

        Args:
            inputs (List[torch.Tensor]): List of input tensors, one for each modality. 
                                         Each tensor should have shape (batch_size, in_chans, img_size, img_size).
            mask_ratio (float): Ratio of patches to be masked during training.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - List of reconstructed outputs, one for each modality. 
                  Each tensor has shape (batch_size, in_chans, img_size, img_size).
                - List of binary masks, one for each modality. 
                  Each tensor has shape (batch_size, num_patches).
        """
        assert len(inputs) == self.num_modalities, f"Expected {self.num_modalities} inputs, but got {len(inputs)}"

        masked_embeddings = []
        masks = []
        ids_restores = []

        for i, x in enumerate(inputs):
            x = self.patch_embeds[i](x)
            x = x + self.pos_embed[:, 1:, :]
            x_masked, mask, ids_restore = self.random_masking(x)
            masked_embeddings.append(x_masked)
            masks.append(mask)
            ids_restores.append(ids_restore)

        x_masked = torch.cat(masked_embeddings, dim=1)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)
        
        encoded = self.encoder(x_masked)
        
        decoded_outputs = []
        segment_length = (encoded.shape[1] - 1) // self.num_modalities
        for i in range(self.num_modalities):
            start_idx = 1 + i * segment_length
            end_idx = 1 + (i + 1) * segment_length
            decoded = self.decoders[i](encoded[:, start_idx:end_idx])
            decoded_outputs.append(decoded)

        reconstructions = []
        for i, decoded in enumerate(decoded_outputs):
            rec = self.modality_heads[i](decoded)
            rec = rec.reshape(shape=(rec.shape[0], -1, self.patch_size, self.patch_size, self.in_chans))
            
            full_rec = torch.zeros((rec.shape[0], self.n_patches, self.patch_size, self.patch_size, self.in_chans), device=rec.device)
            
            n_keep = int(self.n_patches * (1 - self.mask_ratio))
            full_rec[:, :n_keep] = rec
            
            full_rec = torch.gather(full_rec, dim=1, index=ids_restores[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.patch_size, self.patch_size, self.in_chans))
            
            full_rec = full_rec.permute(0, 1, 4, 2, 3).reshape(shape=(full_rec.shape[0], self.in_chans, self.img_size, self.img_size))
            reconstructions.append(full_rec)

        return reconstructions, masks
