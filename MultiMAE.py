import torch
from utils import PatchEmbedding
from encoderDecoder import TransformerEncoder, TransformerDecoder
import torch.nn as nn

class MultiMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, encoder_depth=12, decoder_depth=4, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed1 = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed2 = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed3 = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        self.encoder = TransformerEncoder(embed_dim, encoder_depth, num_heads)
        self.decoder1 = TransformerDecoder(embed_dim, decoder_depth, num_heads)
        self.decoder2 = TransformerDecoder(embed_dim, decoder_depth, num_heads)
        self.decoder3 = TransformerDecoder(embed_dim, decoder_depth, num_heads)
        
        self.modality1_head = nn.Linear(embed_dim, patch_size**2 * in_chans)
        self.modality2_head = nn.Linear(embed_dim, patch_size**2 * in_chans)
        self.modality3_head = nn.Linear(embed_dim, patch_size**2 * in_chans)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x1, x2, x3, mask_ratio=0.75):

        # Patch embedding
        x1 = self.patch_embed1(x1)
        x2 = self.patch_embed2(x2)
        x3 = self.patch_embed3(x3)
        
        # Add positional embedding
        x1 = x1 + self.pos_embed[:, 1:, :]
        x2 = x2 + self.pos_embed[:, 1:, :]
        x3 = x3 + self.pos_embed[:, 1:, :]
        
        # Masking
        x1_masked, mask1, ids_restore1 = self.random_masking(x1, mask_ratio)
        x2_masked, mask2, ids_restore2 = self.random_masking(x2, mask_ratio)
        x3_masked, mask3, ids_restore3 = self.random_masking(x3, mask_ratio)
        
        # Concatenate masked embeddings
        x_masked = torch.cat([x1_masked, x2_masked, x3_masked], dim=1)
        
        # Add CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # TODO: check if this is correct
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)
        
        # Encode
        encoded = self.encoder(x_masked)
        
        # Decode separately for each modality
        decoded1 = self.decoder1(encoded[:, 1:encoded.shape[1]//3+1])  # exclude CLS token and take first third
        decoded2 = self.decoder2(encoded[:, encoded.shape[1]//3+1:2*encoded.shape[1]//3+1])   # take second third
        decoded3 = self.decoder3(encoded[:, 2*encoded.shape[1]//3+1:])   # take last third 

        # Reconstruct patches
        rec1 = self.modality1_head(decoded1)
        rec2 = self.modality2_head(decoded2)
        rec3 = self.modality3_head(decoded3)
        
        # Reshape to image patches
        rec1 = rec1.reshape(shape=(rec1.shape[0], -1, self.patch_size, self.patch_size, self.in_chans))
        rec2 = rec2.reshape(shape=(rec2.shape[0], -1, self.patch_size, self.patch_size, self.in_chans))
        rec3 = rec3.reshape(shape=(rec3.shape[0], -1, self.patch_size, self.patch_size, self.in_chans))
        
        # Prepare full-sized tensors with zeros
        full_rec1 = torch.zeros((rec1.shape[0], self.n_patches, self.patch_size, self.patch_size, self.in_chans), device=rec1.device)
        full_rec2 = torch.zeros((rec2.shape[0], self.n_patches, self.patch_size, self.patch_size, self.in_chans), device=rec2.device)
        full_rec3 = torch.zeros((rec3.shape[0], self.n_patches, self.patch_size, self.patch_size, self.in_chans), device=rec3.device)
        
        # Fill in the reconstructed patches
        n_keep = int(self.n_patches * (1 - mask_ratio))
        full_rec1[:, :n_keep] = rec1
        full_rec2[:, :n_keep] = rec2
        full_rec3[:, :n_keep] = rec3
        
        # Unshuffle patches
        full_rec1 = torch.gather(full_rec1, dim=1, index=ids_restore1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.patch_size, self.patch_size, self.in_chans))
        full_rec2 = torch.gather(full_rec2, dim=1, index=ids_restore2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.patch_size, self.patch_size, self.in_chans))
        full_rec3 = torch.gather(full_rec3, dim=1, index=ids_restore3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.patch_size, self.patch_size, self.in_chans))
        
        # Reshape to images
        full_rec1 = full_rec1.permute(0, 1, 4, 2, 3).reshape(shape=(full_rec1.shape[0], self.in_chans, self.img_size, self.img_size))
        full_rec2 = full_rec2.permute(0, 1, 4, 2, 3).reshape(shape=(full_rec2.shape[0], self.in_chans, self.img_size, self.img_size))
        full_rec3 = full_rec3.permute(0, 1, 4, 2, 3).reshape(shape=(full_rec3.shape[0], self.in_chans, self.img_size, self.img_size))
        
        return full_rec1, full_rec2, full_rec3, mask1, mask2, mask3
