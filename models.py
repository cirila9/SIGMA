# classes
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import math
        
def SinCosEmbed(seq_len, d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False
    
    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    return pe[:, :seq_len, :d_model]

class PosEmb(nn.Module):
    def __init__(self, num, dim, emb_dropout):
        super().__init__()
        self.num = num
        self.dropout = nn.Dropout(emb_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emd = nn.Parameter(SinCosEmbed(num+1, dim), requires_grad=False)
    
    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        return x + self.dropout(self.pos_emd)
        
class Logits(nn.Module):
    def __init__(self, cls_token=False):
        super().__init__()
        self.cls_token = cls_token
    def forward(self, x):
        out = x[:, -1] if self.cls_token else x.mean(dim = 1)
        return out

class MAE(nn.Module):
    def __init__(
                self,*,
                seq_len, 
                channels,
                patch_size,   
                dim = 192, 
                heads = 12,
                mlp_ratio = 4,
                cls_token = True,
                emded_grad = True,
                masking_ratio = 0.7,
                # encodoer paraments
                encoder_depth = 2, 
                dropout = .2, 
                emb_dropout = 0.,
                # decoder paraments
                decoder_depth = 2, 
                ):
        super().__init__()
        assert (seq_len % patch_size) == 0, 'seq_len must be divisible by patch_size'
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'

        self.dim = dim
        self.cls = cls_token
        self.logits = Logits(cls_token)
        self.masking_ratio = masking_ratio
        num_patches = seq_len // patch_size * channels
        self.to_patch = Rearrange('b c (n p) -> b (c n) p', p = patch_size)
        
        self.patch_to_emb = nn.Sequential(
                                            # nn.LayerNorm(patch_size),
                                            nn.Linear(patch_size, dim, bias=False),
                                            nn.LayerNorm(dim),
                                            PosEmb(num_patches, dim, emb_dropout),
                                            )
        # xavier_uniform initialization
        nn.init.xavier_uniform_(self.patch_to_emb[0].weight)
        self.patch_to_emb[0].weight.requires_grad = emded_grad
                
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        # extract some hyperparameters and functions from encoder
        EncoderLayer = nn.TransformerEncoderLayer(d_model=dim, 
                                                    nhead=heads,
                                                    dim_feedforward=int(dim*mlp_ratio),
                                                    dropout=dropout, 
                                                    activation='gelu',
                                                    # layer_norm_eps=1e-3,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    )
        self.encoder = nn.TransformerEncoder(EncoderLayer, num_layers=encoder_depth)
        
        # decoder parameters
        self.mask_token = nn.Parameter(torch.randn(dim))
        DecoderLayer = nn.TransformerEncoderLayer(d_model=dim, 
                                                    nhead=heads,
                                                    dim_feedforward=int(dim*mlp_ratio),
                                                    dropout=dropout, 
                                                    activation='gelu',
                                                    # layer_norm_eps=1e-3,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    )
        self.decoder = nn.TransformerEncoder(DecoderLayer, num_layers=decoder_depth)
        self.decoder_pos_emb = nn.Parameter(SinCosEmbed(num_patches+1, dim), requires_grad=False) if self.cls else nn.Parameter(SinCosEmbed(num_patches, dim), requires_grad=False)
        
        self.to_seqs = nn.Linear(dim, patch_size)
        
        # MSE and Cosine Similarity Loss 
        self.loss = nn.MSELoss()
        self.criterion = nn.CosineSimilarity(dim=1)

    # Modify cross entropy (CE) as: -softmax(z)*log(softmax(p))
    def CE(self, p, z):
        return - (z.softmax(dim=1) * p.log_softmax(dim=1)).mean()

    def ContrastiveLoss(self, p, z, T=.07):
        # normalize
        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [p, z]) / T
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def forward(self, series):
        device = series.device

        # get patches
        patches = self.to_patch(series)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)

        batch_range = torch.arange(batch, device=device)[:, None]
        if self.cls:
            rand_indices = torch.rand(batch, num_patches+1, device=device)
            rand_indices[:, -1] = 1e+7
            rand_indices = rand_indices.argsort(dim = -1)
        else:
            rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # target feature
        z = tokens[batch_range, masked_indices]
        z = self.encoder(z).mean(dim=1)
        
        # get the unmasked tokens to be encoded
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        
        # attend with transformer
        encoded_tokens = self.encoder(tokens)

        # project encoder to decoder dimensions, if they are not equal
        # encoded_tokens += self.decoder_pos_emb[:, unmasked_indices]
        encoded_tokens = self.ln(encoded_tokens) + self.decoder_pos_emb[:, unmasked_indices]
            
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb[:, masked_indices]
        
        
        # concat the masked tokens to the decoder tokens
        if self.cls:
            decoder_tokens = torch.zeros(batch, num_patches+1, self.dim, device=device)
        else:
            decoder_tokens = torch.zeros(batch, num_patches, self.dim, device=device)
        
        decoder_tokens[batch_range, unmasked_indices] = encoded_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens

        # attend with decoder
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the pred_features and pred_values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_values = self.to_seqs(mask_tokens)
        p = mask_tokens.mean(dim=1)

        # calculate reconstruction loss
        recon_loss = self.loss(pred_values, masked_patches)
        # criterion = - (z.detach().softmax(dim=1) * p.log_softmax(dim=1)).mean()
        criterion = -self.criterion(p, z.detach()).mean()
        # criterion = self.ContrastiveLoss(p, z.detach())
        
        return criterion + recon_loss

# define function to training
def Training(model, num_epochs, opt=opt, data_dl=loader):
    loss_history = []
    start_time = time.time()
    path2weights = './models/SIGMA_hESC1000+Non-Specific.pt'
    best_loss = 1e+7
    
    # 模型输出和loss计算
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        model.train()
        running_loss = 0
        for i, (x, _) in enumerate(tqdm(data_dl)):
            # retrieve query and key`````
            x = x.float().to(device, non_blocking=True)
            # compute output and loss
            with amp.autocast():
                loss = model(x)
                if loss == torch.nan:
                    break
                opt.zero_grad()
            # compute gradient and do SGD step
                scaler.scale(loss).backward()
                scaler.step(opt)
                scale = scaler.get_scale()
                scaler.update()
#             loss.backward()
#             opt.step()
            running_loss += loss
            
        if loss == torch.nan:
            break
        # store loss history
        epoch_loss = running_loss / (i+1)
        loss_history.append(epoch_loss.detach().cpu().numpy())
        print('train loss: %.6f, time: %.2f min' %(epoch_loss,(time.time()-start_time)/60))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)

    # save weights
    torch.save(model.state_dict(), path2weights)
    best_encoder = nn.Sequential(copy.deepcopy(best_model.to_patch),
                            copy.deepcopy(best_model.patch_to_emb),
                            copy.deepcopy(best_model.encoder),
                            # copy.deepcopy(best_model.ln),
                            copy.deepcopy(best_model.logits))
    # projector = copy.deepcopy(encoder.spatial.fc)# .fc
    torch.cuda.empty_cache()  # 释放显存
    return best_encoder, loss_history