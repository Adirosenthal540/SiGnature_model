import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from data_loaders.beat2.utils.media import convert_img_to_mp4
from model.mdm_transformer import MDM_TransformerEncoder, MDM_TransformerEncoderLayer
from model.rotation2xyz import Rotation2xyz
from data_loaders.beat2.models.utils.layer import BasicBlock
import pickle
import os
from data_loaders.beat2.utils.build_vocab import Vocab
from einops import rearrange
from .timm_transformer.transformer import Block as mytimmBlock
from utils.misc import recursive_op2


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b (...) r d", r=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1, audio_f_out=196):
        super().__init__()
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential(
            BasicBlock(audio_in, out_dim // 4, 15, 5, first_dilation=1600, downsample=True),
            # BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1700, downsample=True),
            BasicBlock(out_dim // 4, out_dim // 4, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(
                out_dim // 4,
                out_dim // 4,
                15,
                1,
                first_dilation=7,
            ),
            BasicBlock(out_dim // 4, out_dim // 2, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(out_dim // 2, out_dim // 2, 15, 1, first_dilation=7),
            BasicBlock(out_dim // 2, out_dim, 15, 3, first_dilation=0, downsample=True),
        )
        self.fix_out_dim = nn.Linear(194, audio_f_out)

    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1)
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.fix_out_dim(self.feat_extractor(wav_data))
        return out.transpose(1, 2)


class MDM(nn.Module):

    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=512,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="amass",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        device=None,
        audio_rep="onset+amplitude",
        do_not_use_clip=False,
        **kargs,
    ):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset
        self.device = device

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get("action_emb", None)
        self.nframes = kargs.get("nframes", 196)

        self.input_feats = self.njoints * self.nfeats
        self.cond_mask_prob_audio = kargs.get("cond_mask_prob_audio", 0)
        self.normalize_output = kargs.get("normalize_encoder_output", False)
        audio_f = kargs.get("audio_f", 256)

        if audio_rep == "onset+amplitude":
            self.WavEncoder = WavEncoder(audio_f, audio_in=2, audio_f_out=self.nframes)
        self.audio_feat_dim = audio_f

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
        self.input_process2 = InputProcess(self.data_rep, self.input_feats + 2 * self.audio_feat_dim, self.latent_dim)

        self.attention_map = {}  # in use for visualization
        self.attention_lookup = "layer{:02d}_step{:03d}"
        self.get_dict = {}  # for PnP features savings

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        with open(os.path.join(kargs.get("data_path", ""), "weights/vocab.pkl"), "rb") as f:
            self.lang_model = pickle.load(f)
        pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=False)
        self.text_encoder_body = nn.Linear(300, audio_f)

        seqTransEncoderLayer = MDM_TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=activation
        )

        self.seqTransEncoder = MDM_TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("EMBED TEXT")
                print("Loading CLIP...")
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, 2 * self.latent_dim, self.njoints, self.nfeats)

        self.rot2xyz = Rotation2xyz(device=device)
        self.do_not_use_clip = do_not_use_clip

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)
        self.input_process2 = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.num_heads_mytimmblocks = 8
        self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_heads_mytimmblocks)
        self.mytimmblocks = nn.ModuleList(
            [
                mytimmBlock(
                    dim=self.latent_dim, num_heads=self.num_heads_mytimmblocks, mlp_ratio=self.ff_size // self.latent_dim, drop_path=self.dropout
                )  # hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(self.num_layers)
            ]
        )
        self.film = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 2 * self.latent_dim))  # stable conditioning

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not "clip_model." in name]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device="cpu", jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False, shape_feat=(1, 512), device=None):
        if force_mask:
            if cond is None:
                return torch.zeros(shape_feat).to(device)
            else:
                return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            bs = cond.shape[0]
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return (cond * (1.0 - mask)).to(cond)
        else:
            return cond

    def mask_cond_audio(self, cond, force_mask=False, shape_audio_feat=(1, 196, 256), device=None):
        if force_mask:
            if cond is None:
                return torch.zeros(shape_audio_feat).to(device)
            else:
                return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob_audio > 0.0:
            bs = cond.shape[0]
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob_audio).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ["humanml", "kit"] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1).to(device)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def encode_poses(self, x, emb=None, timesteps=None):
        if emb is None:
            emb = self.embed_timestep(timesteps)  # [1, bs, d]
        bs, njoints, nfeats, nframes = x.shape

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]

        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

        output_movment_laten_space = self.seqTransEncoder(xseq)  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        return output_movment_laten_space[1:]

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get("uncond", False)
        if not self.do_not_use_clip:
            text = y["text"]
            if text is not None:
                enc_text = self.encode_text(text)
            else:
                enc_text = None
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask, device=x.device))

        audio_feat = y["audio"]
        if audio_feat is not None:
            audio_feat = torch.cat([a.reshape(1, -1, 2) for a in audio_feat], dim=0).to(x.device)
            audio_feat = self.WavEncoder(audio_feat)
        audio_feat = self.mask_cond_audio(
            audio_feat, force_mask=force_mask, shape_audio_feat=(x.shape[0], x.shape[-1], 256), device=x.device
        )  # .permute(1, 0, 2)

        text_feat = y["tokens"]
        if text_feat is not None:
            text_feat = torch.cat([a.reshape(1, -1).to(x.device) for a in text_feat], dim=0)
            text_feat = self.text_pre_encoder_body(text_feat)
            text_feat = self.text_encoder_body(text_feat)
        text_feat = self.mask_cond_audio(
            text_feat, force_mask=force_mask, shape_audio_feat=(x.shape[0], x.shape[-1], 256), device=x.device
        )  # .permute(1, 0, 2)

        at_feat = torch.cat([audio_feat, text_feat], dim=2)
        at_feat = self.mask_cond_audio(at_feat, force_mask=force_mask, shape_audio_feat=(x.shape[0], x.shape[-1], 512), device=x.device).permute(
            1, 0, 2
        )

        output_movment_laten_space = self.encode_poses(x, emb, timesteps)

        if force_mask:
            at_feat = torch.zeros_like(at_feat)
        elif not self.do_not_use_clip:
            film_params = self.film(at_feat)  # [T, B, 2d]
            gamma, beta = film_params.chunk(2, dim=-1)
            output_movment_laten_space = output_movment_laten_space * (1.0 + gamma) + beta

        # add cond
        xseq = torch.cat((output_movment_laten_space, at_feat), axis=2)
        xseq = self.input_process2(xseq)
        xseq = xseq.permute(1, 0, 2)
        xseq = xseq.view(bs, nframes, self.num_heads_mytimmblocks, -1)
        xseq = xseq.permute(0, 2, 1, 3)
        xseq = xseq.reshape(bs * self.num_heads_mytimmblocks, nframes, -1)
        pos_emb = self.rel_pos(xseq)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
        xseq_rpe = xseq.reshape(bs, self.num_heads_mytimmblocks, nframes, -1)
        xseq = xseq_rpe.permute(0, 2, 1, 3)
        xseq = xseq.view(bs, nframes, -1)

        # Initialize tracking for current timestep
        for i, block in enumerate(self.mytimmblocks):
            layer_str = f"layer{i:02d}"
            # Pass correlation data if available
            xseq = block(xseq)

        xseq = xseq.permute(1, 0, 2)

        output = self.output_process(xseq)  # [bs, njoints, nfeats, nframes]

        return output

    def _apply(self, fn):
        super()._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == "rot_vel":
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == "rot_vel":
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == "rot_vel":
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            output = self.poseFinal(output)
        elif self.data_rep == "rot_vel":
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
