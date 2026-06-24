import torch


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch, max_len=None):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    if max_len is not None:
        maskbatchTensor = lengths_to_mask(lenbatchTensor, max_len).unsqueeze(1).unsqueeze(1)  # unqueeze for broadcasting
    else:
        maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    if "audio" in notnone_batches[0]:
        audiobatch = [b["audio"] for b in notnone_batches]
        cond["y"].update({"audio": audiobatch})

    if "tokens" in notnone_batches[0]:
        textbatch = [b["tokens"] for b in notnone_batches]
        cond["y"].update({"tokens": textbatch})

    if "action" in notnone_batches[0]:
        actionbatch = [b["action"] for b in notnone_batches]
        cond["y"].update({"action": torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if "action_text" in notnone_batches[0]:
        action_text = [b["action_text"] for b in notnone_batches]
        cond["y"].update({"action_text": action_text})

    # For EMAGE
    if "tar_trans" in notnone_batches[0]:
        tar_trans = [b["tar_trans"] for b in notnone_batches]
        cond["y"].update({"tar_trans": tar_trans})

    if "tar_exps" in notnone_batches[0]:
        tar_exps = [b["tar_exps"] for b in notnone_batches]
        cond["y"].update({"tar_exps": tar_exps})

    if "tar_beta" in notnone_batches[0]:
        tar_beta = [b["tar_beta"] for b in notnone_batches]
        cond["y"].update({"tar_beta": tar_beta})

    if "tar_pose" in notnone_batches[0]:
        tar_pose = [b["tar_pose"] for b in notnone_batches]
        cond["y"].update({"tar_pose": tar_pose})

    if "in_audio_resample" in notnone_batches[0]:
        in_audio_resample = [b["in_audio_resample"] for b in notnone_batches]
        cond["y"].update({"in_audio_resample": in_audio_resample})

    if "in_text_semantic" in notnone_batches[0]:
        in_text_semantic = [b["in_text_semantic"] for b in notnone_batches]
        cond["y"].update({"in_text_semantic": in_text_semantic})

    if "tar_id" in notnone_batches[0]:
        tar_id = [b["tar_id"] for b in notnone_batches]
        cond["y"].update({"tar_id": tar_id})

    if "tar_name" in notnone_batches[0]:
        tar_name = [b["tar_name"] for b in notnone_batches]
        cond["y"].update({"tar_name": tar_name})

    return motion, cond


# an adapter to our collate func
# this is one element in batch: word_embeddings, pos_one_hots, caption, sent_len, motion, '_'.join(tokens), '_'.join(tokens)
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            "inp": torch.tensor(b[4].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "text": b[2],  # caption
            "tokens": b[6],  # '_'.join(tokens)
            "lengths": b[5],  # '_'.join(tokens)
        }
        for b in batch
    ]
    return collate(adapted_batch)


def beat2_collate(batch, max_len=None, length=None):
    adapted_batch = [
        {
            "inp": torch.tensor(b["latent_all"].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "text": b["text"],  # b[0]['caption']
            "tokens": b["in_word"],
            "lengths": b["latent_all"].shape[0] if length is None else length,  # all value are same length
            "tar_trans": b["tar_trans"],
            "tar_exps": b["tar_exps"],
            "tar_beta": b["tar_beta"],
            "tar_pose": b["tar_pose"],
            "audio": b["in_audio"],
            "tar_id": b["tar_id"],
            "tar_name": b["tar_name"],
            "in_audio_resample": b["in_audio_resample"],
            "in_text_semantic": b["in_text_semantic"],
        }
        for b in batch
    ]
    return collate(adapted_batch, max_len=max_len)
