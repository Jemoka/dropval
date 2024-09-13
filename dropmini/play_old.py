
#############

model = checkpoint.model
gen = checkpoint.generator

pattern1 = list(random.choice(random.choice(gen.sequences)))
pattern2 = list(random.choice(random.choice(gen.sequences)))
pattern3 = list(random.choice(random.choice(gen.sequences)))
prefix = random.sample(gen.iid_tokens, 3)
suffix = random.sample(gen.iid_tokens, 3)
b1 = random.sample(gen.iid_tokens, 1)
b2 = random.sample(gen.iid_tokens, 1)
# seq = [gen.enc_token] + prefix + pattern1 + b1 + pattern2 + b2 + pattern3 + suffix + [gen.enc_token]

# mask some indicies that are a part of the pairwise elements
# labels = deepcopy(seq)

seq[4] = gen.mask_token
seq[9] = gen.mask_token
seq[14] = gen.mask_token

pre_attn = []

def inspect_hook_attn(module, input, output):
    global pre_attn
    pre_attn.append(input[0])
    return output

pre_proj = []

def inspect_hook_proj(module, input, output):
    global pre_proj
    pre_proj.append(input[0])
    return output

remove_hooks1 = [i.self_attn.register_forward_hook(inspect_hook_attn)
                for i in model.encoder.layers]
remove_hooks2 = [i.linear2.register_forward_hook(inspect_hook_proj)
                 for i in model.encoder.layers]

# call our model to register pre-attn
inputs = torch.tensor(seq).unsqueeze(0).cuda()
labels = torch.tensor(labels).unsqueeze(0).cuda()
res = model(inputs, ~torch.ones_like(inputs).bool())

# x: batch, seq, hidden
(out, scores0) = model.encoder.layers[0].self_attn(pre_attn[0], pre_attn[0], pre_attn[0])
scores_map0 = sns.heatmap(scores0.squeeze(0).cpu().detach().numpy())

scores_map0.figure.clear()

(out, scores1) = model.encoder.layers[1].self_attn(pre_attn[1], pre_attn[1], pre_attn[1])
scores_map1 = sns.heatmap(scores1.squeeze(0).cpu().detach().numpy())

# permute and analyze specific slots
pproj = [i.permute(1,0,2).squeeze() for i in pre_proj]
encodings0 = pproj[0][[4,9,14]]
encodings1 = pproj[1][[4,9,14]]

encodings0[0]


# [i.remove() for i in remove_hooks1]
# [i.remove() for i in remove_hooks2]

