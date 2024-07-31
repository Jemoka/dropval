"""
ig.py
integrated gradients parameter guard (to modify intermidate activations)
"""


import re
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

class ParamGuard:
    """
    contextmanager to hold a single layer in memory and help interpolate its values

    modes: "probe", "suppress", "double"

    >>> with ParamGuard(next(lm.ffn_layers())) as pg:
    >>>    lm.call_strided("The capital of [MASK] is Kunming.", 1)
    >>>    pg.interpolate([4,0])
    >>>    print(lm.call_strided("The capital of [MASK] is Kunming."))
    """

    def __init__(self, module:nn.Module, component:str, mode="probe", steps=20, speed=False):
        self.module = module
        self.component = component # inner attr to modify
        self.mode = mode
        self.__hook_handle = None
        self.__interpolate_indicies = None # which value to stride from 0 to that values
        self.__interpolation_values = None
        self.__mask_idx = None
        self.__baseline = None 
        self.__interpolate_steps = steps
        self.__scale_multiplier = torch.linspace(0, 1, steps=self.__interpolate_steps).to(next(module.parameters()).device)
        self.__speed = speed
        self.__component_cache = None

    @property
    def baseline(self):
        return self.__baseline[0]

    @property
    def interpolations(self):
        return self.__interpolation_values

    def interpolate(self, x):
        self.__interpolate_indicies = x
        # to be computed again
        self.__interpolation_values = None

    def set(self, x, mask_idx):
        # to be computed again
        self.__interpolation_values = x
        self.__interpolation_values.requires_grad_(True)
        self.__mask_idx = mask_idx

    def __compute(self, x):
        if self.mode == "probe":
            iv = self.__interpolation_values
            mi = self.__mask_idx
            x[:, mi] = iv
            return x

        # number of steps is determined by bs
        if not isinstance(self.__interpolate_indicies, torch.Tensor):
            indicies = torch.stack(self.__interpolate_indicies)
        else:
            indicies = self.__interpolate_indicies

        if self.mode == "suppress":
            x[:, indicies[:,0], indicies[:,1]] = 0

        elif self.mode == "double":
            x[:, indicies[:,0], indicies[:,1]] *= 2

        elif isinstance(self.mode, numbers.Number):
            x[:, indicies[:,0], indicies[:,1]] *= self.mode

        return x

    def reset(self):
        self.__interpolate_indicies = None
        self.__interpolation_values = None
        self.__mask_idx = None

    def __enter__(self):
        self.__component_cache = getattr(self.module, self.component)
        
        # this is to make sure the inner class can access
        # a pointer to self 
        context = self

        class Patch(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                intermediate = self.module(x)
                return context.hook_(intermediate)
                
        setattr(self.module, self.component, Patch(self.__component_cache))
        return self

    def hook_(self, output):
        if self.__interpolate_indicies != None or self.__interpolation_values != None:
            return self.__compute(output)
        else: 
            self.__baseline = output
            return output

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.module, self.component, self.__component_cache)

def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    
    from https://github.com/kmeng01/rome/blob/main/util/nethook.py
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)

class LanguageModel:
    """
    An object to hold on to (or automatically download and hold)
    a Bert-style language model and tokenizer. 

    Adapted from ROME code
    """

    def __init__(self, model_name=None, model=None,
                 tokenizer=None, low_cpu_mem_usage=False,
                 torch_dtype=None, device=torch.device("cpu")):

        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForMaskedLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )

        self.tokenizer = tokenizer
        self.model = model.eval()
        set_requires_grad(False, self.model)
        self.layer_names = [
            n for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def ffn_layers(self):
        for i in self.model.bert.encoder.layer:
            yield (i.intermediate, "dense")
            
    @property
    def device(self):
        return next(self.model.parameters()).device

    def tokenize(self, prompts):
        tok = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512).to(self.device)
        mask_locs = (tok["input_ids"] == self.tokenizer.mask_token_id).nonzero()[:,1]

        return tok, mask_locs

    def __call__(self, x, strides=20, return_all_info=True, force_pred_idx=False):
        """perform evaluation through interpolation across `strides` times for `x` prompt

        Parameters
        ----------
        x : str
            the prompt to evaluate
        strides : int
            number of strides
        """
        
        tok = self.tokenize([x for _ in range(strides)])
        return self.__fill_batch(tok, return_all_info, force_pred_idx)

    def __fill_batch(self, prompts, return_all_info=True, force_pred_idx=False):
        """perform mask filling task for an entire batch

        Parameters
        ----------
        prompts : List[str]
            the prompt, with [MASK] within

        Returns
        -------
        List[str], torch.Tensor[Batch,], torch.Tensor[Batch,Vocab]

        mask fill, probability of `probe` or argmax, the output distribution


        """

        tok, mask_locs = prompts
        
        res = self.model(**tok)

        # assume here all uses one mask location
        masked_dists = res.logits[:,mask_locs[0]]
        prob_dist = masked_dists.softmax(dim=1)
        if not force_pred_idx:
            masked_token_idx = masked_dists[-1, :].argmax(dim=0)
        else:
            masked_token_idx = torch.tensor(force_pred_idx).to(masked_dists.device)
        probs = prob_dist[:, masked_token_idx]

        if not return_all_info:
            return probs

        # we do this instead of .argmax(dim=1) because we assume
        # the whole batch has one prompt, and we seek to know the
        # value for the unchanged prompt (i.e. the last one)
        pred_token_idx = masked_dists.argmax(dim=1)
        masked_tokens = self.tokenizer.convert_ids_to_tokens(pred_token_idx)

        return masked_tokens, probs, masked_dists, mask_locs

    def to(self, device):
        self.model = self.model.to(device)

        return self

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__}, "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

