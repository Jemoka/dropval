"""
linear.py
linear parameter guard (to modify intermidate activations)
"""

import re
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

class LinearParamGuard:
    """
    contextmanager to hold a single layer's activations in memory and
    help interpolate its values

    >>> with ParamGuard(model.layer, "dense") as pg:
    >>>    model(**tokenizer("have you ever wanted to intervene a model?"))
    >>>    pre,post = pg.activations
    >>>    pg.intervene(lambda pre,post: return post)
    """

    def __init__(self, module:nn.Module, component:str, tape:bool=False, replace=None):
        self.module = module
        self.component = component # inner attr to modify
        self.__cache = (None, None)
        self.__intervention = None
        self.__gradient_tape = tape
        self.__replace = replace

    @property
    def activations(self):
        return self.__cache

    def grad(self, loss):
        return torch.autograd.grad(loss, self.__cache[1])[0]

    def intervene(self, func):
        self.__intervention = func
    def reset(self):
        self.__intervention = None

    def __enter__(self):
        self.__component_cache = getattr(self.module, self.component)
        
        # this is to make sure the inner class can access
        # a pointer to self 
        context = self

        class LinearPatch(nn.Module):
            def __init__(self, module):
                super().__init__()

                self.module = module
                self.weight = module.weight
                self.bias = module.bias

            def forward(self, x):
                intermediate = torch.einsum("...i,oi -> ...o", x, self.weight)
                result = context.hook_(x, intermediate, self.weight)

                return result + self.bias
                
        if self.__replace:
            setattr(self.module, self.component, self.__replace)
        else:
            setattr(self.module, self.component, LinearPatch(self.__component_cache))
        return self

    def hook_(self, input, output, weights):
        self.__cache = (input, output)

        if self.__intervention != None:
            result = self.__intervention(input, output, weights)
        else:
            result = output

        if self.__gradient_tape:
            result.requires_grad_()
        return result

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


class MultiLinearParamGuard:
    """
    contextmanager to hold a single layer's activations in memory and
    help interpolate its values

    >>> with ParamGuard(model.layer, "dense") as pg:
    >>>    model(**tokenizer("have you ever wanted to intervene a model?"))
    >>>    pre,post = pg.activations
    >>>    pg.intervene(lambda pre,post: return post)
    """

    def __init__(self, modules:list[nn.Module], components:list[str]=None, replace:list[nn.Module]=None):
        self.modules = modules
        if components:
            self.components = components # inner attr to modify
        else:
            self.components = ["dense" for _ in range(len(modules))] # inner attr to modify

        if replace:
            self.__guards = [LinearParamGuard(i,j,replace=k) for i,j,k in zip(self.modules, self.components, replace)]
        else:
            self.__guards = [LinearParamGuard(i,j) for i,j in zip(self.modules, self.components)]

    def __enter__(self):
        for i in self.__guards:
            i.__enter__()
        return self

    def intervene(self, funcs):
        for i, iv in zip(self.__guards, funcs):
            i.intervene(iv)
    def reset(self):
        for i in self.__guards:
            i.reset()

    def __exit__(self, exc_type, exc_value, traceback):
        for i in self.__guards:
            i.__exit__(exc_type, exc_value, traceback)
