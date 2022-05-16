An implementation to test the hypothesis that clipping a step is more efficient than clipping the gradient norm.

## Optimizers
-------------------------------------
| optimizer | note |
|--------|------|
| AdaBelief | Provide the epsilon value as the square root of epsilon presented in the [paper](https://arxiv.org/pdf/2010.07468.pdf). <p> TL;DR Use  typical epsilon of Adam.
|TBD|...|

## usage
----------
```python
from clip_opt import AdaBelief

optimizer = AdaBelief(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-6,
    amsgrad=False,
    weight_decouple=True,
    fixed_decay=False,
    rectify=False,
    clip_step=1,
    norm_ord=2,
)

```