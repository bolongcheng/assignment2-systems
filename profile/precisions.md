float16

```
before autocast
Module: Linear(in_features=512, out_features=10, bias=False), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Bias dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Weight dtype: torch.float32
dtype before fc1:  torch.float32
dtype after fc1:  torch.float16
dtype after relu:  torch.float16
dtype after ln:  torch.float32
dtype after fc2:  torch.float16
after forward pass
Module: Linear(in_features=512, out_features=10, bias=False), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Bias dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Weight dtype: torch.float32
after backward pass
Module: Linear(in_features=512, out_features=10, bias=False), Weight dtype: torch.float32
Module: Linear(in_features=512, out_features=10, bias=False), Grad dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Grad dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Bias dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Grad dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Weight dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Grad dtype: torch.float32
torch.float32
```


bfloat16

```
before autocast
Module: Linear(in_features=512, out_features=10, bias=False), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Bias dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Weight dtype: torch.float32
dtype before fc1:  torch.float32
dtype after fc1:  torch.bfloat16
dtype after relu:  torch.bfloat16
dtype after ln:  torch.float32
dtype after fc2:  torch.bfloat16
after forward pass
Module: Linear(in_features=512, out_features=10, bias=False), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Bias dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Weight dtype: torch.float32
after backward pass
Module: Linear(in_features=512, out_features=10, bias=False), Weight dtype: torch.float32
Module: Linear(in_features=512, out_features=10, bias=False), Grad dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Weight dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Grad dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Bias dtype: torch.float32
Module: LayerNorm((10,), eps=1e-05, elementwise_affine=True), Grad dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Weight dtype: torch.float32
Module: Linear(in_features=10, out_features=36, bias=False), Grad dtype: torch.float32
torch.float32
```