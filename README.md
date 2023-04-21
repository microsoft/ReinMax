![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reinmax) 
![GitHub](https://img.shields.io/github/license/microsoft/reinmax) 
[![Maintenance](https://img.shields.io/badge/doc-yes-success.svg)](https://microsoft.github.io/reinmax/) 
![PyPI](https://img.shields.io/pypi/v/reinmax) 

<h2 align="center">ReinMax</h2>
<h4 align="center"> Beyond Straight-Through</h4>

<p align="center">
  <a href="#st">Straight-Through</a> •
  <a href="#reinmax">ReinMax</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a> •
  <a href="https://github.com/microsoft/reinmax/tree/main/LICENSE">License</a>
</p>

[ReinMax](https://arxiv.org/pdf/2304.08612.pdf) achieves **second-order** accuracy and is **as fast as** the original Straight-Through, which has first-order accuracy.

<h3 align="center" id="st"><i>What is Straight-Through</i></h4>
<!-- ## Straight-Through and How It Works -->

Straight-Through (as below) bridges discrete variables (`y_hard`) and back-propagation. 
```python
y_soft = theta.softmax()

# one_hot_multinomial is a non-differentiable function
y_hard = one_hot_multinomial(y_soft) 

# with straight-through, the derivative of s_hard will
# act as if you had `p_soft` in the forward
y_hard = y_soft - y_soft.detach() + y_hard 
```
It is a long-standing mystery on how straight-through works, lefting doubts on many problems like whether we should use:
- `y_soft - y_soft.detach()`,
- ` (theta/tau).softmax() - (theta/tau).softmax().detach()`,
- or what?



<h3 align="center" id="reinmax"><i>Understand Straight-Through and Go Beyond</i></h3>
<!-- ## Better Performance with Negligible Computation Overheads -->

[We reveal](https://arxiv.org/pdf/2304.08612.pdf) that Straight-Through works as a special case of the forward Euler method, a numerical methods with first-order accuracy. 
Inspired by Heun's Method, a numerical method achieving second-order accuracy without requiring Hession or other second-order derivatives, we propose ReinMax, which *approximates gradient with second-order accuracy with negligible computation overheads.*

### How to use?

`reinmax` can be installed via `pip`
```
pip install reinmax
```

To replace Straight-Through Gumbel-Softmax with ReinMax: 

```diff
from reinmax import reinmax
...
- y_hard = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
+ y_hard, _ = reinmax(logits, tau) # note that reinmax prefers to set tau >= 1, while gumbel-softmax prefers to set tau < 1
...
```

To replace Straight-Through with ReinMax:
```diff
from reinmax import reinmax
...
- y_hard = one_hot_multinomial(logits.softmax()) 
- y_soft_tau = (logits/tau).softmax()
- y_hard = y_soft_tau - y_soft_tau.detach() + y_hard 
+ y_hard, y_soft = reinmax(logits, tau) 
...
```
### Examples

- [Polynomial Programming](https://github.com/LiyuanLucasLiu/reinmax_examples/tree/main/poly)
- [MNIST-VAE](https://github.com/LiyuanLucasLiu/reinmax_examples/tree/main/mnist_vae)
- [ListOps](https://github.com/LiyuanLucasLiu/reinmax_examples/tree/main/listops)

### Citation
Please cite the following papers if you found our model useful. Thanks!

>Liyuan Liu, Chengyu Dong, Xiaodong Liu, Bin Yu, and Jianfeng Gao (2023). Bridging Discrete and Backpropagation: Straight-Through and Beyond. *ArXiv, abs/2304.08612*.
```
@inproceedings{liu2023bridging,
  title={Bridging Discrete and Backpropagation: Straight-Through and Beyond},
  author = {Liu, Liyuan and Dong, Chengyu and Liu, Xiaodong and Yu, Bin and Gao, Jianfeng},
  booktitle = {arXiv:2304.08612 [cs]},
  year={2023}
}
```