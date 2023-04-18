.. ReinMax documentation file.

:github_url: https://github.com/microsoft/ReinMax

*************************
ReinMax documentation
*************************

ReinMax achieves **second-order** accuracy and is **as fast as** the original Straight-Through, which has first-order accuracy.

We reveal that Straight-Through works as a special case of the forward Euler method, a numerical methods with first-order accuracy. 
Inspired by Heun's Method, a numerical method achieving second-order accuracy without requiring Hession or other second-order derivatives, we propose ReinMax, which approximates gradient with second-order accuracy with negligible computation overheads.

.. autofunction:: reinmax.reinmax
