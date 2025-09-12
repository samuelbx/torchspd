# torchspd 🔥📐

Tiny toolkit of spectral operators on symmetric positive definite (SPD) matrices in PyTorch. It gives you differentiable `sqrtm`, `invsqrtm`, `logm`, `powm`, `expm`, a PSD projection, and a generic `apply_quad` for custom functions. Everything is batched, works with autograd (first order), and is written to be numerically stable near repeated or tiny eigenvalues.

```bash
pip install torchspd
```

## Usage

```python
import torch, torchspd as spd

# SPD input
n = 8
A = torch.randn(n, n)
A = A @ A.T + n * torch.eye(n)

R = spd.sqrtm(A)     # Matrix square root
W = spd.invsqrtm(A)  # Inverse square root
L = spd.logm(A)      # Logarithm
A_back = spd.expm(L) # Exponential
B = spd.powm(A, 0.3) # Fractional power

# Projection onto the PSD cone
X = torch.randn(n, n)
X = 0.5 * (X + X.T)
P = spd.proj_psd(X)

# Reuse eigenpairs
Y, (L, V) = spd.logm(A, return_eig=True)
S = spd.sqrtm(A, eig=(L, V))
P = spd.powm(A, 0.3, eig=(L, V))
```

More generally, `apply_quad` lets you define your own spectral function $f$ and its derivative $f'$, and computes $F(A)=V f(\Lambda) V^\top$ differentiably w.r.t. $A=V\Lambda V^\top$.

```python
def f(x): return torch.log1p(x)
def df(x): return 1 / (1 + x)

Y = spd.apply_quad(A, f, df)
```

*⚠ `apply_quad` uses Gauss-Legendre quadrature near close eigenvalues. This is approximate: for important cases prefer explicit formulas like in sqrtm, invsqrtm... (see derivation guide below).*

## Calculations

Let $A=V\Lambda V^\top$ in $\mathbb{R}^{d\times d}$ be SPD, with $\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_n)$ and $VV^\top=I_d$, and let $f:\mathbb{R}_+^*\rightarrow\mathbb{R}$ be continuously differentiable. We define $F(A)=Vf(\Lambda)V^\top$, where $f(\Lambda)=\mathrm{diag}(f(\lambda_1),\ldots,f(\lambda_n))$. This function is well-defined on the set of SPD matrices, as it does not depend on the choice of $V$.

By Daleckii-Krein, the Fréchet derivative of $f(X)$ at $X=A$, applied to the symmetric perturbation $H$, verifies
$$\mathrm{d}F_A(H)=V\left(G\circ (V^\top HV)\right)V^\top,$$
where $\circ$ denotes the coordinatewise matrix product, and
$$G_{ij} = \frac{f(\lambda_i)-f(\lambda_j)}{\lambda_i-\lambda_j} \quad (\text{with } G_{ii}=f'(\lambda_i)).$$

Here are the values of $G_{ij}$ in common special cases: they are used in the implementation. We note $\delta=\frac{\lambda_i-\lambda_j}{\lambda_j}$ and $\mathrm{sinhc}(x)=\frac{\sinh(x)}{x}$ ($\mathrm{sinhc}(0)=1$).

| Function $f(x)$                   | Off-diagonal terms $G_{ij}, i\neq j$                                                                 | Diagonal terms $G_{ii}$          |
|-----------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------|
| $f(x) = \sqrt{x}$                 | $\dfrac{1}{\sqrt{\lambda_i} + \sqrt{\lambda_j}}$                                                        | $\dfrac{1}{2\sqrt{\lambda_i}}$   |
| $f(x) = 1/\sqrt{x}$               | $-\dfrac{1}{\sqrt{\lambda_i}\sqrt{\lambda_j}(\sqrt{\lambda_i}+\sqrt{\lambda_j})}$                   | $-\dfrac{1}{2\lambda_i^{3/2}}$   |
| $f(x) = \log x$                   | $\dfrac{1}{\lambda_j}\dfrac{\log(1+\delta)}{\delta}$ | $\dfrac{1}{\lambda_i}$           |
| $f(x) = e^{x}$                    | $e^{(\lambda_i+\lambda_j)/2}\mathrm{sinhc}\!\left(\tfrac{\lambda_i-\lambda_j}{2}\right)$              | $e^{\lambda_i}$                  |
| $f(x) = x^p,\; p\in\mathbb{R}$    | $\lambda_j^{p-1}\dfrac{(1-\delta)^{p}-1}{\delta}$ | $p\lambda_i^{p-1}$           |

We also use the Taylor expansions of these formulas when $\lambda_i \approx \lambda_j$, replacing divided differences by series in $\delta = (\lambda_i-\lambda_j)/\lambda_j$ to avoid numerical cancellation.

For generic $f$ and $i\ne j$, we rely on the fact that 
$$G_{ij}=\int_{0}^{1}f'\left((1-t)\lambda_{j}+t\lambda_{i}\right)\text{d}t.$$
We currently approximate this integral using the Gauss-Legendre rule.

## References

- Functions of Matrices: Theory and Computation (Higham, 2008).
- Improved Inverse Scaling and Squaring Algorithms for the Matrix Logarithm (Al-Mohy & Higham, 2012).
- A Formula for the Fréchet Derivative of a Generalized Matrix Function (Noferini, 2016).
