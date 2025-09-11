import torch
from torch.autograd.function import once_differentiable
from typing import Callable, Optional, Tuple, Dict

__all__ = ['symmetrize', 'apply_quad', 'powm', 'logm', 'expm', 'sqrtm', 'invsqrtm', 'proj_psd']

_GL3_CACHE: Dict[Tuple[torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

def symmetrize(a: torch.Tensor) -> torch.Tensor:
  '''Return the symmetric part of a matrix: (A+A^T)/2.'''
  return .5 * (a + a.transpose(-2, -1))

def _eig_sym(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return torch.linalg.eigh(symmetrize(a))

def _tiny(dtype: torch.dtype) -> float: return torch.finfo(dtype).tiny

def _eps(dtype: torch.dtype) -> float: return torch.finfo(dtype).eps

def _pairwise_grids(L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return L.unsqueeze(-1), L.unsqueeze(-2)

def _diag_set(G: torch.Tensor, diag_new: torch.Tensor) -> torch.Tensor:
  return G + torch.diag_embed(diag_new - torch.diagonal(G, dim1=-2, dim2=-1))

def _quad_nodes_weights_like(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  key = (x.dtype, x.device)
  if key in _GL3_CACHE:
    return _GL3_CACHE[key]
  s = torch.sqrt(torch.tensor(3.0 / 5.0, dtype=x.dtype, device=x.device))
  t = torch.stack([(1 - s) / 2, torch.tensor(.5, dtype=x.dtype, device=x.device), (1 + s) / 2])
  w = .5 * torch.tensor([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=x.dtype, device=x.device)
  _GL3_CACHE[key] = (t, w)
  return t, w

def _stability_mask(li: torch.Tensor, lj: torch.Tensor, threshold_factor: float) -> torch.Tensor:
  diff = (li - lj).abs()
  scale = torch.maximum(torch.maximum(li.abs(), lj.abs()), torch.ones_like(li))
  return diff <= scale * li.new_tensor(_eps(li.dtype)).sqrt() * threshold_factor

def _sinhc(x: torch.Tensor) -> torch.Tensor:
  small = x.abs() <= x.new_tensor(_eps(x.dtype)).sqrt()
  series = 1 + (x * x) / 6 + (x ** 4) / 120
  val = torch.sinh(x) / x
  return torch.where(small, series, val)

class _SpectralCore(torch.autograd.Function):
  @staticmethod
  def forward(ctx, a: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], g_builder: Callable[[torch.Tensor], torch.Tensor],
              eig: Optional[Tuple[torch.Tensor, torch.Tensor]], return_eig: bool):
    if eig is None:
      L, V = _eig_sym(a)
    else:
      L, V = eig
    y = (V * f(L).unsqueeze(-2)) @ V.transpose(-2, -1)
    if ctx.return_eig:
      ctx.mark_non_differentiable(L, V)
    ctx.save_for_backward(L, V)
    ctx.g_builder = g_builder
    ctx.return_eig = bool(return_eig)
    if ctx.return_eig:
      return y, L, V
    else:
      return y

  @staticmethod
  @once_differentiable
  def backward(ctx, *grad_outputs):
    g = grad_outputs[0]
    L, V = ctx.saved_tensors
    ghat = V.transpose(-2, -1) @ symmetrize(g) @ V
    G = ctx.g_builder(L)
    xhat = G * ghat
    x = V @ xhat @ V.transpose(-2, -1)
    x = symmetrize(x)
    return x, None, None, None, None

def _apply_spectral(a: torch.Tensor,
                    f: Callable[[torch.Tensor], torch.Tensor],
                    g_builder: Callable[[torch.Tensor], torch.Tensor],
                    eig: Optional[Tuple[torch.Tensor, torch.Tensor]],
                    return_eig: bool):
  out = _SpectralCore.apply(a, f, g_builder, eig, return_eig)
  if return_eig:
    y, L, V = out
    return y, (L, V)
  else:
    return out

def apply_quad(a: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], df: Callable[[torch.Tensor], torch.Tensor], *, 
               eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_eig: bool = False, threshold_factor: float = 1.0):
  '''
  Apply f to SPD A: y = V f(Λ) V^T.
  Uses divided differences G_ij = (f(λi)−f(λj))/(λi−λj), replaced when λi, λj are close
  by a 3-point Gauss–Legendre approximation of ∫₀¹ f'((1−t)λj+tλi) dt.

  Parameters
  ----------
  - a: SPD matrix (or batch).
  - f: function on eigenvalues.
  - df: derivative of f.
  - eig: (Λ, V), optional cached eigendecomposition.
  - return_eig: if True, also return (Λ, V).
  - threshold_factor: controls closeness tolerance: |λi−λj| ≤ max(|λi|,|λj|,1)·√ε·factor.
  '''
  def g_builder(L: torch.Tensor) -> torch.Tensor:
    Lc = L.clamp_min(_tiny(L.dtype))
    li, lj = _pairwise_grids(Lc)
    fi, fj = f(li), f(lj)
    diff = li - lj
    Graw = (fi - fj) / diff
    mask = _stability_mask(li, lj, threshold_factor)
    t, w = _quad_nodes_weights_like(li)
    m = (1 - t) * lj.unsqueeze(-1) + t * li.unsqueeze(-1)
    Gquad = (df(m) * w).sum(dim=-1)
    G = torch.where(mask, Gquad, Graw)
    return _diag_set(G, df(Lc))
  return _apply_spectral(a, f, g_builder, eig, return_eig)

def sqrtm(a: torch.Tensor, *, eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_eig: bool = False):
  '''
  Matrix square root f(A) = A^{1/2} = V Λ^{1/2} V^T.

  Parameters
  ----------
  - a: SPD matrix (or batch).
  - eig: (Λ, V), optional cached eigendecomposition.
  - return_eig: if True, also return (Λ, V).
  '''
  def f(L: torch.Tensor) -> torch.Tensor:
    return L.clamp_min(_tiny(L.dtype)).sqrt()
  def g_builder(L: torch.Tensor) -> torch.Tensor:
    Lc = L.clamp_min(_tiny(L.dtype))
    si, sj = _pairwise_grids(f(Lc))
    G = 1 / (si + sj)
    return G
  return _apply_spectral(a, f, g_builder, eig, return_eig)

def invsqrtm(a: torch.Tensor, *, eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_eig: bool = False):
  '''
  Matrix inverse square root f(A) = A^{-1/2} = V Λ^{-1/2} V^T.

  Parameters
  ----------
  - a: SPD matrix (or batch).
  - eig: (Λ, V), optional cached eigendecomposition.
  - return_eig: if True, also return (Λ, V).
  '''
  def f(L: torch.Tensor) -> torch.Tensor:
    return L.clamp_min(_tiny(L.dtype)).sqrt().reciprocal()
  def g_builder(L: torch.Tensor) -> torch.Tensor:
    Lc = L.clamp_min(_tiny(L.dtype))
    s = Lc.sqrt()
    si, sj = _pairwise_grids(s)
    G = -1 / (si * sj * (si + sj))
    return G
  return _apply_spectral(a, f, g_builder, eig, return_eig)

def logm(a: torch.Tensor, *, eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_eig: bool = False, threshold_factor: float = 1.0):
  '''
  Matrix logarithm f(A) = log(A) = V log(Λ) V^T.

  Parameters
  ----------
  - a: SPD matrix (or batch).
  - eig: (Λ, V), optional cached eigendecomposition.
  - return_eig: if True, also return (Λ, V).
  - threshold_factor: controls tolerance for replacing divided
    differences by a series expansion when λi, λj are close.
  '''
  def f(L: torch.Tensor) -> torch.Tensor:
    return torch.log(L.clamp_min(_tiny(L.dtype)))
  def g_builder(L: torch.Tensor) -> torch.Tensor:
    Lc = L.clamp_min(_tiny(L.dtype))
    li, lj = _pairwise_grids(Lc)
    delta = (li - lj) / lj
    r = torch.log1p(delta) / delta
    mask = _stability_mask(li, lj, threshold_factor)
    s = 1 - 0.5 * delta + (delta * delta) / 3 - (delta ** 3) / 4 + (delta ** 4) / 5
    r = torch.where(mask, s, r)
    G = r / lj
    return _diag_set(G, Lc.reciprocal())
  return _apply_spectral(a, f, g_builder, eig, return_eig)

def expm(a: torch.Tensor, *, eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_eig: bool = False):
  '''
  Matrix exponential f(A) = exp(A). Generally falls back to Torch's implementation.
  
  Parameters
  ----------
  - a: symmetric matrix (or batch).
  - eig: (Λ, V), optional cached eigendecomposition.
  - return_eig: if True, also return (Λ, V).
  '''
  if eig is None and not return_eig:
    return torch.linalg.matrix_exp(a)
  def f(L: torch.Tensor) -> torch.Tensor:
    return torch.exp(L)
  def g_builder(L: torch.Tensor) -> torch.Tensor:
    li, lj = L.unsqueeze(-1), L.unsqueeze(-2)
    h, m = .5 * (li - lj), .5 * (li + lj)
    G = torch.exp(m) * _sinhc(h)
    return _diag_set(G, torch.exp(L))
  return _apply_spectral(a, f, g_builder, eig, return_eig)

def powm(a: torch.Tensor, p: float, *, eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_eig: bool = False, threshold_factor: float = 1.0):
  '''
  Matrix power f(A) = A^p = V Λ^p V^T for real p.

  Parameters
  ----------
  - a: SPD matrix (or batch).
  - p: real exponent.
  - eig: (Λ, V), optional cached eigendecomposition.
  - return_eig: if True, also return (Λ, V).
  - threshold_factor: controls tolerance for replacing divided
    differences by a series expansion when λi, λj are close.
  '''
  def f(L: torch.Tensor) -> torch.Tensor:
    return L.clamp_min(_tiny(L.dtype)).pow(p)
  def g_builder(L: torch.Tensor) -> torch.Tensor:
    Lc = L.clamp_min(_tiny(L.dtype))
    li, lj = _pairwise_grids(Lc)
    delta = (li - lj) / lj
    r = torch.expm1(p * torch.log1p(delta)) / delta
    mask = _stability_mask(li, lj, threshold_factor)
    s = p * (1 + (p - 1) * delta / 2 + (p - 1) * (p - 2) * (delta ** 2) / 6
             + (p - 1) * (p - 2) * (p - 3) * (delta ** 3) / 24)
    r = torch.where(mask, s, r)
    G = lj.pow(p - 1) * r
    return _diag_set(G, p * Lc.pow(p - 1))
  return _apply_spectral(a, f, g_builder, eig, return_eig)

def proj_psd(a: torch.Tensor) -> torch.Tensor:
  '''Projection onto the PSD cone.'''
  L, V = _eig_sym(a)
  Lc = L.clamp_min(0)
  return (V * Lc.unsqueeze(-2)) @ V.transpose(-2, -1)
