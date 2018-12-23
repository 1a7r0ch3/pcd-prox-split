## Preconditioned Proximal Splitting Algorithms

Generic C++ classes for implementing preconditioned proximal splitting algorithms.  
Specialization to preconditioned generalized forward-backward or forward-Douglas–Rachford proximal splitting algorithms, on problems involving graph total variation, as explained in our articles [(Raguet and Landrieu, 2015, and Raguet, 2018)](#references).  
Parallel implementation with OpenMP API.  
MEX API for interface with GNU Octave or Matlab.  

### Generic classes
`Pcd_prox` class is the most generic, with minimalist structure for a preconditioned proximal splitting algorithm.  
`Pfdr` class specializes for the preconditioned generalized forward-backward or forward-Douglas–Rachford proximal splitting algorithms: introduces preconditioner _Γ_, weights _W_, Lipschitz metric _L_.  
`Pfdr_d1` class specializes `Pfdr` for the graph total variation.  

### Specialization `Pfdr_d1_ql1b`: quadratic functional, ℓ<sub>1</sub> norm, bounds, and graph total variation
Minimize functionals over a graph _G_ = (_V_, _E_), of the form   

    _F_: _x_ ∈ ℝ<sup>_V_</sup> ↦  1/2 ║_y_<sup>(q)</sup> − _A_<i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _λ_<sub>_v_</sub> |_y_<sup>(ℓ<sub>1</sub>)</sup> − _x_<sub>_v_</sub>| +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>[_m_<sub>_v_</sub>, _M_<sub>_v_</sub>]</sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,   

where _y_<sup>(q)</sup> ∈ ℝ<sup>_n_</sup>, 
_A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, 
_y_<sup>(ℓ<sub>1</sub>)</sup> ∈ ℝ<sup>_V_</sup> and 
_λ_ ∈ ℝ<sup>_V_</sup> and _w_ ∈ ℝ<sup>_E_</sup> are regularization weights, 
_m_, _M_ ∈ ℝ<sup>_V_</sup> are parameters and 
_ι_<sub>[_a_,_b_]</sub> is the convex indicator of [_a_, _b_] : x ↦ 0 if _x_ ∈ [_a_, _b_], +∞ otherwise.  

When _y_<sup>(ℓ<sub>1</sub>)</sup> is zero, the combination of ℓ<sub>1</sub> norm and total variation is sometimes coined _fused LASSO_.  

Currently, _A_ must be provided as a matrix. See the documentation for special cases.  

### Specialization `Pfdr_d1_lsx` : separable loss, simplex constraints, and graph total variation
_K_ being a set of labels, minimize functionals over a graph _G_ = (_V_, _E_), of the form   

    _F_: _x_ ∈ ℝ<sup>_V_ ⨯ _K_</sup> ↦  _f_(_y_, _x_) +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>Δ<sub>_K_</sub></sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 ∑<sub>_k_ ∈ _K_</sub> _λ_<sub>_k_</sub> |_x_<sub>_u_,_k_</sub> − _x_<sub>_v_,_k_</sub>| ,  

where _y_ ∈ ℝ<sup>_V_ ⨯ _K_</sup>, _f_ is a loss functional (see below), _w_ ∈ ℝ<sup>_E_</sup> and _λ_ ∈ ℝ<sup>_K_</sup> are regularization weights, and _ι_<sub>Δ<sub>_K_</sub></sub> is the convex indicator of the simplex
Δ<sub>_K_</sub> = {_x_ ∈ ℝ<sup>_K_</sup> | ∑<sub>_k_</sub> _x_<sub>_k_</sub> = 1 and ∀ _k_, _x_<sub>_k_</sub> ≥ 0}: _x_ ↦ 0 if _x_ ∈ Δ<sub>_K_</sub>, +∞ otherwise. 

The following loss functionals are available.  
Linear: _f_(_y_, _x_) = − ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> _x_<sub>_v_,_k_</sub> _y_<sub>_v_,_k_</sub>  
Quadratic: _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> (_x_<sub>_v_,_k_</sub> − _y_<sub>_v_,_k_</sub>)<sup>2</sup>  
Smoothed Kullback–Leibler divergence: _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub>
KL(_α_ _u_ + (1 − _α_) _y_<sub>_v_</sub>, _α_ _u_ + (1 − _α_) _x_<sub>_v_</sub>),  
where _α_ ∈ \]0,1\[,
_u_ ∈ Δ<sub>_K_</sub> is the uniform discrete distribution over _K_,
and
KL: (_p_, _q_) ↦ ∑<sub>_k_ ∈ _K_</sub> _p_<sub>_k_</sub> log(_p_<sub>_k_</sub>/_q_<sub>_k_</sub>).  

### Directory tree
    .   
    ├── include/    C++ headers, with some doc  
    ├── octave/     GNU Octave or Matlab code  
    │   ├── doc/    some documentation  
    │   └── mex/    MEX API  
    └── src/        C++ sources  

### C++
The C++ classes are documented within the corresponding headers in `include/`.  

### GNU Octave or Matlab
The MEX interfaces are documented within dedicated `.m` files in `mex/doc/`.  
See `mex/compile_mex.m` for typical compilation commands under UNIX systems.  

### References
H. Raguet and L. Landrieu, [Preconditioning of a Generalized Forward-Backward Splitting and Application to Optimization on Graphs](https://1a7r0ch3.github.io/pgfb/), 2015.

H. Raguet, [A Note on the Forward-Douglas-Rachford Splitting Algorithm for Monotone Inclusion and Convex Optimization](https://1a7r0ch3.github.io/fdr/), 2018.
