# DL-ROM-Meth

Source code for deep learning-based reduced order models for the solution of nonlinear time-dependent parametrized PDEs. 

### Abstract
Conventional reduced order modeling techniques such as the reduced basis (RB) method (relying, e.g., on proper orthogonal decomposition (POD)) may incur in severe limitations when dealing with nonlinear time-dependent parametrized PDEs, as these are strongly anchored to the assumption of modal linear superimposition they are based on. For problems featuring coherent structures that propagate over time such as transport, wave, or convection-dominated phenomena, the RB method may yield inefficient reduced order models (ROMs) when very high levels of accuracy are required. To overcome this limitation, in this work, we propose a new nonlinear approach to set ROMs by exploiting deep learning (DL) algorithms. In the resulting nonlinear ROM, which we refer to as DL-ROM, both the nonlinear trial manifold (corresponding to the set of basis functions in a linear ROM) as well as the nonlinear reduced dynamics (corresponding to the projection stage in a linear ROM) are learned in a non-intrusive way by relying on DL algorithms; the latter are trained on a set of full order model (FOM) solutions obtained for different parameter values. We show how to construct a DL-ROM for both linear and nonlinear time-dependent parametrized PDEs. Moreover, we assess its accuracy and efficiency on different parametrized PDE problems. Numerical results indicate that DL-ROMs whose dimension is equal to the intrinsic dimensionality of the PDE solutions manifold are able to efficiently approximate the solution of parametrized PDEs, especially in cases for which a huge number of POD modes would have been necessary to achieve the same degree of accuracy.

### Citation
```
Fresca, S., Dede’, L. and Manzoni, A. A Comprehensive Deep Learning-Based Approach to Reduced Order Modeling of Nonlinear Time-Dependent Parametrized PDEs. J Sci Comput 87, 61 (2021). https://doi.org/10.1007/s10915-021-01462-7
```

For the full implementation of the DL-ROM neural network, please write to stefania.fresca@polimi.it.
