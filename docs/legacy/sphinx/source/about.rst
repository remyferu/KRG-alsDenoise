=====
About
=====

`denoise <https://github.com/AISDC/Noise2Inverse360>`_ provides a
self-supervised deep-learning pipeline for denoising CT reconstructions using the
`Noise2Inverse <https://arxiv.org/abs/2001.11801>`_ method.  Two convolution
modes are available: 2.5D (default) and 3D modes with different axial context and memory costs.

Method
------

Noise2Inverse trains on reconstructions from alternating angular subsets.
The method assumes independent noise in the measured data; shared calibration
and detector errors can violate that assumption.
A U-Net is trained to predict one subset reconstruction from the other, learning
to remove noise without any clean reference images.

The self-supervised training objective is:

.. math::

   \mathcal{L} = \frac{1}{2}\bigl[
       \|f(x_0) - x_1\|_1 + \|f(x_1) - x_0\|_1
   \bigr]

where :math:`x_0` and :math:`x_1` are two statistically independent
sub-reconstructions under the method’s noise assumptions, and :math:`f` is the denoising network.  After a warm-up
phase, a Laplacian Contrast Loss (LCL) term is added as a contrast regularizer.

Convolution modes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 35 30

   * - Mode
     - Flag
     - Network
     - Applications
   * - **2.5D** (default)
     - ``--mode 2.5d``
     - 2D U-Net without skip connections, GroupNorm, LeakyReLU.
       Stacks *N* adjacent axial slices as input channels.
     - Slice prediction with neighboring-slice context.
   * - **3D**
     - ``--mode 3d``
     - Full 3D U-Net with skip connections, layer norm.
       Processes cubic sub-volumes.
     - Volumetric restoration, including coherent X-ray microscopy (XNH).

Model context and memory
^^^^^^^^^^^^^^^^^^^^^^^^

2.5D predicts the center slice from neighboring slices passed as channels to
a 2D U-Net. The baseline uses GroupNorm and LeakyReLU without skip connections.
3D processes cubic patches with a volumetric U-Net and skip connections.
Both training datasets load paired TIFF stacks into CPU memory in each process.
3D convolutions also increase activation memory and computation per patch.

The 3D architecture and augmentation are adapted from Laugros et al. (2025)
and the ELEKTRONN3 implementation. Training applies matching cube rotations
and flips to paired patches. Single-slice inference is available only for 2.5D;
3D uses the ``volume`` command.

Training
--------

Training uses PyTorch Distributed Data Parallel (DDP) across one or more GPUs.
The ``denoise train`` command handles ``torchrun`` and ``PYTHONNOUSERSITE``
internally; no manual ``torchrun`` invocation is required::

    # 2.5D (default)
    denoise train --config baseline_config.yaml --gpus 0,1

    # 3D
    denoise train --config baseline_config.yaml --gpus 0,1 --mode 3d

The training loop alternates between two "views" (the two sub-reconstructions)
and minimises an L1 loss.  After a warm-up phase, a Laplacian Contrast Loss
(LCL) term is added.  In 3D mode, LCL is applied to the center axial slice of
each 3D prediction patch.

Three model checkpoints are saved automatically:

* ``best_val_model.pth``: lowest validation L1 loss
* ``best_lcl_model.pth``: lowest LCL regularisation loss
* ``best_edge_model.pth``: highest Laplacian edge score

Inference
---------

**2.5D mode**: single slice::

    denoise slice --config baseline_config.yaml --slice-number 500

**2.5D mode**: full volume::

    denoise volume --config baseline_config.yaml

**3D mode**: full volume (``slice`` is not supported in 3D mode)::

    denoise volume --config baseline_config.yaml --mode 3d

Both modes use sliding-window patch extraction with overlap-add blending
(Hann window).  In 3D mode the patches and blending window are cubic.

denoise is primarily intended for use with full-angle CT reconstructions
produced by `tomopy <https://tomopy.readthedocs.io>`_ or similar reconstruction
packages, and has been developed for synchrotron tomography beamlines and
coherent X-ray microscopy facilities.

References
----------

* Hendriksen et al. (2020). *Noise2Inverse: Self-supervised deep convolutional
  denoising for tomography.* IEEE Transactions on Computational Imaging.
  https://doi.org/10.1109/TCI.2020.3019647

* Laugros et al. (2025). *Self-supervised image restoration in coherent X-ray
  neuronal microscopy.* bioRxiv.
  https://doi.org/10.1101/2025.02.10.633538
