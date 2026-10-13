*******
Credits
*******

Citations
=========

The upstream implementation draws on the following papers. Cite the relevant
method and architecture when reporting results.

:cite:`Hendriksen:20`: Noise2Inverse

   Self-supervised denoising from reconstructions of interleaved angular subsets.
   Its noise assumptions do not establish independence for every experimental
   scan, especially when calibration or detector errors are shared.

:cite:`Hendriksen:21`: Multi-dimensional Noise2Inverse

   Extension to spatial, temporal and spectral context. The 2.5D implementation
   takes neighboring axial slices as channels, controlled by ``n_slices``.

:cite:`Laugros:25`: Volumetric restoration

   Source of the 3D U-Net and paired cube-rotation augmentation used in
   ``--mode 3d``. See ``model3d.py`` and ``data3d.py`` for implementation details.

:cite:`Yunker:25`: Model selection

   Reference for model selection during self-supervised training. This repository
   saves separate checkpoints for evaluation L1, LCL and edge scores. These
   criteria differ from clean-reference quality measurements.

.. bibliography:: bibtex/cite.bib
   :style: plain
   :labelprefix: A

Reference
---------

.. bibliography:: bibtex/ref.bib
   :style: plain
   :labelprefix: B
   :all:
