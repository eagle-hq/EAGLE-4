# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Kernels of the EAGLE-4 decode path. Raw tensors in, raw tensors out; no engine imports.

==================  =====================================================================================
module              what
==================  =====================================================================================
``arch``            which GPU can run what (one place for every capability check); ``python -m ... arch``
``candidate_ops``   candidate head: shard top-k, exact re-scoring, candidate tail (``candidate_topk``)
``candidate``       ctypes launchers of the native CUDA kernels of ``csrc/candidate.cu`` (``native``, ``build``)
``candidate_triton``  the same kernels in Triton: every supported GPU, bit-identical
``gather_dot``      exact LM-head row scoring (+ the packed TP gather layout)
``tree_step``       one Triton launch for a draft-tree step
``online_softmax``  the online-softmax step of the engines' row attention kernels
``gdn_tree``        GDN tree-verify tree tables (``gdn_tree_conv``: conv; ``gdn_tree_recurrence``: recurrence;
                    ``gdn_tree_gluon``: the recurrence's Gluon twin on B300)
==================  =====================================================================================

Nothing is compiled at import: Triton kernels compile on first launch, the native library is loaded (or built with
nvcc) on first use, and every launcher falls back to a Triton or torch path where the hardware or shape needs it.
"""
