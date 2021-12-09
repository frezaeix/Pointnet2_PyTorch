My Own Readme 
-----
Explaining the flow of Pointnet2


0- class PointnetSAModule(PointnetSAModuleMSG)
   class PointnetSAModuleMSG(_PointnetSAModuleBase)

1- class _PointnetSAModuleBase(nn.Module) -> forward: 
   a. Takes point cloud xyz (B, N, 3) [and features (B, C, N)], 
   b. Applies FPS then gather_operation. output: new_xyz (B, npoint, 3)
   c. In FPS it selects subsets of the point cloud (indices) and marks it (output) as non_differentiable.
   d. In GatherOperation it saves indices for the backward pass. There is a specific backward for this node:

::

   @staticmethod
   def backward(ctx, grad_out):
       idx, features = ctx.saved_tensors
       N = features.size(2)
       grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
       return grad_features, None   

2- class QueryAndGroup(nn.Module) -> forward:
   a. Takes xyz, sampled_xyz or centers of balls (new_xyz) and features.
   b. Applies ball_query(self.radius, self.nsample, xyz, new_xyz) and it outputs:
         new_features : (B, 3 + C, npoint, self.nsample) tensor with the indicies of the features that form the query balls
         It marks the output as non_differentiable.
   c. Applies GroupingOperation: grouping_operation(xyz_trans, idx). This operation is similar to GatherOperation. It saves indices for the backward pass. Backward for this node:

::

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)
::   
::   
   
   d. Applies GroupingOperation: grouping_operation(features, idx).
   e. Return: 
   
 ::
 
   new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)


Original Readme by the authors of Pointnet2/Pointnet++ PyTorch
============================


**Project Status**: Unmaintained.  Due to finite time, I have no plans to update this code and I will not be responding to issues.

* Implemention of Pointnet2/Pointnet++ written in `PyTorch <http://pytorch.org>`_.

* Supports Multi-GPU via `nn.DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_.

* Supports PyTorch version >= 1.0.0.  Use `v1.0 <https://github.com/erikwijmans/Pointnet2_PyTorch/releases/tag/v1.0>`_
  for support of older versions of PyTorch.


See the official code release for the paper (in tensorflow), `charlesq34/pointnet2 <https://github.com/charlesq34/pointnet2>`_,
for official model definitions and hyper-parameters.

The custom ops used by Pointnet++ are currently **ONLY** supported on the GPU using CUDA.

Setup
-----

* Install ``python`` -- This repo is tested with ``{3.6, 3.7}``

* Install ``pytorch`` with CUDA -- This repo is tested with ``{1.4, 1.5}``.
  It may work with versions newer than ``1.5``, but this is not guaranteed.


* Install dependencies

  ::

    pip install -r requirements.txt







Example training
----------------

Install with: ``pip install -e .``

There example training script can be found in ``pointnet2/train.py``.  The training examples are built
using `PyTorch Lightning <https://github.com/williamFalcon/pytorch-lightning>`_ and `Hydra <https://hydra.cc/>`_.


A classifion pointnet can be trained as

::

  python pointnet2/train.py task=cls

  # Or with model=msg for multi-scale grouping

  python pointnet2/train.py task=cls model=msg


Similarly, semantic segmentation can be trained by changing the task to ``semseg``

::

  python pointnet2/train.py task=semseg



Multi-GPU training can be enabled by passing a list of GPU ids to use, for instance

::

  python pointnet2/train.py task=cls gpus=[0,1,2,3]


Building only the CUDA kernels
----------------------------------


::

  pip install pointnet2_ops_lib/.

  # Or if you would like to install them directly (this can also be used in a requirements.txt)

  pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"






Contributing
------------

This repository uses `black <https://github.com/ambv/black>`_ for linting and style enforcement on python code.
For c++/cuda code,
`clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ is used for style.  The simplest way to
comply with style is via `pre-commit <https://pre-commit.com/>`_

::

  pip install pre-commit
  pre-commit install



Citation
--------

::

  @article{pytorchpointnet++,
        Author = {Erik Wijmans},
        Title = {Pointnet++ Pytorch},
        Journal = {https://github.com/erikwijmans/Pointnet2_PyTorch},
        Year = {2018}
  }

  @inproceedings{qi2017pointnet++,
      title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
      author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5099--5108},
      year={2017}
  }
