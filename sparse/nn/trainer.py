r"""
Matching Pursuit Trainers.

.. autosummary::
   :toctree: toctree/nn

   TrainMatchingPursuit
   TrainLISTA
   TestMatchingPursuit
   TestMatchingPursuitParameters

"""


import torch
import torch.nn as nn
import torch.utils.data

from mighty.monitor.var_online import MeanOnline
from mighty.trainer import TrainerAutoencoder
from mighty.trainer.trainer import Trainer
from mighty.utils.algebra import compute_psnr, compute_sparsity
from mighty.utils.common import input_from_batch, batch_to_cuda, find_layers
from mighty.utils.data import DataLoader
from mighty.utils.stub import OptimizerStub
from sparse.nn.model import Softshrink, MatchingPursuit, LISTA

__all__ = [
    "TestMatchingPursuit",
    "TestMatchingPursuitParameters",
    "TrainMatchingPursuit",
    "TrainLISTA"
]


class TestMatchingPursuitParameters(TrainerAutoencoder):
    r"""
    Swipe through a range of Softshrink threshold parameters
    :code:`bmp_params_range` and **show** the best one.

    The user then can pick the "best" parameter by looking at the PSNR and
    sparsity plots - the choice is not trivial and depends on the application.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 bmp_params_range: torch.Tensor,
                 param_name="param",
                 **kwargs):
        super().__init__(model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=OptimizerStub(),
                         **kwargs)
        self.bmp_params = bmp_params_range
        self.param_name = param_name

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch

    def full_forward_pass(self, train=True):
        if not train:
            return None
        assert isinstance(self.criterion,
                          nn.MSELoss), "BMP can work only with MSE loss"

        mode_saved = self.model.training
        self.model.train(False)
        use_cuda = torch.cuda.is_available()

        loss_online = MeanOnline()
        psnr_online = MeanOnline()
        sparsity_online = MeanOnline()
        with torch.no_grad():
            for batch in self.data_loader.eval(
                    description="Full forward pass (eval)"):
                if use_cuda:
                    batch = batch_to_cuda(batch)
                input = input_from_batch(batch)
                loss = []
                psnr = []
                sparsity = []
                for bmp_param in self.bmp_params:
                    outputs = self.model(input, bmp_param)
                    latent, reconstructed = outputs
                    loss_lambd = self._get_loss(batch, outputs)
                    psnr_lmdb = compute_psnr(input, reconstructed)
                    sparsity_lambd = compute_sparsity(latent)
                    loss.append(loss_lambd.cpu())
                    psnr.append(psnr_lmdb.cpu())
                    sparsity.append(sparsity_lambd.cpu())

                loss_online.update(torch.stack(loss))
                psnr_online.update(torch.stack(psnr))
                sparsity_online.update(torch.stack(sparsity))

        loss = loss_online.get_mean()
        self.monitor.viz.line(Y=loss, X=self.bmp_params, win='Loss',
                              opts=dict(
                                  xlabel=f'BMP {self.param_name}',
                                  ylabel='Loss',
                                  title='Loss'
                              ))

        psnr = psnr_online.get_mean()
        self.monitor.viz.line(Y=psnr, X=self.bmp_params, win='PSNR',
                              opts=dict(
                                  xlabel=f'BMP {self.param_name}',
                                  ylabel='Peak signal-to-noise ratio',
                                  title='PSNR'
                              ))

        sparsity = sparsity_online.get_mean()
        self.monitor.viz.line(Y=sparsity, X=self.bmp_params, win='Sparsity',
                              opts=dict(
                                  xlabel=f'BMP {self.param_name}',
                                  ylabel='sparsity',
                                  title='L1 output sparsity'
                              ))

        self.monitor.viz.close(win='Accuracy')
        self.model.train(mode_saved)

        return loss

    def _epoch_finished(self, loss):
        Trainer._epoch_finished(self, loss)


class TestMatchingPursuit(TrainerAutoencoder):
    r"""
    Test Matching Pursuit with the fixed Softshrink threshold (embedded in a
    model) and trained weights.
    """

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch


class TrainMatchingPursuit(TrainerAutoencoder):
    r"""
    Train :code:`MatchingPursuit` or :code:`LISTA` AutoEncoder with
    :code:`LossPenalty` loss function, defined as

    .. math::
        L(\boldsymbol{W}, x_i) = \frac{1}{2} \left|\left| x_i -
        \boldsymbol{W} z_i \right|\right| +
        \lambda \left|\left| z_i \right|\right|_1^2

    where :math:`\boldsymbol{W} z_i` is a reconstruction of an input vector
    :math:`x_i`.

    The training process alternates between two steps:
      1) fix the dictionary matrix :math:`\boldsymbol{W}` and find the
      coefficients :math:`z_i` with Basis Pursuit;

      2) fix the coefficients :math:`z_i` and update the dictionary
      :math:`\boldsymbol{W}` with gradient descent.

    """

    watch_modules = TrainerAutoencoder.watch_modules + (Softshrink,
                                                        MatchingPursuit,
                                                        LISTA)

    def monitor_functions(self):
        super().monitor_functions()

        try:
            softshrink = next(find_layers(self.model, layer_class=Softshrink))
        except StopIteration:
            softshrink = None

        def lambda_mean(viz):
            # effective (positive) lambda
            lambd = softshrink.lambd.data.clamp(min=0).cpu()
            viz.line_update(y=[lambd.mean()], opts=dict(
                xlabel='Epoch',
                ylabel='Lambda mean',
                title='Softshrink lambda threshold',
            ))

        if softshrink is not None:
            self.monitor.register_func(lambda_mean)

        solver_online = self.model.solver.online

        def plot_dv_norm(viz):
            dv_norm = solver_online['dv_norm'].get_mean()
            viz.line_update(y=[dv_norm], opts=dict(
                xlabel='Epoch',
                ylabel='dv_norm (final improvement)',
                title='Solver convergence',
            ))

        def plot_iterations(viz):
            iterations = solver_online['iterations'].get_mean()
            viz.line_update(y=[iterations], opts=dict(
                xlabel='Epoch',
                ylabel='solver iterations',
                title='Solver iterations run',
            ))

        self.monitor.register_func(plot_dv_norm)
        self.monitor.register_func(plot_iterations)

    def _epoch_finished(self, loss):
        self.model.solver.reset_statistics()
        super()._epoch_finished(loss)

    def full_forward_pass(self, train=True):
        # Save convergence statistics for train forward pass only
        self.model.solver.save_stats = train
        loss = super().full_forward_pass(train=train)
        self.model.solver.save_stats = False
        return loss


class TrainLISTA(TrainMatchingPursuit):
    r"""
    Train LISTA with the original loss, defined in the paper as MSE between the
    latent vector Z (forward pass of LISTA NN) and the best possible latent
    vector Z*, obtained by running Basis Pursuit ADMM (shows better results
    that using original ISTA as the ground truth) on input X.

    .. math::
        L(W, X) = \frac{1}{2} \left|\left| Z^* - Z \right|\right|^2

    :code:`TrainLISTA` performs worse than :code:`TrainMatchingPursuit`.

    """

    def __init__(self,
                 model: nn.Module,
                 model_reference : nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer,
                 scheduler=None,
                 **kwargs):
        super().__init__(model, criterion=criterion, data_loader=data_loader,
                         optimizer=optimizer, scheduler=scheduler, **kwargs)
        model_reference.train(False)
        for param in model_reference.parameters():
            param.requires_grad_(False)
        self.model_reference = model_reference
        self.model.solver = self.model_reference.solver

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log("Reference model:")
        self.monitor.log_model(self.model_reference)

    def _get_loss(self, batch, output):
        assert isinstance(self.model, LISTA)
        input = input_from_batch(batch)
        latent, reconstructed = output
        latent_best, _ = self.model_reference(input)
        loss = self.criterion(latent, latent_best)
        return loss

    def _epoch_finished(self, loss):
        solver_online = self.model.solver.online
        self.monitor.plot_psnr(solver_online['psnr'].get_mean(),
                               mode='solver')
        self.monitor.update_sparsity(solver_online['sparsity'].get_mean(),
                                     mode='solver')
        super()._epoch_finished(loss)
