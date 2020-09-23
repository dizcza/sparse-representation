import torch
import torch.nn as nn
from torchvision.datasets import MNIST

from mighty.loss import LossPenalty
from mighty.monitor.accuracy import AccuracyEmbedding
from mighty.monitor.monitor import MonitorLevel
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader, TransformDefault
from mighty.utils.stub import OptimizerStub
from sparse.nn.model import MatchingPursuit, LISTA
from sparse.nn.solver import BasisPursuitADMM
from sparse.nn.trainer import TestMatchingPursuitParameters, \
    TestMatchingPursuit, TrainMatchingPursuit, TrainLISTA


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=15,
                                                           threshold=1e-3,
                                                           min_lr=1e-4)
    return optimizer, scheduler


def test_matching_pursuit_lambdas(dataset_cls=MNIST):
    model = MatchingPursuit(784, 2048)
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist(
                                 normalize=None
                             ))
    bmp_lambdas = torch.linspace(0.05, 0.95, steps=10)
    trainer = TestMatchingPursuitParameters(model,
                                            criterion=nn.MSELoss(),
                                            data_loader=data_loader,
                                            bmp_params_range=bmp_lambdas,
                                            param_name='lambda')
    trainer.train(n_epochs=1, mutual_info_layers=0)


def test_matching_pursuit(dataset_cls=MNIST):
    # vanilla Matching Pursuit: the weights are fixed, no training
    model = MatchingPursuit(784, 2048)
    data_loader = DataLoader(dataset_cls,
                             eval_size=10000,
                             transform=TransformDefault.mnist(
                                 normalize=None
                             ))
    trainer = TestMatchingPursuit(model,
                                  criterion=nn.MSELoss(),
                                  data_loader=data_loader,
                                  optimizer=OptimizerStub(),
                                  accuracy_measure=AccuracyEmbedding())
    trainer.train(n_epochs=1, mutual_info_layers=0)


def train_matching_pursuit(dataset_cls=MNIST, n_epochs=5):
    # Typically, the 'out_features', the second parameter of MatchingPursuit
    # model, should be greater than the 'in_features'.
    # In case of MNIST, it works even with the smaller values, but the
    # resulting embedding vector is dense.

    model = MatchingPursuit(784, 256, solver=BasisPursuitADMM())
    # model = LISTA(784, 128)

    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist(
                                 normalize=None
                             ))
    criterion = LossPenalty(nn.MSELoss(), lambd=model.lambd)
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainMatchingPursuit(model,
                                   criterion=criterion,
                                   data_loader=data_loader,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   accuracy_measure=AccuracyEmbedding())
    trainer.restore()
    n_epochs = max(n_epochs - trainer.epoch, 0)
    trainer.train(n_epochs=n_epochs, mutual_info_layers=0)
    return trainer.model


def train_lista(dataset_cls=MNIST):
    model_ref = train_matching_pursuit(dataset_cls=dataset_cls)
    model = LISTA(784, out_features=model_ref.out_features)
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist(
                                 normalize=None
                             ))
    criterion = nn.MSELoss()
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainLISTA(model,
                         model_reference=model_ref,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         accuracy_measure=AccuracyEmbedding())
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epochs=10, mutual_info_layers=0)


if __name__ == '__main__':
    set_seed(28)
    # test_matching_pursuit()
    # train_matching_pursuit()
    train_lista()
