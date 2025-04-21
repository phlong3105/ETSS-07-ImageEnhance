#!/usr/bin/env python

"""Tests for NIR import and export."""

import nir
import pytest
import snntorch as snn
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir
import torch

# sample data for snntorch_sequential
@pytest.fixture(scope="module")
def sample_data():
    return torch.ones((4, 784))


# sample data for snntorch with conv2d_avgpool
@pytest.fixture(scope="module")
def sample_data2():
    return torch.randn(1, 1, 28, 28)


class NetWithAvgPool(torch.nn.Module):
    def __init__(self):
        super(NetWithAvgPool, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        self.fc1 = torch.nn.Linear(28 * 28 * 16 // 4, 500)
        self.lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(
            self.conv1(x), kernel_size=2, stride=2
        )
        x = x.view(-1, 28 * 28 * 16 // 4)
        x = self.lif1(x)
        x = self.fc1(x)
        x = self.lif2(x)
        return x


@pytest.fixture(scope="module")
def net_with_avg_pool():
    net = NetWithAvgPool()
    return net


@pytest.fixture(scope="module")
def snntorch_sequential():
    lif1 = snn.Leaky(beta=0.9, init_hidden=True)
    lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


@pytest.fixture(scope="module")
def snntorch_recurrent():
    v = torch.ones((500,))
    lif1 = snn.RSynaptic(
        alpha=0.5, beta=0.9, V=v, all_to_all=False, init_hidden=True
    )
    lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


class TestNIR:
    """Test import and export from snnTorch to NIR."""

    def test_export_sequential(self, snntorch_sequential, sample_data):
        nir_graph = export_to_nir(snntorch_sequential, sample_data)
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output"] + [str(i) for i in range(4)]
        ), nir_graph.nodes.keys()
        assert set(nir_graph.edges) == set(
            [
                ("3", "output"),
                ("input", "0"),
                ("2", "3"),
                ("1", "2"),
                ("0", "1"),
            ]
        )
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1"], nir.LIF)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)

    def test_export_NetWithAvgPool(self, net_with_avg_pool, sample_data2):
        nir_graph = export_to_nir(net_with_avg_pool, sample_data2)
        assert nir_graph is not None
        # dict_keys(['conv1', 'fc1', 'input', 'lif1', 'lif2', 'output'])
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output"]
            + ["conv1", "fc1", "input", "lif1", "lif2", "output"]
        ), nir_graph.nodes.keys()
        assert set(nir_graph.edges) == set(
            [
                ("lif2", "output"),
                ("lif1", "fc1"),
                ("fc1", "lif2"),
                ("input", "conv1"),
                ("conv1", "output"),
            ]
        )
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["conv1"], nir.Conv2d)
        assert isinstance(nir_graph.nodes["lif1"], nir.LIF)
        assert isinstance(nir_graph.nodes["fc1"], nir.Affine)
        assert isinstance(nir_graph.nodes["lif2"], nir.LIF)

    def test_export_recurrent(self, snntorch_recurrent, sample_data):
        nir_graph = export_to_nir(snntorch_recurrent, sample_data)
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output", "0", "1.lif", "1.w_rec", "2", "3"]
        ), nir_graph.nodes.keys()
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1.lif"], nir.CubaLIF)
        assert isinstance(nir_graph.nodes["1.w_rec"], nir.Linear)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)
        assert set(nir_graph.edges) == set(
            [
                ("1.lif", "1.w_rec"),
                ("1.w_rec", "1.lif"),
                ("0", "1.lif"),
                ("3", "output"),
                ("2", "3"),
                ("input", "0"),
                ("1.lif", "2"),
            ]
        )

    def test_import_nir(self):
        graph = nir.read("tests/lif.nir")
        net = import_from_nir(graph)
        out, _ = net(torch.ones(1, 1))
        assert out.shape == (1, 1), out.shape

    def test_import_conv_nir(self):
        graph = nir.read("examples/testconv2d+avgpool.nir")
        net = import_from_nir(graph)
        assert net is not None
        out, _ = net(torch.randn(1, 1, 1, 1))
        assert out.shape == (1, 16, 1, 1), out.shape

    def test_commute_sequential(self, snntorch_sequential, sample_data):
        x = torch.rand((4, 784))
        y_snn, state = snntorch_sequential(x)
        assert y_snn.shape == (4, 10)
        nir_graph = export_to_nir(snntorch_sequential, sample_data)
        net = import_from_nir(nir_graph)
        y_nir, state = net(x)
        assert y_nir.shape == (4, 10), y_nir.shape
        assert torch.allclose(y_snn, y_nir)
