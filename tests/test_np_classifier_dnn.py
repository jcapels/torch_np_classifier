import pytest
import torch
from torch_np_classifier.models.np_classifier import NPClassifierDNN


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def input_tensor(batch_size):
    return torch.randn(batch_size, 6144)


class TestNPClassifierDNNDefault:
    def test_output_shape(self, input_tensor, batch_size):
        model = NPClassifierDNN(num_categories=730, original=False)
        out = model([input_tensor])
        assert out.shape == (batch_size, 730)

    def test_output_range(self, input_tensor):
        model = NPClassifierDNN(num_categories=730, original=False)
        out = model([input_tensor])
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_return_embedding(self, input_tensor, batch_size):
        model = NPClassifierDNN(num_categories=730, original=False)
        out, embedding = model([input_tensor], return_embedding=True)
        assert out.shape == (batch_size, 730)
        assert embedding.shape == (batch_size, 1536)

    def test_custom_num_categories(self, input_tensor, batch_size):
        model = NPClassifierDNN(num_categories=10, original=False)
        out = model([input_tensor])
        assert out.shape == (batch_size, 10)

    def test_no_nan_in_output(self, input_tensor):
        model = NPClassifierDNN(original=False)
        out = model([input_tensor])
        assert not torch.isnan(out).any()


class TestNPClassifierDNNOriginal:
    def test_output_shape(self, input_tensor, batch_size):
        model = NPClassifierDNN(num_categories=730, original=True)
        out = model([input_tensor])
        assert out.shape == (batch_size, 730)

    def test_output_range(self, input_tensor):
        model = NPClassifierDNN(num_categories=730, original=True)
        out = model([input_tensor])
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_return_embedding(self, input_tensor, batch_size):
        model = NPClassifierDNN(num_categories=730, original=True)
        out, embedding = model([input_tensor], return_embedding=True)
        assert out.shape == (batch_size, 730)
        assert embedding.shape == (batch_size, 1152)

    def test_no_nan_in_output(self, input_tensor):
        model = NPClassifierDNN(original=True)
        out = model([input_tensor])
        assert not torch.isnan(out).any()


class TestNPClassifierDNNTraining:
    def test_gradients_flow(self, input_tensor):
        model = NPClassifierDNN(num_categories=730, original=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        target = torch.randint(0, 2, (input_tensor.shape[0], 730)).float()

        out = model([input_tensor])
        loss = torch.nn.BCELoss()(out, target)
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_eval_mode_single_sample(self):
        model = NPClassifierDNN(num_categories=730, original=False)
        model.eval()
        x = torch.randn(1, 6144)
        with torch.no_grad():
            out = model([x])
        assert out.shape == (1, 730)
