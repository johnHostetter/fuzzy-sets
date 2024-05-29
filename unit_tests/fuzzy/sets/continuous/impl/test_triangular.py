import unittest

import torch
import numpy as np

from soft.utilities.reproducibility import set_rng
from soft.fuzzy.sets.continuous.impl import Triangular


def triangular_numpy(element: torch.Tensor, center: np.ndarray, width: np.ndarray):
    """
        Triangular membership function that receives an 'element' value, and uses
        the 'center' and 'width' to determine a degree of membership for 'element'.
        Implemented in Numpy and used in testing.

        https://www.mathworks.com/help/fuzzy/trimf.html

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Triangular fuzzy set.
        width: The width of the Triangular fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    values = 1.0 - (1.0 / width) * np.abs(element - center)
    values[(values < 0)] = 0
    return values


class TestTriangular(unittest.TestCase):
    """
    Test the Triangular fuzzy set (i.e., membership function).
    """

    def test_single_input(self) -> None:
        """
        Test that single input works for the Triangular membership function.

        Returns:
            None
        """
        set_rng(0)
        element = 0.0
        triangular_mf = Triangular(
            centers=np.array([1.5409961]), widths=np.array([0.30742282])
        )
        center = triangular_mf.get_centers().cpu().detach().numpy()
        width = triangular_mf.get_widths().cpu().detach().numpy()
        mu_pytorch = triangular_mf(torch.tensor(element)).degrees.to_dense()
        mu_numpy = triangular_numpy(element, center, width)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(triangular_mf.get_centers().cpu(), torch.tensor(center))
        assert torch.allclose(triangular_mf.get_widths().cpu(), torch.tensor(width))
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(mu_pytorch.cpu().detach().numpy(), mu_numpy, atol=1e-2)

        # test that this is compatible with torch.jit.script
        triangular_mf_scripted = torch.jit.script(triangular_mf)
        assert torch.allclose(
            triangular_mf_scripted(torch.tensor(element)).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input(self) -> None:
        """
        Test that multiple input works for the Triangular membership function.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        triangular_mf = Triangular(
            centers=np.array([1.5410]), widths=np.array([0.3074])
        )
        centers, widths = (
            triangular_mf.get_centers().cpu().detach().numpy(),
            triangular_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = triangular_mf(elements).degrees.to_dense()
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers().cpu(), torch.tensor(centers).float()
        )
        assert torch.allclose(triangular_mf.get_widths().cpu(), torch.tensor(widths).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
        )

        # test that this is compatible with torch.jit.script
        triangular_mf_scripted = torch.jit.script(triangular_mf)
        assert torch.allclose(
            triangular_mf_scripted(elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_centers_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when centers are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        triangular_mf = Triangular(centers=centers, widths=np.array([0.4962566]))
        widths = triangular_mf.get_widths().cpu().detach().numpy()
        mu_pytorch = triangular_mf(elements).degrees.to_dense()
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers().cpu(), torch.tensor(centers).float()
        )
        assert torch.allclose(triangular_mf.get_widths().cpu(), torch.tensor(widths).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
        )

        # test that this is compatible with torch.jit.script
        triangular_mf_scripted = torch.jit.script(triangular_mf)
        assert torch.allclose(
            triangular_mf_scripted(elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_widths_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when widths are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        widths = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        triangular_mf = Triangular(centers=np.array([1.5409961]), widths=widths)
        centers = triangular_mf.get_centers().cpu().detach().numpy()
        mu_pytorch = triangular_mf(elements).degrees.to_dense()
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.get_centers().cpu(), torch.tensor(centers).float()
        )
        assert torch.allclose(triangular_mf.get_widths().cpu(), torch.tensor(widths).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
        )

        # test that this is compatible with torch.jit.script
        triangular_mf_scripted = torch.jit.script(triangular_mf)
        assert torch.allclose(
            triangular_mf_scripted(elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_both_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when centers
        and widths are specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = np.array([-0.5, -0.25, 0.25, 0.5, 0.75])
        widths = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        triangular_mf = Triangular(centers=centers, widths=widths)
        mu_pytorch = triangular_mf(elements).degrees.to_dense()
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert np.allclose(triangular_mf.get_centers().cpu().detach().numpy(), centers)
        assert np.allclose(triangular_mf.get_widths().cpu().detach().numpy(), widths)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
        )

        # test that this is compatible with torch.jit.script
        triangular_mf_scripted = torch.jit.script(triangular_mf)
        assert torch.allclose(
            triangular_mf_scripted(elements).degrees.to_dense(), mu_pytorch
        )

    def test_create_random(self) -> None:
        """
        Test that a random fuzzy set of this type can be created and that the results are consistent
        with the expected membership degrees.

        Returns:
            None
        """
        triangular_mf = Triangular.create(number_of_variables=4, number_of_terms=4)
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = triangular_numpy(
            element,
            triangular_mf.get_centers().cpu().detach().numpy(),
            triangular_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = triangular_mf(torch.tensor(element[0])).degrees.to_dense()
        assert np.allclose(
            mu_pytorch.cpu().detach().numpy(),
            target_membership_degrees,
            rtol=1e-1,
            atol=1e-1,
        )

        # test that this is compatible with torch.jit.script
        triangular_mf_scripted = torch.jit.script(triangular_mf)
        assert torch.allclose(
            triangular_mf_scripted(torch.tensor(element[0])).degrees.to_dense(),
            mu_pytorch,
        )
