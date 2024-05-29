import unittest

import torch
import numpy as np

from soft.utilities.reproducibility import set_rng
from soft.fuzzy.sets.continuous.impl import Gaussian


def gaussian_numpy(element: torch.Tensor, center: np.ndarray, sigma: np.ndarray):
    """
        Gaussian membership function that receives an 'element' value, and uses
        the 'center' and 'sigma' to determine a degree of membership for 'element'.
        Implemented in Numpy and used in testing.

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Gaussian fuzzy set.
        sigma: The width of the Gaussian fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    return np.exp(-1.0 * (np.power(element - center, 2) / (1.0 * np.power(sigma, 2))))


class TestGaussian(unittest.TestCase):
    """
    Test the Gaussian fuzzy set (i.e., membership function).
    """

    def test_single_input(self) -> None:
        """
        Test that single input works for the Gaussian membership function.

        Returns:
            None
        """
        set_rng(0)
        element = torch.zeros(1)
        gaussian_mf = Gaussian(
            centers=np.array([1.5409961]), widths=np.array([0.30742282])
        )
        sigma = gaussian_mf.get_widths().cpu().detach().numpy()
        center = gaussian_mf.get_centers().cpu().detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(element)).degrees.to_dense()
        mu_numpy = gaussian_numpy(element, center, sigma)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.get_widths().cpu(), torch.tensor(sigma))
        assert torch.allclose(gaussian_mf.get_centers().cpu(), torch.tensor(center))
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(mu_pytorch.cpu().detach().numpy(), mu_numpy, rtol=1e-6)

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(element)).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        gaussian_mf = Gaussian(centers=np.array([1.5410]), widths=np.array([0.3074]))
        centers, sigmas = (
            gaussian_mf.get_centers().cpu().detach().numpy(),
            gaussian_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = gaussian_mf(elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.get_widths().cpu(), torch.tensor(sigmas).float())
        assert torch.allclose(gaussian_mf.get_centers().cpu(), torch.tensor(centers).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.allclose(mu_pytorch.cpu(), mu_numpy)

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(elements)).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_centers_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when centers are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        sigmas = np.array([0.4962566, 0.7682218, 0.08847743, 0.13203049, 0.30742282])
        gaussian_mf = Gaussian(centers=centers, widths=sigmas)
        mu_pytorch = gaussian_mf(elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.get_widths().cpu(), torch.tensor(sigmas).float())
        assert torch.allclose(gaussian_mf.get_centers().cpu(), torch.tensor(centers).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-4
        )

        expected_areas = torch.tensor(
            [0.7412324, 1.1474512, 0.13215375, 0.1972067, 0.45918167]
        )
        assert torch.allclose(gaussian_mf.area(), expected_areas)

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(elements)).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_sigmas_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when sigmas are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        sigmas = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(centers=np.array([1.5410]), widths=sigmas)
        mu_pytorch = gaussian_mf(elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(
            elements,
            gaussian_mf.get_centers().cpu().detach().numpy(),
            sigmas,
        )

        # make sure the Gaussian parameters are still identical afterward
        assert np.allclose(gaussian_mf.get_widths().cpu().detach().numpy(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(elements)).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_both_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when centers and
        sigmas are specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = torch.tensor([-0.5, -0.25, 0.25, 0.5, 0.75])
        sigmas = torch.tensor(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(centers=centers, widths=sigmas)
        mu_pytorch = gaussian_mf(elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(
            elements, centers.cpu().detach().numpy(), sigmas.cpu().detach().numpy()
        )

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.get_centers().cpu(), centers)
        assert torch.allclose(gaussian_mf.get_widths().cpu(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(elements)).degrees.to_dense(), mu_pytorch
        )

    def test_multi_centers(self) -> None:
        """
        Test that multidimensional centers work with the Gaussian membership function.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.Tensor(
            [
                [
                    [0.6960, 0.8233, 0.8147],
                    [0.1024, 0.3122, 0.5160],
                    [0.8981, 0.6810, 0.2366],
                ],
                [
                    [0.2447, 0.4218, 0.6146],
                    [0.8887, 0.6273, 0.6697],
                    [0.1439, 0.9383, 0.8101],
                ],
            ]
        )
        centers = torch.tensor(
            [
                [
                    [0.1236, 0.4893, 0.8372],
                    [0.8275, 0.2979, 0.7192],
                    [0.2328, 0.1418, 0.1036],
                ],
                [
                    [0.9651, 0.7622, 0.1544],
                    [0.1274, 0.5798, 0.6425],
                    [0.1518, 0.6554, 0.3799],
                ],
            ]
        )
        sigmas = torch.tensor([0.1, 0.25, 0.5])  # negative widths are missing sets
        gaussian_mf = Gaussian(centers=centers, widths=sigmas)
        mu_pytorch = gaussian_mf(elements.unsqueeze(dim=0)).degrees.to_dense()
        mu_numpy = gaussian_numpy(
            elements, centers.cpu().detach().numpy(), sigmas.cpu().detach().numpy()
        )

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.get_centers().cpu(), centers)
        assert torch.allclose(gaussian_mf.get_widths().cpu(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=0).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(
                torch.tensor(
                    elements.unsqueeze(dim=0)
                )  # add batch dimension (size is 1)
            ).degrees.to_dense(),
            mu_pytorch,
        )

    def test_consistency(self) -> None:
        """
        Test that the results are consistent with the expected membership degrees.

        Returns:
            None
        """
        set_rng(0)
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = torch.tensor(
            [
                [9.99838713e-01, 4.29740733e-01, 2.21413949e-01, 1.05996197e-02],
                [9.99951447e-01, 4.24180250e-01, 2.76996031e-01, 8.40594458e-04],
                [2.34987238e-06, 9.68724670e-01, 5.72716145e-01, 2.14063307e-01],
                [1.86889382e-02, 9.99939174e-01, 4.46363185e-01, 7.51629552e-03]
            ]
        )
        centers = np.array(
            [
                [0.01497397, -1.3607662, 1.0883657, 1.9339248],
                [-0.01367673, 2.3560243, -1.8339163, -3.3379893],
                [-4.489564, -0.01467094, -0.13278057, 0.08638719],
                [0.17008819, 0.01596639, -1.7408595, 2.797653],
            ]
        )
        sigmas = np.array(
            [
                [1.16553577, 1.48497267, 0.91602303, 0.91602303],
                [1.98733806, 2.53987592, 1.58646032, 1.24709336],
                [1.24709336, 0.10437003, 0.12908118, 0.08517358],
                [0.08517358, 1.54283158, 1.89779089, 1.27380911],
            ]
        )

        gaussian_mf = Gaussian(
            centers=centers,
            widths=sigmas,
        )
        mu_pytorch = gaussian_mf(torch.tensor(element[0])).degrees.to_dense()

        # make sure the Gaussian parameters are still identical afterward
        assert np.allclose(
            gaussian_mf.get_centers().cpu().detach().numpy(), centers[: element.shape[1]]
        )
        assert np.allclose(
            gaussian_mf.get_widths().cpu().detach().numpy(), sigmas[: element.shape[1]]
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.allclose(
            mu_pytorch.cpu().float(), target_membership_degrees, rtol=1e-1
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(element)).degrees.to_dense(), mu_pytorch
        )

    def test_create_random(self) -> None:
        """
        Test that a random fuzzy set of this type can be created and that the results are consistent
        with the expected membership degrees.

        Returns:
            None
        """
        gaussian_mf = Gaussian.create(number_of_variables=4, number_of_terms=4)
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = gaussian_numpy(
            element.reshape(4, 1),  # column vector
            gaussian_mf.get_centers().cpu().detach().numpy(),
            gaussian_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = gaussian_mf(torch.tensor(element[0])).degrees.to_dense()
        assert np.allclose(
            mu_pytorch.cpu().detach().numpy(), target_membership_degrees, atol=1e-1
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(torch.tensor(element)).degrees.to_dense(), mu_pytorch
        )
