"""
Test that various continuous fuzzy set implementations are working as intended, such as the
Gaussian fuzzy set (i.e., membership function), and the Triangular fuzzy set (i.e., membership
function).
"""
import os
import unittest
from pathlib import Path
from typing import MutableMapping
from collections import OrderedDict

import numpy as np

import torch

from soft.utilities.reproducibility import set_rng
from soft.fuzzy.sets.continuous.abstract import ContinuousFuzzySet
from soft.fuzzy.sets.continuous.impl import LogisticCurve, Gaussian, Triangular


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


class TestLogistic(unittest.TestCase):
    """
    Test the LogisticCurve class.
    """

    def test_logistic_curve(self) -> None:
        """
        Test the calculation of the logistic curve.

        Returns:
            None
        """
        elements = torch.tensor(
            [
                [-1.1258, -1.1524, -0.2506],
                [-0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.4681],
                [-0.1577, 1.4437, 0.2660],
                [0.1665, 0.8744, -0.1435],
                [-0.1116, 0.9318, 1.2590],
                [2.0050, 0.0537, 0.6181],
                [-0.4128, -0.8411, -2.3160],
            ]
        )
        logistic_curve = LogisticCurve(midpoint=0.5, growth=10, supremum=1)

        self.assertTrue(
            torch.allclose(
                logistic_curve(elements).cpu().float(),
                torch.tensor(
                    [
                        [8.6944e-08, 6.6637e-08, 5.4947e-04],
                        [8.7920e-05, 9.7032e-01, 8.7214e-01],
                        [2.8578e-04, 4.3886e-12, 4.2092e-01],
                        [1.3901e-03, 9.9992e-01, 8.7864e-02],
                        [3.4390e-02, 9.7689e-01, 1.6018e-03],
                        [2.2024e-03, 9.8685e-01, 9.9949e-01],
                        [1.0000e00, 1.1396e-02, 7.6513e-01],
                        [1.0857e-04, 1.4986e-06, 5.8921e-13],
                    ]
                ).float(),
                atol=1e-4,
                rtol=1e-4,
            )
        )


class TestContinuousFuzzySet(unittest.TestCase):
    """
    Test the abstract ContinuousFuzzySet class.
    """

    def test_illegal_attempt_to_create(self) -> None:
        """
        Test that an illegal attempt to create a ContinuousFuzzySet raises an error.

        Returns:
            None
        """
        with self.assertRaises(NotImplementedError):
            ContinuousFuzzySet.create(number_of_variables=4, number_of_terms=2)

    def test_save_and_load(self) -> None:
        for subclass in ContinuousFuzzySet.__subclasses__():
            membership_func = subclass.create(number_of_variables=4, number_of_terms=4)
            state_dict: MutableMapping = membership_func.state_dict()

            # test that the path must be valid
            with self.assertRaises(ValueError):
                membership_func.save(Path(""))
            with self.assertRaises(ValueError):
                membership_func.save(Path("test"))
            with self.assertRaises(ValueError):
                membership_func.save(
                    Path("test.pth")
                )  # this file extension is not supported; see error message to learn why

            # test that saving the state dict works
            saved_state_dict: OrderedDict = membership_func.save(
                Path("membership_func.pt")
            )

            # check that the saved state dict is the same as the original state dict
            for key in state_dict.keys():
                assert key in saved_state_dict and torch.allclose(
                    state_dict[key], saved_state_dict[key]
                )
            # except the saved state dict includes additional information not captured by
            # the original state dict, such as the class name and the labels
            assert "labels" in saved_state_dict.keys()
            assert "class_name" in saved_state_dict.keys() and saved_state_dict[
                "class_name"
            ] in (subclass.__name__ for subclass in ContinuousFuzzySet.__subclasses__())

            loaded_membership_func = ContinuousFuzzySet.load(Path("membership_func.pt"))
            # check that the parameters and members are the same
            assert membership_func == loaded_membership_func
            assert torch.allclose(
                membership_func.centers, loaded_membership_func.centers
            )
            assert torch.allclose(membership_func.widths, loaded_membership_func.widths)
            if (
                type(subclass) == Gaussian
            ):  # Gaussian has an additional parameter (alias for widths)
                assert torch.allclose(
                    membership_func.sigmas, loaded_membership_func.sigmas
                )
            assert membership_func.labels == loaded_membership_func.labels
            # check some functionality that it is still working
            assert torch.allclose(membership_func.area(), loaded_membership_func.area())
            assert torch.allclose(
                membership_func(torch.tensor([[0.1, 0.2, 0.3, 0.4]])).degrees,
                loaded_membership_func(torch.tensor([[0.1, 0.2, 0.3, 0.4]])).degrees,
            )
            # delete the file
            os.remove("membership_func.pt")


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
        gaussian_mf = Gaussian(centers=[1.5409961], widths=[0.30742282])
        sigma = gaussian_mf.sigmas.cpu().detach().numpy()
        center = gaussian_mf.centers.cpu().detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(element)).degrees
        mu_numpy = gaussian_numpy(element, center, sigma)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.sigmas.cpu(), torch.tensor(sigma)).all()
        assert torch.isclose(gaussian_mf.centers.cpu(), torch.tensor(center)).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.cpu().detach().numpy(), mu_numpy, rtol=1e-6).all()

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
        gaussian_mf = Gaussian(centers=[1.5410], widths=[0.3074])
        centers, sigmas = (
            gaussian_mf.centers.cpu().detach().numpy(),
            gaussian_mf.sigmas.cpu().detach().numpy(),
        )
        mu_pytorch = gaussian_mf(elements).degrees
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(
            gaussian_mf.sigmas.cpu(), torch.tensor(sigmas).float()
        ).all()
        assert torch.isclose(
            gaussian_mf.centers.cpu(), torch.tensor(centers).float()
        ).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.isclose(mu_pytorch.cpu(), mu_numpy).all()

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
        mu_pytorch = gaussian_mf(elements).degrees
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.sigmas.cpu(), torch.tensor(sigmas).float())
        assert torch.allclose(gaussian_mf.centers.cpu(), torch.tensor(centers).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-4
        )

        expected_areas = torch.tensor(
            [0.7412324, 1.1474512, 0.13215375, 0.1972067, 0.45918167]
        )
        assert torch.allclose(gaussian_mf.area(), expected_areas)

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
        sigmas = torch.tensor(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(centers=[1.5410], widths=sigmas)
        mu_pytorch = gaussian_mf(elements).degrees
        mu_numpy = gaussian_numpy(
            elements,
            gaussian_mf.centers.cpu().detach().numpy(),
            sigmas.cpu().detach().numpy(),
        )

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.widths.cpu(), sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

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
        mu_pytorch = gaussian_mf(elements).degrees
        mu_numpy = gaussian_numpy(
            elements, centers.cpu().detach().numpy(), sigmas.cpu().detach().numpy()
        )

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.centers.cpu(), centers).all()
        assert torch.isclose(gaussian_mf.widths.cpu(), sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

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
        mu_pytorch = gaussian_mf(elements.unsqueeze(dim=0)).degrees
        mu_numpy = gaussian_numpy(
            elements, centers.cpu().detach().numpy(), sigmas.cpu().detach().numpy()
        )

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.centers.cpu(), centers).all()
        assert torch.isclose(gaussian_mf.widths.cpu(), sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=0).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

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
                [9.9984e-01, 4.3174e-01, 2.4384e-01, 1.1603e-02],
                [9.9992e-01, 4.2418e-01, 2.6132e-01, 7.6078e-04],
                [2.9000e-06, 9.5753e-01, 5.7272e-01, 1.2510e-01],
                [7.1018e-03, 9.9948e-01, 4.3918e-01, 7.5163e-03],
            ]
        )
        centers = torch.tensor(
            [
                [0.01497397, -1.3607662, 1.0883657, 1.9339248],
                [-0.01367673, 2.3560243, -1.8339163, -3.3379893],
                [-4.489564, -0.01467094, -0.13278057, 0.08638719],
                [0.17008819, 0.01596639, -1.7408595, 2.797653],
            ]
        )
        sigmas = torch.tensor(
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
        mu_pytorch = gaussian_mf(torch.tensor(element[0])).degrees

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(gaussian_mf.centers.cpu(), centers[: element.shape[1]])
        assert torch.allclose(gaussian_mf.widths.cpu(), sigmas[: element.shape[1]])
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.allclose(
            mu_pytorch.cpu().float(), target_membership_degrees, rtol=1e-1
        )

    def test_create_random(self) -> None:
        """
        Test that a random fuzzy set of this type can be created and that the results are consistent
        with the expected membership degrees.

        Returns:
            None
        """
        gaussian = Gaussian.create(number_of_variables=4, number_of_terms=4)
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = gaussian_numpy(
            element.reshape(4, 1),  # column vector
            gaussian.centers.cpu().detach().numpy(),
            gaussian.widths.cpu().detach().numpy(),
        )
        mu_pytorch = gaussian(torch.tensor(element[0])).degrees
        assert np.allclose(
            mu_pytorch.cpu().detach().numpy(), target_membership_degrees, atol=1e-1
        )


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
        triangular_mf = Triangular(centers=[1.5409961], widths=[0.30742282])
        center = triangular_mf.centers.cpu().detach().numpy()
        width = triangular_mf.widths.cpu().detach().numpy()
        mu_pytorch = triangular_mf(torch.tensor(element)).degrees
        mu_numpy = triangular_numpy(element, center, width)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(triangular_mf.centers.cpu(), torch.tensor(center))
        assert torch.allclose(triangular_mf.widths.cpu(), torch.tensor(width))
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(mu_pytorch.cpu().detach().numpy(), mu_numpy, atol=1e-2)

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
        triangular_mf = Triangular(centers=[1.5410], widths=[0.3074])
        centers, widths = (
            triangular_mf.centers.cpu().detach().numpy(),
            triangular_mf.widths.cpu().detach().numpy(),
        )
        mu_pytorch = triangular_mf(elements).degrees
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.centers.cpu(), torch.tensor(centers).float()
        )
        assert torch.allclose(triangular_mf.widths.cpu(), torch.tensor(widths).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
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
        triangular_mf = Triangular(centers=centers, widths=[0.4962566])
        widths = triangular_mf.widths.cpu().detach().numpy()
        mu_pytorch = triangular_mf(elements).degrees
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.centers.cpu(), torch.tensor(centers).float()
        )
        assert torch.allclose(triangular_mf.widths.cpu(), torch.tensor(widths).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
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
        triangular_mf = Triangular(centers=[1.5409961], widths=widths)
        centers = triangular_mf.centers.cpu().detach().numpy()
        mu_pytorch = triangular_mf(elements).degrees
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert torch.allclose(
            triangular_mf.centers.cpu(), torch.tensor(centers).float()
        )
        assert torch.allclose(triangular_mf.widths.cpu(), torch.tensor(widths).float())
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
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
        mu_pytorch = triangular_mf(elements).degrees
        mu_numpy = triangular_numpy(elements.cpu().detach().numpy(), centers, widths)

        # make sure the Triangular parameters are still identical afterward
        assert np.allclose(triangular_mf.centers.cpu().detach().numpy(), centers)
        assert np.allclose(triangular_mf.widths.cpu().detach().numpy(), widths)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, atol=1e-2
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
            triangular_mf.centers.cpu().detach().numpy(),
            triangular_mf.widths.cpu().detach().numpy(),
        )
        mu_pytorch = triangular_mf(torch.tensor(element[0])).degrees
        assert np.isclose(
            mu_pytorch.cpu().detach().numpy(),
            target_membership_degrees,
            rtol=1e-1,
            atol=1e-1,
        ).all()
