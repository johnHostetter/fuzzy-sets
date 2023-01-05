import torch
import sympy
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt


class Base(torch.nn.Module):
    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None,
                 trainable=True, sort_by='centers'):
        super(Base, self).__init__()
        self.in_features = in_features
        self._log_widths = None
        self.sort_by = sort_by

        # initialize centers
        if centers is None:
            self.centers = torch.nn.parameter.Parameter(torch.randn(self.in_features))
        else:
            centers = self.convert_to_tensor(centers)
            self.centers = torch.nn.parameter.Parameter(centers)

        # initialize widths -- never adjust the widths directly, use the logarithm of them to avoid negatives
        if widths is None:  # we apply the logarithm to the widths, so later, if we train them, and they become
            # nonzero, with an exponential function they are still positive
            # in other words, since gradient descent may make the widths negative, we nullify that effect
            self.widths = torch.rand(self.in_features)
        else:
            # we assume the widths are given to us are within (0, 1)
            widths = self.convert_to_tensor(widths)
            self.widths = torch.nn.parameter.Parameter(torch.abs(widths))

        self.log_widths()  # update the stored log widths

        # initialize support
        if supports is None:
            self.supports = torch.ones(self.in_features)
        else:
            supports = self.convert_to_tensor(supports)
            self.supports = torch.abs(supports)

        # used for feature selection/reduction
        self.special_idx = None

        self.labels = labels
        self.trainable = self.train(mode=trainable)
        if sort_by == 'centers':
            self.sort()

    def log_widths(self):
        with torch.no_grad():
            self._log_widths = torch.nn.parameter.Parameter(torch.log(self.widths))
        return self._log_widths

    @staticmethod
    def convert_to_tensor(values):
        if isinstance(values, torch.Tensor):
            return values
        else:
            return torch.tensor(np.array(values)).float()

    def train(self, mode):
        """
        Disable/enable training for the granules' parameters.
        """
        self.centers.requires_grad = mode
        self._log_widths.requiresGrad = mode
        if not mode:
            self.centers.grad = None
            self._log_widths.grad = None
        return mode

    def sort(self):
        pass
        # with torch.no_grad():
        #     # sorting according to centers
        #     if self.centers.nelement() > 1:
        #         sorted_centers, indices = torch.sort(self.centers)
        #         sorted_widths = self.widths.gather(0, indices.argsort())
        #         sorted_supports = self.supports.gather(0, indices.argsort())
        #         self.centers = torch.nn.Parameter(sorted_centers)
        #         self.widths = sorted_widths
        #         self.supports = sorted_supports
        # self.log_widths()  # update the stored log widths
        # self.train(self.trainable)

    def reshape_parameters(self):
        if self.centers.nelement() == 1:
            self.centers = torch.nn.Parameter(self.centers.reshape(1))
        if self.widths.nelement() == 1:
            self.widths = torch.nn.Parameter(self.widths.reshape(1))
        if self.supports.nelement() == 1:
            self.supports = self.supports.reshape(1)
        self.log_widths()  # update the stored log widths
        self.train(self.trainable)

    def extend(self, centers, widths, supports=None):
        with torch.no_grad():
            self.in_features += len(centers)
            self.reshape_parameters()
            if not isinstance(centers, torch.Tensor):
                centers = torch.tensor(np.array(centers))
            self.centers = torch.nn.Parameter(torch.cat([self.centers, centers]))
            if not isinstance(widths, torch.Tensor):
                widths = torch.tensor(widths)
            self.widths = torch.nn.Parameter(torch.cat([self.widths, widths]))
            if supports is None:
                self.supports = torch.cat([self.supports, torch.ones(len(centers))])
            else:
                if not isinstance(supports, torch.Tensor):
                    supports = torch.tensor(supports)
                self.supports = torch.cat([self.supports, supports])
        self.log_widths()  # update the stored log widths
        self.sort()

    def hstack(self, centers, widths, supports=None):
        with torch.no_grad():
            self.in_features += len(centers)
            self.reshape_parameters()
            if not isinstance(centers, torch.Tensor):
                centers = torch.tensor(np.array(centers))
            self.centers = torch.nn.Parameter(torch.hstack([self.centers, centers]))
            if not isinstance(widths, torch.Tensor):
                widths = torch.tensor(widths)
            self.widths = torch.nn.Parameter(torch.hstack([self.widths, widths]))
            if supports is None:
                self.supports = torch.hstack([self.supports, torch.ones(len(centers))])
            else:
                if not isinstance(supports, torch.Tensor):
                    supports = torch.tensor(supports)
                self.supports = torch.hstack([self.supports, supports])
        self.log_widths()  # update the stored log widths
        self.sort()

    def make_dont_care_membership(self):
        """
        Create a 'don't care' membership function for each linguistic variable. Useful for feature selection/reduction.

        Returns:
            (int) column index of the 'don't care' membership function.
        """
        if self.special_idx is None:
            size = np.array(self.centers.shape)
            self.hstack(centers=torch.tensor([torch.nan] * size[0]).unsqueeze(dim=1),
                        widths=torch.tensor([torch.nan] * size[0]).unsqueeze(dim=1),
                        supports=torch.tensor([torch.nan]))
            self.special_idx = size[1]
            self.log_widths()  # update the stored log widths
            self.sort()
        return self.special_idx

    def increase_support_of(self, index):
        """
        The 'index' refers to the index of a Gaussian fuzzy set on this dimension.

        We want to increase the support or the count of this fuzzy set as the number of
        data points increase that shows the fuzzy set located at 'index' is satisfactory for representing them.

        Args:
            index:

        Returns:

        """
        values = torch.zeros(self.supports.shape)
        values[index] = 1
        with torch.no_grad():
            self.supports = torch.add(self.supports, values)

    def forward(self):
        raise NotImplementedError('The Base Fuzzy Set has no membership function defined.')


class Gaussian(Base):
    """
    Implementation of the Gaussian membership function.
    """

    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None,
                 trainable=True, sort_by='centers'):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - sigmas: trainable parameter
            centers and sigmas are initialized randomly by default,
            but sigmas must be > 0
        """
        super(Gaussian, self).__init__(in_features, centers, widths, supports, labels, trainable, sort_by)

    @property
    def sigmas(self):
        return self.widths

    @sigmas.setter
    def sigmas(self, sigmas):
        self.widths = sigmas

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        log_results = torch.exp(
            -1.0 * (torch.pow(x.unsqueeze(dim=-1) - self.centers, 2) / torch.pow(torch.exp(self._log_widths), 2)))
        # no_log_results = torch.exp(
        #     -1.0 * (torch.pow(x.unsqueeze(dim=-1) - self.centers, 2) / torch.pow(self.widths, 2)))
        return log_results
        # return torch.exp(-1.0 * (torch.pow(x - self.centers, 2) / torch.pow(torch.exp(self._log_widths), 2)))


class Triangular(Base):
    """
    Implementation of the Triangular membership function.
    """

    def __init__(self, in_features, centers=None, widths=None, supports=None, labels=None,
                 trainable=True, sort_by='centers'):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - centers: trainable parameter
            - widths: trainable parameter
            centers and widths are initialized randomly by default,
            but widths must be > 0
        """
        super(Triangular, self).__init__(in_features, centers, widths, supports, labels, trainable, sort_by)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        # https://stackoverflow.com/questions/65022269/how-to-use-a-learnable-parameter-in-pytorch-constrained-between-0-and-1

        return torch.max(1.0 - (1.0 / torch.exp(self._log_widths)) * torch.abs(x.unsqueeze(dim=-1) - self.centers),
                         torch.tensor(0.0))


# https://docs.sympy.org/latest/modules/integrals/integrals.html
# https://docs.sympy.org/latest/modules/sets.html
# https://numpydoc.readthedocs.io/en/latest/example.html


class DiscreteFuzzySet:
    """
    A parent class for all fuzzy sets to inherit. Allows the user to visualize the fuzzy set.
    """

    def graph(self, lower=0, upper=100, samples=100):
        """
        Graphs the fuzzy set in the universe of elements.

        Parameters
        ----------
        lower : 'float', optional
            Default value is 0. Specifies the infimum x value for the graph.
        upper : 'float', optional
            Default value is 100. Specifies the supremum x value for the graph.
        samples : 'int', optional
            Default value is 100. Specifies the number of x values to test in the domain
            to approximate the graph. A higher sample value will yield a higher resolution
            of the graph, but large values will lead to performance issues.
        """
        x_list = np.linspace(lower, upper, samples)
        y_list = []
        for x in x_list:
            y_list.append(self.degree(x))
        if self.name is not None:
            plt.title('%s Fuzzy Set' % self.name)
        else:
            plt.title('Unnamed Fuzzy Set')

        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.plot(x_list, y_list, color='grey', label='mu')
        plt.legend()
        plt.show()


class OrdinaryDiscreteFuzzySet(DiscreteFuzzySet):
    """
    An ordinary fuzzy set that is of type 1 and level 1.
    """

    def __init__(self, formulas, name=None):
        """
        Parameters
        ----------
        formulas : 'list'
            A list of 2-tuples. The first element in the tuple at index 0 is the formula
            equal to f(x) and the second element in the tuple at index 1 is the Interval
            where the formula in the tuple is valid.

            Warning: Formulas should be organized in the list such that the formulas and
            their corresponding intervals are specified from the smallest possible x values
            to the largest possible x values.

            The list of formulas provided constitutes the piece-wise function of the
            fuzzy set's membership function.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        DiscreteFuzzySet.__init__(self)
        self.formulas = formulas
        self.name = name

    def fetch(self, x):
        """
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        formula : 'tuple'/'None'
            Returns the tuple containing the formula and corresponding Interval. Returns
            None if a formula for the element x could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(x):  # check the formula's interval to see if it contains x
                return formula
        return None

    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        formula = self.fetch(x)[0]
        try:
            y = float(formula.subs(Symbol('x'), x))
        except AttributeError:
            y = formula
        return y

    def height(self):
        """
        Calculates the height of the fuzzy set.

        Returns
        -------
        height : 'float'
            The height, or supremum, of the fuzzy set.
        """
        heights = []
        for formula in self.formulas:
            if isinstance(formula[0], sympy.Expr):
                inf_x = formula[1].inf
                sup_x = formula[1].sup
                if formula[1].left_open:
                    inf_x += 1e-8
                if formula[1].right_open:
                    sup_x -= 1e-8
                inf_y = formula[0].subs(Symbol('x'), inf_x)
                sup_y = formula[0].subs(Symbol('x'), sup_x)
                heights.append(inf_y)
                heights.append(sup_y)
            else:
                heights.append(formula[0])
        return max(heights)


class FuzzyVariable(DiscreteFuzzySet):
    """
    A fuzzy variable, or linguistic variable, that contains fuzzy sets.
    """

    def __init__(self, fuzzySets, name=None):
        """
        Parameters
        ----------
        fuzzySets : 'list'
            A list of elements each of type OrdinaryFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        DiscreteFuzzySet.__init__(self)
        self.fuzzySets = fuzzySets
        self.name = name

    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.

        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.

        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        degrees = []
        for fuzzySet in self.fuzzySets:
            degrees.append(fuzzySet.degree(x))
        return tuple(degrees)

    def graph(self, lower=0, upper=100, samples=100):
        """
        Graphs the fuzzy set in the universe of elements.

        Parameters
        ----------
        lower : 'float', optional
            Default value is 0. Specifies the infimum x value for the graph.
        upper : 'float', optional
            Default value is 100. Specifies the supremum x value for the graph.
        samples : 'int', optional
            Default value is 100. Specifies the number of x values to test in the domain
            to approximate the graph. A higher sample value will yield a higher resolution
            of the graph, but large values will lead to performance issues.
        """
        for fuzzySet in self.fuzzySets:
            x_list = np.linspace(lower, upper, samples)
            y_list = []
            for x in x_list:
                y_list.append(fuzzySet.degree(x))
            plt.plot(x_list, y_list, color=np.random.rand(3, ), label=fuzzySet.name)

        if self.name is not None:
            plt.title('%s Fuzzy Variable' % self.name)
        else:
            plt.title('Unnamed Fuzzy Variable')

        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.legend()
        plt.show()
