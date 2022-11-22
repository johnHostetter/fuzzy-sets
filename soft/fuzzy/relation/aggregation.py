import torch.nn


class OrderedWeightedAveraging(torch.nn.Module):
    """
    Yager's On Ordered Weighted Averaging Aggregation Operators in Multicriteria Decisionmaking (1988)

    An operator that lies between the 'anding' or the 'oring' of multiple criteria. The weight vector
    allows us to easily adjust the degree of 'anding' and 'oring' implicit in the aggregation.
    """
    def __init__(self, in_features, weight):
        super(OrderedWeightedAveraging, self).__init__()
        self.in_features = in_features
        with torch.no_grad():
            if weight.sum() == 1.0:
                self.weight = torch.nn.parameter.Parameter(torch.abs(weight))
            else:
                raise AttributeError('The weight vector of the Ordered Weighted Averaging operator must sum to 1.0.')

    def forward(self, x):
        """

        Args:
            x: Argument vector, unordered.

        Returns:
            The aggregation of the ordered argument vector with the weight vector.
        """
        ordered_argument_vector = torch.sort(x, descending=True)  # namedtuple with 'values' and 'indices' properties
        return (self.weight * ordered_argument_vector.values).sum()
