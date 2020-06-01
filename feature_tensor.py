import numpy as np
import math
import tensornetwork as tn

class FeatureTensor:
    def __init__(self, input: np.ndarray, feature_map=None):
        """
        Represents a "feature tensor" obtained from applying {feature_map}
        uniformly to every component of the vector {input}.

        :param input: input vector of shape (d, )
        :param feature_map: R -> R^n
        """
        self.input = input
        self.feature_map = feature_map

    @classmethod
    def apply_feature_map(cls, vec, feature_map):
        feature = []
        for i in range(len(vec)):
            feature.append(feature_map(vec[i]))
        return np.array(feature)

    def get_node(self):
        if self.feature_map is not None:
            feature_vector = FeatureTensor.apply_feature_map(self.input, self.feature_map)
        else:
            feature_vector = self.input
        if len(feature_vector.shape) != 1:
            raise ValueError('Feature vector does not produce a vector.')
        return tn.Node(feature_vector, axis_names=['in'])
