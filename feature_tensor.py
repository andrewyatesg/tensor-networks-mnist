import numpy as np
import math
import tensornetwork as tn
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence, Tuple

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

    def get_nodes(self) -> List[tn.Node]:
        if self.feature_map is not None:
            feature_vector = FeatureTensor.apply_feature_map(self.input, self.feature_map)
        else:
            feature_vector = self.input
        if len(feature_vector.shape) != 1:
            raise ValueError('Feature vector does not produce a vector.')
        d = feature_vector.shape[0]
        input_tn = []
        # Loop through features
        for i in range(int(d / 2)):
            feature = np.array([feature_vector[2 * i], feature_vector[2 * i + 1]])
            input_tn.append(tn.Node(feature, axis_names=['in']))
        return input_tn
