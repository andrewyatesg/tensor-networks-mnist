from mps import MPS
from feature_tensor import FeatureTensor
import numpy as np
import math
import tensornetwork as tn
from tensornetwork.network_components import Node, BaseNode
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence


class Sweeper:
    """
    Represents the sweeping algorithm. It should be reinstantiated
    per epoch.
    """
    def __init__(self, mps: MPS, Xs_batch: np.ndarray, Ys_batch: np.ndarray, feature_map=None):
        self.mps = mps
        self.Xs = Xs_batch
        self.Ys = Ys_batch
        # Left-most projection node for each vector in Xs_batch
        # This is so we don't have to recompute it after every translation
        self.projections : List[Node] = []
        self.feature_map = feature_map

    def update_projection(self, input_idx: int, projection: Node):
        self.projections[input_idx] = projection

    def __projection__(self, feature: FeatureTensor) -> List[Union[Node, BaseNode]]:
        # todo

    def __gradient__(self, input_idx: int) -> np.ndarray:
        """
        Calculates the gradient of the inner-product <MPS|input_tensor>
        with respect to the bond tensor.

        :param input_tensor: tensor formed from the input vector
        :return: d<MPS|input_tensor>/d{bond}
        """
        feature = FeatureTensor(self.Xs[input_idx])
        proj = self.__projection__(feature)
        bond = self.mps.get_contracted_bond()

        leftmost = False
        if proj[0] is None:
            projection1 = tn.replicate_nodes(proj[1:])
            projection2 = tn.replicate_nodes(proj[1:])
            leftmost = True
        else:
            projection1 = tn.replicate_nodes(proj)
            projection2 = tn.replicate_nodes(proj)

        if leftmost:
            projection1[0]['in'] ^ bond['in1']
            projection1[1]['in'] ^ bond['in2']
            projection1[2]['in'] ^ bond['r']
        else:
            projection1[0]['in'] ^ bond['l']
            projection1[1]['in'] ^ bond['in1']
            projection1[2]['in'] ^ bond['in2']
            projection1[3]['in'] ^ bond['r']
        out_edge = bond['out']
        mps_prod = tn.contractors.auto(projection1 + [bond], output_edge_order=[out_edge]) - tn.Node(self.Ys[input_idx])
        mps_prod.add_axis_names(['out'])

        if leftmost:
            in1_edge = projection2[0]['in']
            in2_edge = projection2[1]['in']
            r_edge = projection2[2]['in']
            edge_order = [r_edge, in1_edge, in2_edge]
        else:
            l_edge = projection2[0]['in']
            in1_edge = projection2[1]['in']
            in2_edge = projection2[2]['in']
            r_edge = projection2[3]['in']
            edge_order = [l_edge, r_edge, in1_edge, in2_edge]

        edge_order.append(mps_prod['out'])
        return tn.contractors.auto([mps_prod] + projection2, output_edge_order=edge_order).tensor