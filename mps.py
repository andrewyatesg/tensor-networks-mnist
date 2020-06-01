import numpy as np
import math
import tensornetwork as tn
from tensornetwork.network_components import Node, BaseNode, Edge
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence, Tuple

class MPS:
    def __init__(self, leftmost_node: np.ndarray) -> None:
        if len(leftmost_node.shape) != 3:
            raise ValueError('Matrix does not have 3 axes (r, l, out).')
        self.tensors = [tn.Node(leftmost_node, axis_names=['r', 'in', 'out'])]
        self.output_idx = 0

    @classmethod
    def perturb_matrix(cls, matrix, std) -> np.ndarray:
        shape = matrix.shape
        return matrix + np.random.normal(0.0, std, size=shape)

    @classmethod
    def random(cls, bond_dim, physical_dim, output_dim, length, std=1e-3):
        """
        Creates a MPS as a list of tensornetwork Node objects.

        For convenience, each node in the network will have shape (left, right, input, output)
        If a node doesn't have a left or right left (i.e. the endpoint nodes) then these
        dimensions will be omitted
        :param physical_dim: input dimension of each node (usually 2)
        :param bond_dim: bond dimension of MPS, for now each will be the same
        :param output_dim: output dimension of a single node in the MPS (usually 10)
        :param length: number of nodes
        :return: list of tensornetwork nodes in the proper format of [1]
        """
        # Initialize the first node with no left leg
        boundary_left = np.zeros((bond_dim, physical_dim, output_dim))
        boundary_left[0, :, :] = 1
        mps = cls(MPS.perturb_matrix(boundary_left, std))

        # Add each non-endpoint node to the MPS
        for i in range(length - 2):
            mps_matrix = np.array(physical_dim * [np.eye(bond_dim)]).T
            mps.add_node(MPS.perturb_matrix(mps_matrix, std))

        # Finally add the right-most node
        boundary_right = np.zeros((bond_dim, physical_dim))
        boundary_right[0, :] = 1
        mps.add_node(MPS.perturb_matrix(boundary_right, std))
        return mps

    @classmethod
    def connect_mps_tensors(cls, tensors: List[Node]) -> None:
        # Fully connect mps
        for i in range(len(tensors) - 1):
            tensors[i]['r'] ^ tensors[i + 1]['l']

    def add_node(self, matrix: np.ndarray) -> None:
        bond_dim = matrix.shape[0]
        if len(matrix.shape) != 3 and len(matrix.shape) != 2:
            raise ValueError('Matrix does not have 2 (or 3) axes.')

        if len(self.tensors) > 0 and self.tensors[-1]['r'].dimension != bond_dim:
            raise ValueError('The left-most node has right leg dimension %d != %d bond_dim'
                             .format(self.tensors[-1]['r'].dimension, bond_dim))
        elif len(self.tensors) == 0:
            raise ValueError('Must have left-most boundary node before adding middle node.')

        if len(matrix.shape) == 3:
            self.tensors.append(tn.Node(matrix, axis_names=['l', 'r', 'in']))
        elif len(matrix.shape) == 2:
            self.tensors.append(tn.Node(matrix, axis_names=['l', 'in']))

    def form_bond_tensor(self) -> List[Union[Node, BaseNode]]:
        """
        Forms the bond tensor and partitions the MPS around the bond tensor, returning
        the bond tensor.

        :return: bond tensor
        """
        # First, select two nodes where one has the output leg
        output_node = self.tensors[self.output_idx].copy()
        node2 = self.tensors[self.output_idx + 1].copy()

        # Connect right leg of {output_node} with left leg of {node2}
        output_node['r'] ^ node2['l']

        return [output_node, node2]

    def get_bond_axis_names_edge_order(self, bond) -> Tuple[List[Edge], List[str]]:
        output_node = bond[0]
        node2 = bond[1]
        in1_edge = output_node.get_edge('in')
        in2_edge = node2.get_edge('in')
        out_edge = output_node.get_edge('out')
        edge_order = [in1_edge, in2_edge, out_edge]
        axis_lbs = ['in1', 'in2', 'out']

        if node2.axis_names.count('r') > 0:
            r_edge = node2['r']
            edge_order.insert(0, r_edge)
            axis_lbs.insert(0, 'r')

        if output_node.axis_names.count('l') > 0:
            l_edge = output_node['l']
            edge_order.insert(0, l_edge)
            axis_lbs.insert(0, 'l')

        return edge_order, axis_lbs

    def get_contracted_bond(self) -> Union[BaseNode, Node]:
        bond = self.form_bond_tensor()
        edge_order, axis_lbs = self.get_bond_axis_names_edge_order(bond)
        bond = tn.contractors.auto(bond, output_edge_order=edge_order)
        bond.add_axis_names(axis_lbs)
        return bond

    def get_right(self):
        return tn.replicate_nodes(self.tensors[self.output_idx + 2:])

    def update_bond(self, new_bond: Node, max_singular_values: int) -> None:
        """
        Updates the bond tensor of this MPS with {new_bond} by performing a SVD on new_bond,
        keeping the {max_singular_values} largest singular values of {new_bond}.

        Concretely, this method will replace the nodes located at indices {output_idx} and
        {output_idx + 1} with nodes US^(1/2) and S^(1/2)V where {new_bond = USV} is the singular
        value decomposition of {new_bond}.

        :param new_bond: new_bond to replace in
        :param max_singular_values: number of largest singular values to keep
        :return:
        """
        normalization = tn.Node(1. / math.sqrt(np.sum(new_bond.tensor ** 2)))
        new_bond = normalization * new_bond

        left_edges = []
        if new_bond.axis_names.count('l') > 0:
            left_edges.append(new_bond['l'])
        left_edges.append(new_bond['in1'])

        right_edges = [new_bond['in2']]
        if new_bond.axis_names.count('r') > 0:
            right_edges.append(new_bond['r'])
        right_edges.append(new_bond['out'])

        left, right, sings = tn.split_node(new_bond, left_edges=left_edges, right_edges=right_edges,
                                           max_singular_values=max_singular_values, edge_name='connect')
        left['connect'].disconnect()
        if len(left.get_all_dangling()) < 3:
            left.add_axis_names(['in', 'r'])
        else:
            left.add_axis_names(['l', 'in', 'r'])

        if len(right.get_all_dangling()) < 4:
            right.add_axis_names(['l', 'in', 'out'])
        else:
            right.add_axis_names(['l', 'in', 'r', 'out'])

        self.tensors[self.output_idx] = left
        self.tensors[self.output_idx + 1] = right
        self.output_idx = self.output_idx + 1
