from mps import MPS
from feature_tensor import FeatureTensor
import numpy as np
import math
import tensornetwork as tn
from tensornetwork.network_components import Node, BaseNode
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import threading


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
        self.projections : List[Node] = Xs_batch.shape[1] * [None]
        self.feature_map = feature_map

    def __leftmost__(self):
        return self.mps.output_idx == 0

    def __contract_with_right(self, right: List[Node], input_tensor: List[Node]) -> Union[BaseNode, Node]:
        MPS.connect_mps_tensors(right)
        offset = self.mps.output_idx + 2
        for i in range(len(right)):
            right[i]['in'] ^ input_tensor[offset + i]['in']
        contracted_right = tn.contractors.auto(right + input_tensor[offset:])
        contracted_right.add_axis_names(['in'])
        return contracted_right

    def __projection__(self, feature: FeatureTensor) -> List[Union[Node, BaseNode]]:
        input_tensor = feature.get_nodes()
        leftmost = self.__leftmost__()
        right = self.mps.get_right()
        if leftmost:
            node2 = input_tensor[0]
            node3 = input_tensor[1]
            node4 = self.__contract_with_right(right, input_tensor)
            return [node2, node3, node4]
        else:
            node1 = self.projections[feature.idx]
            node2 = input_tensor[self.mps.output_idx]
            node3 = input_tensor[self.mps.output_idx + 1]
            node4 = self.__contract_with_right(right, input_tensor)
            return [node1, node2, node3, node4]

    def __gradient__(self, input_idx: int) -> np.ndarray:
        """
        Calculates the gradient of the inner-product <MPS|input_tensor>
        with respect to the bond tensor.

        :param input_tensor: tensor formed from the input vector
        :return: d<MPS|input_tensor>/d{bond}
        """
        feature = FeatureTensor(self.Xs[:, input_idx], input_idx)
        proj = self.__projection__(feature)
        bond = self.mps.get_contracted_bond()

        leftmost = self.__leftmost__()

        if leftmost:
            proj[0]['in'] ^ bond['in1']
            proj[1]['in'] ^ bond['in2']
            proj[2]['in'] ^ bond['r']
        else:
            proj[0]['in'] ^ bond['l']
            proj[1]['in'] ^ bond['in1']
            proj[2]['in'] ^ bond['in2']
            proj[3]['in'] ^ bond['r']

        mps_prod = tn.contractors.auto(proj + [bond]) - tn.Node(self.Ys[:, input_idx])
        mps_prod.add_axis_names(['out'])
        proj2 = tn.replicate_nodes(proj)

        if leftmost:
            in1_edge = proj2[0]['in']
            in2_edge = proj2[1]['in']
            r_edge = proj2[2]['in']
            edge_order = [r_edge, in1_edge, in2_edge]
        else:
            l_edge = proj2[0]['in']
            in1_edge = proj2[1]['in']
            in2_edge = proj2[2]['in']
            r_edge = proj2[3]['in']
            edge_order = [l_edge, r_edge, in1_edge, in2_edge]

        edge_order.append(mps_prod['out'])
        return tn.contractors.auto([mps_prod] + proj2, output_edge_order=edge_order).tensor

    def __precompute_projections__(self, input_idx: int):
        # We have this redefinition bc this method is called after we update the MPS
        # and translate its output label to the right
        feature = FeatureTensor(self.Xs[:, input_idx], input_idx)
        output_idx = self.mps.output_idx - 1
        leftmost = output_idx == 0
        input_tensor = feature.get_nodes()
        if leftmost:
            node2_cpy = input_tensor[0].copy()
            above_node2 = self.mps.tensors[0].copy()
            node2_cpy['in'] ^ above_node2['in']
            precomputed_proj = tn.contractors.auto([node2_cpy, above_node2])
            precomputed_proj.add_axis_names(['in'])
            self.projections[feature.idx] = precomputed_proj
        else:
            node1_cpy = self.projections[feature.idx]
            node2_cpy = input_tensor[output_idx]
            above_node2 = self.mps.tensors[output_idx].copy()
            node1_cpy['in'] ^ above_node2['l']
            node2_cpy['in'] ^ above_node2['in']
            precomputed_proj = tn.contractors.auto([node1_cpy, node2_cpy, above_node2])
            precomputed_proj.add_axis_names(['in'])
            self.projections[feature.idx] = precomputed_proj

    def __add_newbond_names__(self, bond_tensor):
        output_idx = self.mps.output_idx
        if output_idx == 0:
            bond_tensor.add_axis_names(['r', 'in1', 'in2', 'out'])
        elif output_idx == len(self.mps.tensors) - 2:
            bond_tensor.add_axis_names(['l', 'in1', 'in2', 'out'])
        else:
            bond_tensor.add_axis_names(['l', 'r', 'in1', 'in2', 'out'])

    def translate(self, alpha: float, max_singular_values: int, num_threads: int) -> bool:
        B = self.Xs.shape[1]
        num_per_thd = int(B / num_threads)
        bond = self.mps.get_contracted_bond()
        bond_shape = bond.shape
        leftmost = self.__leftmost__()

        if leftmost:
            grads = np.zeros((num_threads, bond_shape[0], bond_shape[1], bond_shape[2], bond_shape[3]))
        else:
            grads = np.zeros((num_threads, bond_shape[0], bond_shape[1], bond_shape[2], bond_shape[3], bond_shape[4]))

        iter_barrier = threading.Barrier(num_threads + 1)

        def thread_main(ithread):
            grad = np.zeros(bond_shape)
            for i in range(ithread * num_per_thd, (ithread + 1) * num_per_thd):
                grad = grad + self.__gradient__(i)
            grads[ithread] = grad
            iter_barrier.wait()

        worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]
        for t in worker_threads:
            t.start()
        iter_barrier.wait()

        for t in worker_threads:
            t.join()

        # Here, all gradients have been calculated
        total_grad = np.sum(grads, axis=0)
        new_bond = tn.Node(bond.tensor - alpha * total_grad)
        self.__add_newbond_names__(new_bond)
        self.mps.update_bond(new_bond, max_singular_values)
        for i in range(B):
            self.__precompute_projections__(i)
        return self.mps.output_idx < len(self.mps.tensors) - 2
