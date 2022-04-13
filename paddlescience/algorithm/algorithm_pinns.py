# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from .algorithm_base import AlgorithmBase
from ..ins import InsAttr


class PINNs(AlgorithmBase):
    """
    The Physics Informed Neural Networks Algorithm.

    Parameters:
        net(NetworkBase): The NN network used in PINNs algorithm.
        loss(LossBase): The loss used in PINNs algorithm.

    Example:
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net, loss=loss)
    """

    def __init__(self, net, loss):
        super(PINNs, self).__init__()
        self.net = net
        self.loss = loss

    def create_ins(self, pde):

        ins = list()
        ins_attr = dict()

        # TODO: hard code
        ins_attr_i = dict()
        points = pde.geometry.interior
        data = points  # 
        # data = paddle.to_tensor(points, dtype='float32', stop_gradient=False)
        ins.append(data)
        ins_attr_i["0"] = InsAttr(0, 0)
        ins_attr["interior"] = ins_attr_i

        ins_attr_b = dict()
        for name, points in pde.geometry.boundary.items():
            data = points
            # data = paddle.to_tensor(points, dtype='float32', stop_gradient=False)
            ins.append(data)
            ins_attr_b[name] = InsAttr(0, 0)
        ins_attr["boundary"] = ins_attr_b

        return ins, ins_attr

    def compute(self, *args, ins_attr, pde):

        # print(args[1])

        outs = list()

        # interior out and loss
        n = 0
        for attr in ins_attr["interior"].values():
            input = args[n]
            loss_i, out_i = self.loss.eq_loss(
                pde, self.net, input, attr, bs=-1)  # TODO: bs is not used
            loss = loss_i  # TODO: += 1
            outs.append(out_i)
            n += 1

        # print("\n *********************** \n")

        # boundary out and loss
        for attr in ins_attr["boundary"].values():
            input = args[n]
            loss_b, out_b = self.loss.bc_loss(
                pde, self.net, input, attr, bs=-1)  # TODO: bs is not used
            loss += loss_b
            outs.append(out_b)
            n += 1

        return loss, outs  # TODO: return more
