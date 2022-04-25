# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
# paddle.disable_static()

geo = psci.geometry.CylinderInCube(
    origin=(-0.05, -0.05, -0.05),
    extent=(0.05, 0.05, 0.05),
    circle_center=(0.0, 0.0),
    circle_radius=0.01)

geo.add_boundary(name="top", criteria=lambda x, y, z: z == 0.05)
geo.add_boundary(name="down", criteria=lambda x, y, z: z == -0.05)
geo.add_boundary(name="left", criteria=lambda x, y, z: x == -0.05)
geo.add_boundary(name="right", criteria=lambda x, y, z: x == 0.05)
geo.add_boundary(name="front", criteria=lambda x, y, z: y == -0.05)
geo.add_boundary(name="back", criteria=lambda x, y, z: y == 0.05)

# N-S
pde = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=3, time_dependent=True)

# set bounday condition
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0)
bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_top_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_down_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_right_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_front_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_front_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_front_w = psci.bc.Dirichlet('w', rhs=0.0)

bc_back_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_back_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_back_w = psci.bc.Dirichlet('w', rhs=0.0)

pde.add_geometry(geo)

# add bounday and boundary condition
pde.add_bc("top", bc_top_u, bc_top_v, bc_top_w)
pde.add_bc("down", bc_down_u, bc_down_v, bc_down_w)
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_u, bc_right_v, bc_right_w)
pde.add_bc("front", bc_front_u, bc_front_v, bc_front_w)
pde.add_bc("back", bc_back_u, bc_back_v, bc_back_w)

# Discretization
pde_disc = psci.discretize(
    pde,
    time_method="implicit",
    time_step=0.01,
    space_npoints=1000,
    space_method="sampling")

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=3,
    num_outs=4,
    num_layers=10,
    hidden_size=50,
    dtype="float32",
    activation='tanh')

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

n = pde_disc.geometry.interior.shape[0]

data_n = np.zeros((n, 3), dtype='float32')
solver.feed_data_n(data_n)

solution = solver.solve(num_epoch=2)

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)
