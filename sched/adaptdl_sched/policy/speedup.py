# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


# Context: This file serves as a crucial adapter between the low-level `GoodputFunction`
# and the high-level cluster scheduler (`allocator.py`). It abstracts the complex "Goodput"
# metric into a simple, standardized "Speedup" ratio, which schedulers can easily use
# to make decisions about resource allocation across different jobs.
#
# 上下文: 此文件是底层 `GoodputFunction` 与上层集群调度器 (`allocator.py`) 之间的关键适配器。
# 它将复杂的“有效吞吐量”（Goodput）指标抽象成一个简单、标准化的“加速比”（Speedup），
# 使调度器能轻易地用它来为不同任务做出资源分配决策。
#

class SpeedupFunction(object):
    """
    Wraps a GoodputFunction to provide a standardized speedup metric.
    It answers the question: "How many times faster will this job run with N GPUs
    compared to running with a single GPU?"

    此类封装了一个 GoodputFunction，以提供一个标准化的加速比指标。
    它回答了这样一个问题：“此任务在 N 个 GPU 上运行的速度，是在单个 GPU 上运行速度的多少倍？”
    """

    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, mem_size=32):
        self._goodput_fn = goodput_fn  # The underlying GoodputFunction model. | 底层的 GoodputFunction 模型。
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._mem_size = mem_size

        # [Core Logic] Calculate the base_goodput for a single replica, which serves as the
        # baseline (speedup = 1.0) for all other configurations.
        # [核心逻辑] 计算单个副本时的基准 goodput，它将作为所有其他配置的基线（此时加速比为1.0）。
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation)

        # Memoization for fast repeated queries.
        self._mem_speedup = -np.ones((mem_size, mem_size))
        self._mem_speedup[0, 0] = 0.0

    def __call__(self, num_nodes, num_replicas):
        """
        Calculates the speedup for a given number of nodes and replicas.

        为给定的节点和副本数计算加速比。
        """
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))
        # ... (Input flattening and memoization logic)
        output_scalar = np.isscalar(num_nodes) and np.isscalar(num_replicas)
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        speedup = -np.ones(output_shape).flatten()
        indices = num_replicas < self._mem_size
        mem_idx = (num_nodes[indices], num_replicas[indices])
        speedup[indices] = self._mem_speedup[mem_idx]
        missing = speedup < 0
        if np.count_nonzero(missing) > 0:
            num_nodes, num_replicas = num_nodes[missing], num_replicas[missing]
            (num_nodes, num_replicas), inverse = np.unique(
                    np.stack([num_nodes, num_replicas]),
                    axis=1, return_inverse=True)

            # 1. Get the optimal goodput for the given configuration.
            #    获取给定配置下的最优 goodput。
            goodput, _, _ = self._goodput_fn.optimize(
                num_nodes, num_replicas,
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._atomic_bsz_range,
                accumulation=self._accumulation)

            # 2. [Core Logic] Calculate speedup by normalizing with the base_goodput.
            #    [核心逻辑] 通过与基准 goodput 相除来计算加速比，实现标准化。
            # SELF-AWARE: 这是连接底层模型和上层调度的“适配器”。它将GoodputFunction封装起来，
            # 对外提供一个更简洁的接口。调度器不关心具体的 Goodput 数值，
            # 而是关心“加速比”——即“给一个任务分配N个GPU，其运行速度会是分配1个GPU时的多少倍？”。
            # SpeedupFunction通过计算不同GPU数量下的最大Goodput并与单GPU时的基准值相除，
            # 来回答这个问题，从而为全局资源分配提供标准化的决策依据。
            calculated_speedup = goodput / self._base_goodput

            # Memoize results.
            indices = num_replicas < self._mem_size
            mem_idx = (num_nodes[indices], num_replicas[indices])
            self._mem_speedup[mem_idx] = calculated_speedup[indices]

            # Fill in computed results.
            speedup[missing] = calculated_speedup[inverse]

        assert np.all(np.less_equal(0, speedup))
        speedup = speedup.reshape(output_shape)
        return speedup.item() if output_scalar else speedup
