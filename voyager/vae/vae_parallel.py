#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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

import torch
import torch.distributed as dist
import torch_npu
from functools import reduce
import functools
import os


class Parallel_Vae():
    def __init__(self, local_rank=None, world_size=None, gather_rank=None):
        self.gather_rank = gather_rank
        self.local_rank = local_rank
        self.world_size = world_size or dist.get_world_size()
        self.rank = dist.get_rank()
        self.last = (self.world_size - 1) == self.rank
        self.warm_stage = True
        self.all_policy = [None] * self.world_size
        self.policy = None
        self.tiling_num = 0
        self.cache_result = []
        self.init_decoder_decode()
        self.init_vae_decode()

    def init_decoder_decode(self):
        self.record_shape = {}
        self.first = True
        self.count = -1

    def decoder_decode(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.warm_stage:
                self.tiling_num += 1
                result = func(*args, **kwargs)
                self.record_shape[self.tiling_num - 1] = result.shape
                return result
            elif self.first:
                self.count += 1
                if self.count == self.tiling_num - 1:
                    self.first = False
                if self.count in self.policy:
                    result = func(*args, **kwargs)
                    self.cache_result[self.count] = result
                return torch.empty(self.record_shape[self.count], device="meta")
            else:
                self.count += 1
                if self.count == self.tiling_num - 1:
                    self.first = True
                return self.cache_result[self.count]

        return wrapper

    def init_vae_decode(self):
        self.decode_warmup_stage = True
        self.device = None
        self.dtype = None

    def vae_decode(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.decode_warmup_stage:
                self.decode_warmup_stage = False
                result = func(*args, **kwargs)
                if isinstance(result, tuple):
                    self.device = result[0].device
                    self.dtype = result[0].dtype
                else:
                    self.device = result.device
                    self.dtype = result.dtype
                self.generate_policy()
                self.init_cache_result()
                self.init_gather_shape()
                return result
            else:
                self.warm_stage = False
                self.count = -1
                func(*args, **kwargs)
                self.count = -1
                self.gather_result()
                return func(*args, **kwargs)

        return wrapper

    def gather_result(self):
        gather_list = [torch.empty(self.gather_shape[i], device=self.device, dtype=self.dtype)
                       for i in range(self.world_size)]
        rank_cache_list = [None] * len(self.policy)
        for i, tile_idx in enumerate(self.policy):
            rank_cache_list[i] = self.cache_result[tile_idx].view(-1)
        gather_tensor = torch.cat(rank_cache_list)
        dist.all_gather(gather_list, gather_tensor)

        for i in range(self.world_size):
            for j, tile_idx in enumerate(self.all_policy[i]):
                idx_start = self.split_shape[i][j]
                idx_end = self.split_shape[i][j + 1]
                self.cache_result[tile_idx] = gather_list[i][idx_start:idx_end].view(self.record_shape[tile_idx])

    def generate_policy(self):
        for i in range(self.world_size):
            self.all_policy[i] = set(list(range(i, self.tiling_num, self.world_size)))
        self.policy = self.all_policy[self.rank]

    def init_cache_result(self):
        self.cache_result = [None] * self.tiling_num

    def init_gather_shape(self):
        self.gather_shape = [0] * self.world_size
        self.split_shape = [[0] for _ in range(self.world_size)]
        for i in range(self.world_size):
            for tiling_idx in self.all_policy[i]:
                flattshape = torch.prod(torch.tensor(self.record_shape[tiling_idx]))
                self.gather_shape[i] += flattshape
                self.split_shape[i].append(flattshape)
            self.split_shape[i] = torch.cumsum(torch.tensor(self.split_shape[i]), dim=0)


def parallel_vae_tile(vae, decode, decoder_decode, local_rank=None):
    parallel_vae = Parallel_Vae(local_rank)

    decode_lst = decode.split(".")
    ori_decode = reduce(getattr, decode_lst, vae)
    decode_func = decode_lst.pop()
    ori_vae = reduce(getattr, decode_lst, vae)

    decoder_decode_lst = decoder_decode.split(".")
    ori_decoder_decode = reduce(getattr, decoder_decode_lst, vae)
    decoder_decode_func = decoder_decode_lst.pop()
    ori_vae_decoder = reduce(getattr, decoder_decode_lst, vae)

    new_decode = parallel_vae.vae_decode(ori_decode)
    new_decoder_decode = parallel_vae.decoder_decode(ori_decoder_decode)
    setattr(ori_vae, decode_func, new_decode)
    setattr(ori_vae_decoder, decoder_decode_func, new_decoder_decode)
    return vae, ori_decode, ori_decoder_decode