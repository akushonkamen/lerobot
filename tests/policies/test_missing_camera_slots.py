#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Regression tests for missing-camera slot assignment (issue #5118).

When a camera is missing from the batch, the remaining cameras must stay in their
own `image_features` slots (missing ones become padded slots with zero masks) —
they must not shift forward into earlier slots.

These tests call `_preprocess_images` / `prepare_images` directly on a lightweight
shim that provides only `config` and `parameters()`, so no policy weights are
constructed and no downloads happen.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

RES = (8, 8)
B = 2
IMG_KEYS = [
    "observation.images.base_0_rgb",
    "observation.images.wrist_0_rgb",
    "observation.images.side_0_rgb",
]


def _identity(img, *args, **kwargs):
    return img


@pytest.fixture(autouse=True)
def no_resize(monkeypatch):
    import lerobot.policies.pi0.modeling_pi0 as pi0_mod
    import lerobot.policies.pi05.modeling_pi05 as pi05_mod
    import lerobot.policies.smolvla.modeling_smolvla as smolvla_mod

    monkeypatch.setattr(pi05_mod, "resize_with_pad_torch", _identity)
    monkeypatch.setattr(pi0_mod, "resize_with_pad_torch", _identity)
    monkeypatch.setattr(smolvla_mod, "resize_with_pad", _identity)


class _Shim:
    """Stand-in policy: the preprocessing methods only read `config` and `parameters()`."""

    def __init__(self, config):
        self.config = config
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])


def _img(name):
    # Distinct constant value per camera so slots can be told apart after *2-1 normalization.
    return torch.full((B, 3, *RES), 0.25 * (IMG_KEYS.index(name) + 1))


def _expected(present):
    exp_img, exp_mask = [], []
    for key in IMG_KEYS:
        if key in present:
            exp_img.append(_img(key) * 2.0 - 1.0)
            exp_mask.append(torch.ones(B, dtype=torch.bool))
        else:
            exp_img.append(torch.ones(B, 3, *RES) * -1.0)
            exp_mask.append(torch.zeros(B, dtype=torch.bool))
    return exp_img, exp_mask


def _make_config(cls, **extra):
    config = cls(max_action_dim=7, max_state_dim=14)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *RES)) for k in IMG_KEYS},
    }
    for key, value in extra.items():
        setattr(config, key, value)
    return config


def _assert_slots(config, fn, batch, present):
    images, masks = fn(_Shim(config), batch)
    assert len(images) == len(IMG_KEYS)
    exp_img, exp_mask = _expected(present)
    for i, key in enumerate(IMG_KEYS):
        assert torch.equal(images[i], exp_img[i]), f"slot {i} image wrong for {key}"
        assert torch.equal(masks[i], exp_mask[i]), f"slot {i} mask wrong for {key}"


PI05 = pytest.param(PI05Policy._preprocess_images, PI05Config, id="pi05")
PI0 = pytest.param(PI0Policy._preprocess_images, PI0Config, id="pi0")
SMOLVLA = pytest.param(SmolVLAPolicy.prepare_images, SmolVLAConfig, id="smolvla")


@pytest.mark.parametrize("fn,config_cls", [PI05, PI0, SMOLVLA])
def test_first_camera_missing_keeps_slots(fn, config_cls):
    present = IMG_KEYS[1:]
    batch = {"observation.state": torch.zeros(B, 7), **{k: _img(k) for k in present}}
    _assert_slots(_make_config(config_cls, empty_cameras=3), fn, batch, present)


@pytest.mark.parametrize("fn,config_cls", [PI05, PI0, SMOLVLA])
def test_all_cameras_present(fn, config_cls):
    batch = {"observation.state": torch.zeros(B, 7), **{k: _img(k) for k in IMG_KEYS}}
    _assert_slots(_make_config(config_cls, empty_cameras=3), fn, batch, IMG_KEYS)


@pytest.mark.parametrize("fn,config_cls", [PI05, PI0, SMOLVLA])
def test_last_camera_missing_unchanged(fn, config_cls):
    # Legacy scenario: only the last camera missing must behave exactly as before.
    present = IMG_KEYS[:-1]
    batch = {"observation.state": torch.zeros(B, 7), **{k: _img(k) for k in present}}
    _assert_slots(_make_config(config_cls, empty_cameras=3), fn, batch, present)


def test_smolvla_empty_cameras_zero_keeps_legacy_subset():
    # With empty_cameras=0 SmolVLA outputs only the present cameras (legacy behaviour),
    # but each present camera must still sit in its own slot.
    config = _make_config(SmolVLAConfig, empty_cameras=0, resize_imgs_with_padding=None)
    present = IMG_KEYS[1:]
    batch = {"observation.state": torch.zeros(B, 7), **{k: _img(k) for k in present}}
    images, masks = SmolVLAPolicy.prepare_images(_Shim(config), batch)
    assert len(images) == 2
    for i, key in enumerate(present):
        assert torch.equal(images[i], _img(key) * 2.0 - 1.0)
        assert torch.equal(masks[i], torch.ones(B, dtype=torch.bool))
