"""OpenPI transforms for the RoboTwin Panthera dual-arm dataset."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class PantheraInputs(transforms.DataTransformFn):
    """Map the Panthera LeRobot schema to OpenPI's canonical observation schema."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    STATE_DIM: ClassVar[int] = 14

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (self.STATE_DIM,):
            raise ValueError(f"Panthera state must have shape ({self.STATE_DIM},), got {state.shape}")

        images = {}
        for camera_name in self.EXPECTED_CAMERAS:
            if camera_name not in data["images"]:
                raise ValueError(f"Panthera dataset is missing required camera {camera_name!r}")
            images[camera_name] = _decode_image(data["images"][camera_name])

        output = {
            "image": {
                "base_0_rgb": images["cam_high"],
                "left_wrist_0_rgb": images["cam_left_wrist"],
                "right_wrist_0_rgb": images["cam_right_wrist"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != self.STATE_DIM:
                raise ValueError(
                    f"Panthera actions must have shape [T, {self.STATE_DIM}], got {actions.shape}"
                )
            output["actions"] = actions

        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class PantheraOutputs(transforms.DataTransformFn):
    """Map OpenPI actions back to the 14-dimensional Panthera action space."""

    ACTION_DIM: ClassVar[int] = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < self.ACTION_DIM:
            raise ValueError(f"OpenPI actions must have at least {self.ACTION_DIM} dims, got {actions.shape}")
        return {"actions": actions[:, : self.ACTION_DIM]}


def _decode_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Panthera images must be rank 3, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[0] in (1, 3):
        return einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] in (1, 3):
        return image
    raise ValueError(f"Panthera images must be CHW or HWC with 1/3 channels, got shape {image.shape}")
