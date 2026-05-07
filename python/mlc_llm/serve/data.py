"""Classes denoting multi-modality data used in MLC LLM serving"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union  # noqa: UP035

import tvm
import tvm_ffi
from tvm.runtime import Object, Tensor

from . import _ffi_api


@tvm_ffi.register_object("mlc.serve.Data")
class Data(Object):
    """The base class of multi-modality data (text, tokens, embedding, etc)."""

    def __init__(self):
        pass


@tvm_ffi.register_object("mlc.serve.TextData")
class TextData(Data):
    """The class of text data, containing a text string.

    Parameters
    ----------
    text : str
        The text string.
    """

    def __init__(self, text: str):
        self.__init_handle_by_constructor__(_ffi_api.TextData, text)

    @property
    def text(self) -> str:
        """The text data in `str`."""
        return str(_ffi_api.TextDataGetTextString(self))

    def __str__(self) -> str:
        return self.text


@tvm_ffi.register_object("mlc.serve.TokenData")
class TokenData(Data):
    """The class of token data, containing a list of token ids.

    Parameters
    ----------
    token_ids : List[int]
        The list of token ids.
    """

    def __init__(self, token_ids: List[int]):  # noqa: UP006
        self.__init_handle_by_constructor__(_ffi_api.TokenData, *token_ids)

    @property
    def token_ids(self) -> List[int]:  # noqa: UP006
        """Return the token ids of the TokenData."""
        return list(_ffi_api.TokenDataGetTokenIds(self))


# mypy: disable-error-code="attr-defined"
@tvm_ffi.register_object("mlc.serve.ImageData")
class ImageData(Data):
    """The class of image data, containing the image as Tensor.

    Parameters
    ----------
    image : tvm.runtime.Tensor
        The image data.
    """

    def __init__(
        self,
        image: Tensor,
        embed_size: int,
        grid_thw: Optional[Tuple[int, int, int]] = None,  # noqa: UP006
    ):
        if grid_thw is None:
            self.__init_handle_by_constructor__(_ffi_api.ImageData, image, embed_size)
        else:
            grid_t, grid_h, grid_w = grid_thw
            self.__init_handle_by_constructor__(
                _ffi_api.ImageDataWithGrid, image, embed_size, grid_t, grid_h, grid_w
            )

    @property
    def image(self) -> Tensor:
        """Return the image data."""
        return _ffi_api.ImageDataGetImage(self)

    def get_grid_thw(self) -> Optional[Tuple[int, int, int]]:  # noqa: UP006
        """Return Qwen-style patch grid metadata, when available."""
        grid_t, grid_h, grid_w = _ffi_api.ImageDataGetGridTHW(self)
        if grid_t == 0 or grid_h == 0 or grid_w == 0:
            return None
        return int(grid_t), int(grid_h), int(grid_w)

    def __len__(self):
        return int(_ffi_api.ImageDataGetEmbedSize(self))

    @staticmethod
    def from_url(url: str, config: Dict) -> "ImageData":  # noqa: UP006
        """Get the image from the given URL, process and return the image tensor as TVM Tensor."""

        import base64
        from io import BytesIO

        import numpy as np
        import requests
        from PIL import Image

        if url.startswith("data:image"):
            # The image is encoded in base64 format
            base64_image = url.split(",")[1]
            image_data = base64.b64decode(base64_image)
            image_tensor = Image.open(BytesIO(image_data)).convert("RGB")
        elif url.startswith("http"):
            response = requests.get(url, timeout=5)
            image_tensor = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image URL format: {url}")

        image_embed_size, grid_thw = ImageData.get_embed_size(
            config, image_tensor.size[::-1], return_grid=True
        )
        image_tensor = np.expand_dims(np.asarray(image_tensor, dtype="uint8"), axis=0)
        image_features = tvm.runtime.tensor(image_tensor)
        image_data = ImageData(image_features, image_embed_size, grid_thw)
        return image_data

    @staticmethod
    def _smart_resize(
        height: int,
        width: int,
        factor: int,
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
    ) -> Tuple[int, int]:
        """Resize like Qwen VL processors: keep aspect ratio and align to patch factor."""

        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image shape: height={height}, width={width}.")
        if max(height, width) / min(height, width) > 200:
            raise ValueError(f"Image aspect ratio is too large: height={height}, width={width}.")

        resized_height = max(factor, round(height / factor) * factor)
        resized_width = max(factor, round(width / factor) * factor)
        if resized_height * resized_width > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = math.floor(height / beta / factor) * factor
            resized_width = math.floor(width / beta / factor) * factor
        elif resized_height * resized_width < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        return int(resized_height), int(resized_width)

    @staticmethod
    def get_embed_size(
        config: Dict,
        image_shape: Optional[Tuple[int, int]] = None,  # noqa: UP006
        return_grid: bool = False,
    ) -> Union[int, Tuple[int, Optional[Tuple[int, int, int]]]]:  # noqa: UP006
        """Get the image embedding size from the model config file."""
        model_config = config["model_config"]
        vision_config = model_config["vision_config"]
        if config["model_type"] == "qwen3_5":
            if image_shape is None:
                raise ValueError("image_shape is required to compute Qwen3.5 image embed size.")
            height, width = image_shape
            patch_size = int(vision_config.get("patch_size", 16))
            merge_size = int(vision_config.get("spatial_merge_size", 2))
            factor = patch_size * merge_size
            min_pixels = int(vision_config.get("min_pixels", 65536))
            max_pixels = int(vision_config.get("max_pixels", 16777216))
            resized_height, resized_width = ImageData._smart_resize(
                height, width, factor, min_pixels, max_pixels
            )
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            embed_size = (grid_h // merge_size) * (grid_w // merge_size)
            if return_grid:
                return embed_size, (1, grid_h, grid_w)
            return embed_size

        if config["model_type"] == "phi3_v":
            return (1921, None) if return_grid else 1921
        if "image_size" in vision_config:
            image_size = vision_config["image_size"]
            patch_size = vision_config["patch_size"]
            embed_size = (image_size // patch_size) ** 2
            return (embed_size, None) if return_grid else embed_size
        return (576, None) if return_grid else 576

    @staticmethod
    def get_input_size(config: Dict) -> int:  # noqa: UP006
        """Get the image input size from the model config file."""
        vision_config = config["model_config"]["vision_config"]
        if config["model_type"] == "qwen3_5":
            return int(math.sqrt(int(vision_config.get("min_pixels", 65536))))
        image_size = vision_config["image_size"]
        return image_size


@dataclass
class SingleRequestStreamOutput:
    """The request stream output of a single request.

    Attributes
    ----------
    delta_token_ids : List[int]
        The new generated tokens since the last callback invocation
        for the input request.

    delta_logprob_json_strs : Optional[List[str]]
        The logprobs JSON strings of the new generated tokens
        since last invocation.

    finish_reason : Optional[str]
        The finish reason of the request when it is finished,
        of None if the request has not finished yet.
    """

    delta_token_ids: List[int]  # noqa: UP006
    delta_logprob_json_strs: Optional[List[str]]  # noqa: UP006
    finish_reason: Optional[str]
    request_final_usage_json_str: Optional[str]
    extra_prefix_string: str


@tvm_ffi.register_object("mlc.serve.RequestStreamOutput")
class RequestStreamOutput(Object):
    """The generated delta request output that is streamed back
    through callback stream function.
    It contains four fields (in order):

    request_id : str
        The id of the request that the function is invoked for.

    stream_outputs : List[SingleRequestStreamOutput]
        The output instances, one for a request.

    Note
    ----
    We do not provide constructor, since in practice only C++ side
    instantiates this class.
    """

    def unpack(self) -> Tuple[str, List[SingleRequestStreamOutput]]:  # noqa: UP006
        """Return the fields of the delta output in a tuple.

        Returns
        -------
        request_id : str
            The id of the request that the function is invoked for.

        stream_outputs : List[SingleRequestStreamOutput]
            The output instances, one for a request.
        """
        fields = _ffi_api.RequestStreamOutputUnpack(self)
        request_final_usage_json_str = fields[4]
        request_id = str(fields[0])
        if request_final_usage_json_str is not None:
            return (
                request_id,
                [SingleRequestStreamOutput([], None, None, request_final_usage_json_str, "")],
            )

        stream_outputs = []
        for i, (delta_token_ids, finish_reason, extra_prefix_string) in enumerate(
            zip(fields[1], fields[3], fields[5])
        ):
            delta_logprob_json_strs = (
                [str(logprob_json_str) for logprob_json_str in fields[2][i]]
                if fields[2] is not None
                else None
            )
            stream_outputs.append(
                SingleRequestStreamOutput(
                    delta_token_ids=list(delta_token_ids),
                    delta_logprob_json_strs=delta_logprob_json_strs,
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                    request_final_usage_json_str=None,
                    extra_prefix_string=str(extra_prefix_string),
                )
            )
        return request_id, stream_outputs
