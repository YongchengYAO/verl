# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import traceback
from typing import Optional

import datasets
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class MedVisionDataset(RLHFDataset):
    """
    A verl dataset class for this custom data:
        - MedVision Dataset: https://huggingface.co/datasets/YongchengYAO/MedVision

    Inherited from RLHFDataset with these modifications:
        - We do not resize images in MedVisionDataset, as the prompts already contain image and pixel size info.
          Resizing the image may mislead the model as the size info in the prompt becomes
          inconsistent with the actual image size.

    We keep the rest of the functionalities same as RLHFDataset:

        Load and preprocess RLHF data from Parquet files.
        - Caches files locally.
        - Reads into a HuggingFace Dataset and tokenizes prompts.
        - Optionally handles images/videos via a ProcessorMixin.
        - Filters prompts over a max length.
        - Supports resuming from checkpoints.

        Args:
            data_files (str or list): Path(s) to Parquet file(s).
            tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
            config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
            processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc, key=self.prompt_key)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                            **apply_kwargs,
                        )

                        # NOTE: we skip image processing here to avoid resizing images in MedVisionDataset
                        if image_key in doc and doc[image_key]:
                            images = doc[image_key]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            raise NotImplementedError("Video processing is not implemented in MedVisionDataset.")
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(
                                text=[raw_prompt],
                                images=images,
                                videos=videos,
                                videos_kwargs=videos_kwargs,
                            )["input_ids"][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(
                                doc[prompt_key],
                                add_generation_prompt=True,
                                **apply_kwargs,
                            )
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def _build_messages(self, example: dict, key: str):
        # The MedVision parquet stores message content as a list of typed segments,
        # e.g. [{"type": "image"}, {"type": "text", "text": ...}], with the actual
        # image bytes in a separate `images` column. The parent _build_messages expects
        # string content with <image>/<video>/<audio> placeholders, so normalize to that
        # form here and let the parent inject images from the `images` column.
        placeholder = {"image": "<image>", "video": "<video>", "audio": "<audio>"}
        for message in example[key]:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = "".join(
                    placeholder.get(seg.get("type"), seg.get("text") or "") for seg in content
                )
        return super()._build_messages(example, key)

