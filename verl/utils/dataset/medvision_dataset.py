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
from verl.utils.tokenizer import build_multimodal_processor_inputs

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
        # Curriculum sample filtering (data.curriculum.enable). The trainer calls
        # init_curriculum() on the train dataset only, so the val dataset never
        # carries a manager and the disabled path is unchanged.
        self.curriculum_manager = None

    def init_curriculum(self, curriculum_config, min_active_size: int):
        """Creates the curriculum manager from per-row task labels (train dataset only)."""
        from verl.utils.dataset.curriculum import CurriculumManager
        from verl.utils.dataset.temperature_sampler import parse_task_group_map

        task_key = curriculum_config.get("task_key", "ability")
        if task_key not in self.dataframe.column_names:
            raise ValueError(f"Curriculum requires a dataset column '{task_key}' for per-task pooling.")
        group_map = parse_task_group_map(curriculum_config.get("task_group_map", None))
        task_labels = [group_map.get(str(label), str(label)) for label in self.dataframe[task_key]]

        # Detection is scored by CIoU overlap error (1-CIoU)/2, not MRE, so it gets its own
        # promotion gate (matched to IoU>0.5) instead of the MRE<mre_gate bar used by A/D + T/L.
        detection_gate = curriculum_config.get("detection_gate", None)
        gate_overrides = {} if detection_gate is None else {"medvision-detection": float(detection_gate)}

        self.curriculum_manager = CurriculumManager(
            task_labels,
            gate_overrides=gate_overrides,
            easy_top_frac=curriculum_config.get("easy_top_frac", 0.20),
            mre_gate=curriculum_config.get("mre_gate", 0.10),
            threshold_frac=curriculum_config.get("threshold_frac", 0.50),
            mixin_easy_frac=curriculum_config.get("mixin_easy_frac", 0.30),
            demote_easy=curriculum_config.get("demote_easy", True),
            min_active_size=min_active_size,
            ema_alpha=curriculum_config.get("ema_alpha", 0.4),
            promote_patience=curriculum_config.get("promote_patience", 1),
            demote_patience=curriculum_config.get("demote_patience", 1),
            demote_margin=curriculum_config.get("demote_margin", 1.5),
            audit_frac=curriculum_config.get("audit_frac", 0.05),
            mixin_ramp=curriculum_config.get("mixin_ramp", True),
            task_floor_frac=curriculum_config.get("task_floor_frac", 0.10),
        )
        print(
            f"[Info] Curriculum filtering enabled: {self.curriculum_manager.n_samples} samples, "
            f"tasks {dict((t, len(idx)) for t, idx in self.curriculum_manager.task_indices.items())}"
        )

    def on_batch_end(self, batch):
        """Tallies per-sample rewards/errors after each training step (called by the trainer)."""
        if self.curriculum_manager is None:
            return
        non_tensor = batch.non_tensor_batch
        if "answer_error" not in non_tensor:
            raise RuntimeError(
                "Curriculum filtering needs per-sample 'answer_error' in the batch, which is only "
                "populated by the reward loop. Launch with +reward_model.use_reward_loop=True."
            )
        if "index" not in non_tensor:
            raise RuntimeError(
                "Curriculum filtering needs per-sample 'index' in the batch. It is injected by "
                "MedVisionDataset.__getitem__ and must be retained in the trainer-side batch by "
                "_get_gen_batch (the agent-loop output does not echo it when the reward loop is on)."
            )
        scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().tolist()
        self.curriculum_manager.record(non_tensor["index"], scores, non_tensor["answer_error"])

    def advance_curriculum(self):
        """Reclassifies pools at an epoch boundary; returns the next epoch's active indices."""
        active = self.curriculum_manager.advance_epoch()
        return active

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        if self.curriculum_manager is not None:
            # Stable per-sample identity = post-filter row position; flows into
            # non_tensor_batch["index"] and survives the n-rollout repeat.
            row_dict["index"] = item
        return row_dict

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

                        if images is None:
                            # text-only prompt: count tokens via the tokenizer directly
                            return len(
                                processor.tokenizer(
                                    text=raw_prompt,
                                    add_special_tokens=False,  # avoid adding special tokens
                                    return_attention_mask=False,
                                )["input_ids"]
                            )
                        # multimodal prompt: route through the maintained helper (mirrors RLHFDataset);
                        # the pre-resized images are passed straight through, so MedVisionDataset still
                        # does not re-resize.
                        return len(
                            build_multimodal_processor_inputs(
                                processor,
                                text=[raw_prompt],
                                images=images,
                                videos=None,
                                audio=None,
                                mm_processor_kwargs=self.mm_processor_kwargs,
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

