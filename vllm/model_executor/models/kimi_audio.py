# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, Union, Any

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from ...transformers_utils.configs import KimiAudioConfig
from ...transformers_utils.processors import KimiAudioProcessor, WhisperEncoder
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .moonaudio import MoonshotKimiaModel
from .utils import AutoWeightsLoader, maybe_prefix, WeightsMapper, _flatten_embeddings


class KimiAudioMultiModalProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim,
                      config.hidden_size,
                      bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size,
                         eps=config.rms_norm_eps,
                         bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class KimiAudioInputs(TypedDict):
    audio_input_ids: torch.Tensor
    """Shape: `(num_audios, seq_len)`"""

    is_continuous_mask: torch.Tensor
    """Shape: `(num_audios, seq_len)`"""

    whisper_input_feature: torch.Tensor
    """Shape: `(num_audios, seq_len, feature_dim)`"""


class KimiAudioProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> KimiAudioConfig:
        return self.ctx.get_hf_config(KimiAudioConfig)

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class KimiAudioDummyInputsBuilder(
        BaseDummyInputsBuilder[KimiAudioProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "请识别电话沟通场景中如下声音片段的类别。若检测到清晰语音起始或强烈发言意愿（如语句开头、语气转折），应回复<3>；否则若含附和词（如\"嗯\"、\"yeah\"），应回复<2>；除1、2外的所有其他情况如非语言声音（如喷嚏、咳嗽、笑声）、噪声或近似静默等非打断性信号，应回复<1>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio":
            self._get_dummy_audios(length=30*16000, num_audios=num_audios)
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts)
        tokenization_kwargs = {"truncation": False}

        res = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(),
            dict(text=dummy_text, audio=dummy_mm_data["audio"]),
            # dict(**mm_kwargs, **tok_kwargs),
        )

        return ProcessorInputs(prompt=res['input_ids'][0].tolist(),
                               mm_data=dummy_mm_data,
                               tokenization_kwargs=tokenization_kwargs)

class KimiAudioMultiModalProcessor(
        BaseMultiModalProcessor[KimiAudioProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            # audio=MultiModalFieldConfig.batched("audio"),
            audio_input_ids=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            whisper_input_feature=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)

            num_audio_tokens = processor.get_num_audio_tokens(
                audio_path={"data": audio})

            return [processor.audio_token_id] * num_audio_tokens

        return [
            PromptReplacement(
                modality="audio",
                target="",  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        return_mm_hashes: bool,
    ):
        (
            prompt_ids,
            mm_kwargs,
            mm_hashes,
            _,
        ) = super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            return_mm_hashes=return_mm_hashes,
        )

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_kwargs, mm_hashes, True

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # NOTE - we rename audios -> audio in mm data because transformers has
        # deprecated audios for the qwen2audio processor and will remove
        # support for it in transformers 4.54.
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt, bos=False, eos=False, allowed_special="all")
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=16000,
        )

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vq_adaptor.": "multi_modal_projector.",
            "model.": "language_model.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_media_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config
        self.config = config
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        mel_batch_size = getattr(config, "mel_batch_size", 20)
        encoder_path = os.path.join(model_config.model, "whisper-large-v3")
        self.audio_tower = WhisperEncoder(
            encoder_path,
            mel_batch_size=mel_batch_size,
        )
        self.multi_modal_projector = KimiAudioMultiModalProjector(self.config)
        self.language_model = MoonshotKimiaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "multi_modal_model"),
        )

        # text only
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                self.config.vocab_size,
                                                logit_scale)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. \
                             Got type: {type(mm_input)}")

        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[KimiAudioInputs]:
        audio_input_ids = kwargs.pop('audio_input_ids', None)
        is_continuous_mask = kwargs.pop('is_continuous_mask', None)
        whisper_input_feature = kwargs.pop('whisper_input_feature', None)

        if whisper_input_feature is None:
            return None

        audio_input_ids = self._validate_and_reshape_mm_tensor(
            audio_input_ids, 'audio_input_ids')
        is_continuous_mask = self._validate_and_reshape_mm_tensor(
            is_continuous_mask, 'is_continuous_mask')
        whisper_input_feature = self._validate_and_reshape_mm_tensor(
            whisper_input_feature, 'whisper_input_feature')

        return KimiAudioInputs(
            audio_input_ids=audio_input_ids,
            is_continuous_mask=is_continuous_mask,
            whisper_input_feature=whisper_input_feature,
        )


    def _process_audio_input(self,
                             audio_input: KimiAudioInputs) -> torch.Tensor:
        audio_input_ids = audio_input["audio_input_ids"]
        whisper_input_feature = audio_input["whisper_input_feature"]
        is_continuous_mask = audio_input["is_continuous_mask"]
        assert 151666 not in audio_input_ids
        batch_size = whisper_input_feature.shape[0]

        whisper_input_feature_tmp = []
        for i in range(batch_size):
            tmp = self.audio_tower.tokenize_waveform(whisper_input_feature[i])
            whisper_input_feature_tmp.append(tmp.reshape(tmp.shape[0], int(tmp.shape[1] // 4), tmp.shape[2] * 4))
        whisper_input_feature = whisper_input_feature_tmp

        device = self.language_model.embed_tokens.weight.device
        audio_input_ids = audio_input_ids.to(device)
        start_idx = (audio_input_ids == 151670).nonzero()[0, 1] + 1
        end_idx = (audio_input_ids == 151661).nonzero()[0, 1]
        audio_input_ids[:, start_idx:end_idx] = 151666

        audio_emb = self.language_model.get_input_embeddings(audio_input_ids)
        if self.config.use_whisper_feature:
            assert isinstance(whisper_input_feature, list)

            media_start_idx = (
                audio_input_ids == self.kimia_media_begin).nonzero()
            media_end_idx = (audio_input_ids == self.kimia_media_end).nonzero()
            # shape: batch, seq_len, hidden_size
            whisper_input_dim = whisper_input_feature[0].shape[-1]
            whisper_dtype = whisper_input_feature[0].dtype
            projector_device = self.multi_modal_projector.\
                                layers[0].weight.device
            expanded_whisper = torch.zeros(batch_size,
                                           audio_emb.shape[1],
                                           whisper_input_dim,
                                           dtype=whisper_dtype,
                                           device=projector_device)
            for (seg_idx, start_idx), (_,
                                       end_idx) in zip(media_start_idx,
                                                       media_end_idx):
                feat_len = end_idx - (start_idx + 1)
                whisper_input_feature_i = whisper_input_feature[seg_idx].\
                                        squeeze(0)
                assert feat_len == is_continuous_mask[seg_idx].sum()
                expanded_whisper[seg_idx, start_idx + 1:end_idx, :] = (
                    whisper_input_feature_i[:feat_len, :])

            whisper_emb = self.multi_modal_projector(expanded_whisper)
            whisper_emb = whisper_emb.to(device)
            is_continuous_mask = is_continuous_mask.to(device)
            whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

            encoder_input_addwith_discrete_token = (
                audio_emb + whisper_emb) * torch.sqrt(
                    torch.tensor(2.0,
                                 dtype=whisper_emb.dtype,
                                 device=whisper_emb.device))
            audio_emb = (audio_emb * (~is_continuous_mask[:, :, None]) +
                         encoder_input_addwith_discrete_token *
                         is_continuous_mask[:, :, None])
        return audio_emb

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    # def get_multimodal_embeddings(
    #         self, **kwargs: object) -> Union[MultiModalEmbeddings, None]:
    #     audio_input = self._parse_and_validate_audio_input(**kwargs)
    #     if audio_input is None:
    #         return None

    #     processed_features = self._process_audio_input(audio_input)
    #     return processed_features

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Union[MultiModalEmbeddings, None]:
        audio_input_ids = kwargs.pop('audio_input_ids', None)
        is_continuous_mask = kwargs.pop('is_continuous_mask', None)
        whisper_input_feature = kwargs.pop('whisper_input_feature', None)

        audio_embeddings = []
        for i in range(len(audio_input_ids)):
            audio_input = KimiAudioInputs(
                audio_input_ids=audio_input_ids[i],
                is_continuous_mask=is_continuous_mask[i],
                whisper_input_feature=whisper_input_feature[i],
            )
            audio_embeddings.append(self._process_audio_input(audio_input)[0])
        return audio_embeddings

    def _merge_multimodal_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        is_multimodal: torch.Tensor,
        multimodal_embeddings: NestedTensors,
    ) -> torch.Tensor:
        flattened = _flatten_embeddings(multimodal_embeddings)
        inputs_embeds[is_multimodal] += flattened.to(dtype=inputs_embeds.dtype)
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        text_input_ids = input_ids.clone()
        text_input_ids[text_input_ids == 151650] = 151666
        inputs_embeds = self.language_model.get_input_embeddings(text_input_ids)

        if multimodal_embeddings is not None and \
            len(multimodal_embeddings) != 0:
            # customized merge
            start_indices = (input_ids == 151670).nonzero(as_tuple=True)[0]
            end_indices = (input_ids == 151671).nonzero(as_tuple=True)[0]
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for start, end in zip(start_indices, end_indices):
                mask[start:end+1] = True
            inputs_embeds = self._merge_multimodal_embeddings(
                inputs_embeds,
                mask,
                multimodal_embeddings,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[tuple[torch.Tensor], IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        **kwargs: object,
    ) -> Optional[torch.Tensor]:
        # currently only text logits are supported
        text_logits = self.logits_processor(self.lm_head, hidden_states,
                                            sampling_metadata, **kwargs)
        return text_logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mimo_output.", "language_model.mimo_layers."])
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        skips = {item[0] for item in list(self.named_parameters()) if item[0].startswith("audio_tower")}
        return loaded | skips
