# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builder utilities."""

from typing import Any, Dict

from composer import DataSpec
from transformers import PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.utils.registry_utils import construct_from_registry


#from llmfoundry.utils.config_utils import TrainConfig
from olmo.config import TrainConfig, DataConfig
from olmo.data import MemMapDataset

from torch.utils.data import DataLoader

__all__ = [
    'build_dataloader',
    'build_train_dataloader',
]


def build_dataloader(
    cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
) -> DataSpec:
    """Builds a dataloader from a config.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        tokenizer (PreTrainedTokenizerBase): The tokenizer that the model will use.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    name = cfg.pop('name')
    kwargs: Dict[str, Any] = {
        **cfg,
        'tokenizer': tokenizer,
        'device_batch_size': device_batch_size,
    }

    return construct_from_registry(
        name=name,
        registry=registry.dataloaders,
        partial_function=False,
        pre_validation_function=None,
        post_validation_function=None,
        kwargs=kwargs,
    )


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []

    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
        instance_filter_config=data_config.instance_filter,
    )

def build_train_dataloader(train_config: TrainConfig) -> DataLoader:
    assert train_config.device_train_batch_size is not None

    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    dataset = build_memmap_dataset(train_config, train_config.data, include_instance_metadata=False)
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
