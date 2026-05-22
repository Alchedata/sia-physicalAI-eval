import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from AlphaBrain.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataloader_module="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataloader_module == "lerobot_datasets":
        from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
            timeout=120,
            pin_memory=True,
        )        
        if not dist.is_initialized() or dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataloader_module == "paligemma_datasets":
        from AlphaBrain.dataloader.paligemma_datasets import get_pi0_dataset as get_pi0_dataloader
        from AlphaBrain.dataloader.lerobot_datasets import collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        dataset = get_pi0_dataloader(data_cfg=vla_dataset_cfg)
        vla_train_dataloader = DataLoader(
            dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
            timeout=120,
            pin_memory=True,
        )
        return vla_train_dataloader
    elif dataloader_module == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]

        return vlm_train_dataloader
    elif dataloader_module == "cosmos_datasets":
        from AlphaBrain.dataloader.cosmos_datasets import CosmosLIBERODataset, cosmos_collate_fn
        vla_cfg = cfg.datasets.vla_data
        dataset = CosmosLIBERODataset(
            data_dir=vla_cfg.data_dir,
            chunk_size=getattr(vla_cfg, 'chunk_size', 16),
            final_image_size=getattr(vla_cfg, 'final_image_size', 224),
            use_image_aug=getattr(vla_cfg, 'use_image_aug', True),
            use_stronger_image_aug=getattr(vla_cfg, 'use_stronger_image_aug', True),
        )
        persistent_workers = getattr(vla_cfg, 'persistent_workers', False)
        return DataLoader(
            dataset,
            batch_size=vla_cfg.per_device_batch_size,
            collate_fn=cosmos_collate_fn,
            num_workers=12,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent_workers,
        )
