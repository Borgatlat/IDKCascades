from pathlib import Path
import torch
from torch._inductor.ir import NoneLayout
from models.dual_modal_cnn import build_ki_model
from utils.classifier_registry import ClassifierRegistry
from utils.labels import KI_REGISTRY
from training.trainer import get_device


def load_cascade_models(checkpoint_dir: Path, registry_path: Path) -> dict:
    registry = ClassifierRegistry.load(registry_path)
    device = get_device()
    models = {}
    
    for ki_name in KI_REGISTRY:
        rec = registry.get(ki_name)
        if rec is None or not rec.checkpoint:
            raise ValueError(f"No record found for {ki_name}")
            continue

        checkpoint = torch.load(rec.checkpoint, map_location = device, weights_only = True)

        model = build_ki_model(ki_name, len(rec.class_names)).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        models[ki_name] = model

    return models, registry, device

    
    
    