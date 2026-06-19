"""
PATCH 2 — classifier_pipeline.py
Frozen CIFAR-10 classifier used to compute the Config B metric
(top-1 classification accuracy on reconstructed images).

>>> NOTE ON WEIGHTS <<<
The spec originally said "load torchvision pretrained weights". torchvision's
ResNet weights are pretrained on **ImageNet** (1000 classes, 224x224) and are
NOT a CIFAR-10 classifier. Per confirmed decision, we use the
**chenyaofo/pytorch-cifar-models** CIFAR-10 ResNet-20 via torch.hub
(auto-download, reported test accuracy ~92.6%). No manual weight handling.
  https://github.com/chenyaofo/pytorch-cifar-models

DESIGN REQUIREMENTS MET
  * Frozen: requires_grad=False on all params; .eval() always (train() is a
    no-op that keeps eval); forward under torch.no_grad(). ONE shared instance /
    weights for both M1 and M2.
  * Input: reconstructed image tensor in [0,1], shape (B,3,32,32). CIFAR-10
    normalization (mean/std) is applied INSIDE forward, so callers pass raw
    [0,1] reconstructions (what both JSCC decoders emit).
  * Output: top-1 accuracy over a CIFAR-10 test loader.
  * Sanity: clean CIFAR-10 top-1 ~0.926 for chenyaofo ResNet-20;
    assert_clean_accuracy() floor = 0.91 (confirmed).

NOTE: not run yet (needs hub download + inference). Verification happens after
review and after the GPU is free.
"""
from __future__ import annotations
import torch
import torch.nn as nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
HUB_REPO = "chenyaofo/pytorch-cifar-models"
HUB_MODEL = "cifar10_resnet20"
EXPECTED_CLEAN_TOP1 = 0.9260   # chenyaofo cifar10_resnet20
SANITY_FLOOR = 0.91            # confirmed


class FrozenCIFARClassifier(nn.Module):
    """Frozen, eval-only CIFAR-10 classifier (chenyaofo ResNet-20).
    ONE shared instance for M1 and M2. Never train; never backprop through it.
    Normalization is applied INSIDE forward, so callers pass images in [0,1].
    """

    def __init__(self, hub_repo: str = HUB_REPO, hub_model: str = HUB_MODEL):
        super().__init__()
        self.net = torch.hub.load(hub_repo, hub_model, pretrained=True)
        self._source = f"{hub_repo}:{hub_model} (~{EXPECTED_CLEAN_TOP1:.3f})"

        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # FREEZE
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # hard-freeze: ignore train() calls, always stay in eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, imgs01: torch.Tensor) -> torch.Tensor:
        """imgs01: (B,3,32,32) in [0,1]. Returns class logits (B,10)."""
        x = (imgs01.clamp(0, 1) - self.mean) / self.std
        return self.net(x)

    @torch.no_grad()
    def top1(self, loader, device: str = "cuda:0") -> float:
        """Top-1 accuracy over a loader yielding (imgs01, labels)."""
        self.to(device)
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = self.forward(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
        return correct / max(total, 1)


@torch.no_grad()
def assert_clean_accuracy(clf: "FrozenCIFARClassifier", loader,
                          device: str = "cuda:0", floor: float = SANITY_FLOOR) -> float:
    """Sanity gate: clean CIFAR-10 top-1 must be >= floor (0.91); expect ~0.926."""
    acc = clf.top1(loader, device=device)
    print(f"[classifier sanity] source={clf._source} clean top-1 = {acc:.4f} "
          f"(expected ~{EXPECTED_CLEAN_TOP1:.4f}, floor {floor})")
    assert acc >= floor, f"clean accuracy {acc:.4f} below floor {floor}"
    return acc
