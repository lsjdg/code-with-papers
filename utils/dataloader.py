from torch.utils.data import DataLoader  # Provides iterables that wraps Dataset
from dataset import (
    MVTecDataset,
)

IMG_SIZE = 256
BATCH_SIZE = 16
MVTEC_CLASS = "tile"
PHASE = "train"

print(f"Loading MVTec {MVTEC_CLASS} dataset for {PHASE} phase...")
dataset = MVTecDataset(
    img_size=IMG_SIZE, apply_transform=True, cls=MVTEC_CLASS, phase=PHASE
)
print(f"Dataset loaded. Total samples: {len(dataset)}")

# Create an instance of the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True if PHASE == "train" else False,
    num_workers=0,
    drop_last=(
        True if PHASE == "train" else False
    ),  # drops last batch if num_samples < batch_size
)

for i, (images, masks, labels) in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(
        f"  Images shape: {images.shape}"
    )  # Expected: [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
    print(
        f"  Masks shape: {masks.shape}"
    )  # Expected: [BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE]
    print(f"  Labels: {labels}")  # Expected: [BATCH_SIZE]
