import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .loss import BCEDiceLoss
from .dataset import CCPDatasetWrapper, visualize_generated_data
from .model import UNetPlusPlus
from .config import DEVICE, BATCH_SIZE, TRAIN_SAMPLES, VAL_SAMPLES, MIN_CELLS, MAX_CELLS, PATCH_SIZE, SIM_CONFIG, DETECTION_MODEL, SAM3_CHECKPOINT, SAM3_MODEL_TYPE
from .utils import patch_dataloader

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, weight_decay=1e-4, patience=5, device='cuda'):
    model = model.to(device)
    criterion = BCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    print(f"Training U-Net++ with IoU Loss for {epochs} epochs")
    print(f"Device: {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                pred = torch.sigmoid(outputs) > 0.5
                gt = masks > 0.5
                intersection = (pred * gt).sum()
                union = pred.sum() + gt.sum()
                dice = (2. * intersection / (union + 1e-6)).item()
                val_dice += dice
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best model (val_loss: {val_loss:.4f})")
            torch.save(model.state_dict(), "best_model.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    return model, history

def train_unet_pipeline(train_samples=TRAIN_SAMPLES, val_samples=VAL_SAMPLES, epochs=20, batch_size=BATCH_SIZE, learning_rate=1e-3, weight_decay=1e-4, patience=5, device=DEVICE):
    # Patch DataLoader to fix RecursionError
    patch_dataloader()
    
    print("Training Configuration:")
    print(f"  TRAIN_SAMPLES: {train_samples}")
    print(f"  VAL_SAMPLES: {val_samples}")
    print(f"  EPOCHS: {epochs}")
    print(f"  BATCH_SIZE: {batch_size}")
    print(f"  LEARNING_RATE: {learning_rate}")
    print(f"  WEIGHT_DECAY: {weight_decay}")
    print(f"  PATIENCE: {patience}")
    print(f"  MIN_CELLS: {MIN_CELLS}")
    print(f"  MAX_CELLS: {MAX_CELLS}")
    print(f"  PATCH_SIZE: {PATCH_SIZE}")
    print(f"  SIM_CONFIG: {SIM_CONFIG}")
    
    print("Creating train dataset...")
    train_dataset = CCPDatasetWrapper(length=train_samples, min_n=MIN_CELLS, max_n=MAX_CELLS, patch_size=PATCH_SIZE, sim_config=SIM_CONFIG)
    print("Creating validation dataset...")
    val_dataset = CCPDatasetWrapper(length=val_samples, min_n=MIN_CELLS, max_n=MAX_CELLS, patch_size=PATCH_SIZE, sim_config=SIM_CONFIG)
    
    # Visualize data before training (sanity check)
    print("Visualizing generated data samples...")
    try:
        visualize_generated_data(train_dataset, num_samples=4)
    except Exception as e:
        print(f"Could not visualize data: {e}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print("Creating U-Net++ model...")
    model = UNetPlusPlus(in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], use_attention=True, dropout_rate=0.1)
    
    model, history = train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, weight_decay=weight_decay, patience=patience, device=device)
    return model, history
