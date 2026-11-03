from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = val_losses[0]
    best_epoch = 0
    count = 0
    stop_epoch = len(val_losses) - 1
    for epoch in range(1, len(val_losses)):
        curr_loss = val_losses[epoch]
        if best_loss - curr_loss > min_delta:
            best_loss = curr_loss
            best_epoch = epoch
            count = 0
        else:
            count += 1
        if count >= patience:
            stop_epoch = epoch
            break
    return stop_epoch, best_epoch