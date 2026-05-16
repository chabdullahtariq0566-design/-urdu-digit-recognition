# Urdu Handwritten Digits (0–5) Recognition

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy     | 86.51% |
| Best Val Accuracy | 90.40% |

## Architecture
```
Input    : 784 neurons (28x28 flattened)
Hidden 1 : 128 neurons + ReLU
Hidden 2 :  64 neurons + ReLU
Hidden 3 :  32 neurons + ReLU
Output   :   6 neurons (digits 0-5)
```

## Training
- Optimizer : SGD (lr=0.01, momentum=0.9)
- Loss      : CrossEntropyLoss
- Epochs    : 50
- Batch     : 32

## Quick Start
```python
import torch
model = torch.load('urdu_digit_3layer_full.pth')
model.eval()
with torch.no_grad():
    out   = model(image_tensor)   # image_tensor shape: (1, 784)
    digit = out.argmax(1).item()
    print(f'Predicted: {digit}')
```

## Files
| File | Description |
|------|-------------|
| `urdu_digit_3layer_model.pth` | Model weights (state_dict) |
| `urdu_digit_3layer_full.pth`  | Complete model with architecture |
| `training_history.json`       | Loss and accuracy per epoch |
| `notebook.ipynb`              | Full training notebook |

## Author
[chabdullahtariq0566-design](https://github.com/chabdullahtariq0566-design)
