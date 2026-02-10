"""Training progress monitor script"""
import os
import glob

# Find the latest training directory
possible_paths = [
    'runs/detect/runs/train/baseline_e50',
    'runs/train/baseline_e50',
]

train_dir = None
for path in possible_paths:
    if os.path.exists(path):
        train_dir = path
        break

if not train_dir:
    train_dirs = glob.glob('runs/train/baseline_e50*') + glob.glob('runs/detect/runs/train/baseline_e50*')
    if train_dirs:
        train_dir = max(train_dirs, key=os.path.getmtime)

if not train_dir:
    print("Training directory not found")
    exit()

print(f"Training directory: {train_dir}")
print("=" * 60)

# Check results.csv
results_file = os.path.join(train_dir, 'results.csv')
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        lines = f.readlines()

    if len(lines) > 1:
        print("\nLatest training results:")
        print(lines[0].strip())  # Header
        for line in lines[-5:]:  # Last 5 epochs
            print(line.strip())
else:
    print("results.csv not generated yet")

# Check weights
weights_dir = os.path.join(train_dir, 'weights')
if os.path.exists(weights_dir):
    weights = os.listdir(weights_dir)
    if weights:
        print(f"\nSaved weights: {weights}")
        for w in weights:
            wpath = os.path.join(weights_dir, w)
            size_mb = os.path.getsize(wpath) / (1024 * 1024)
            print(f"  {w}: {size_mb:.2f} MB")
    else:
        print("\nNo weights saved yet")
else:
    print("\nWeights directory not created yet")

print("\n" + "=" * 60)
print("Note: Best weights will be saved to weights/best.pt")
