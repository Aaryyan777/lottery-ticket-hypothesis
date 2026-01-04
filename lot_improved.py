import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Model Definition (LeNet-300-100)
# ==========================================
class LotteryMLP(nn.Module):
    def __init__(self):
        super(LotteryMLP, self).__init__()
        # Standard LeNet-300-100 architecture
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 2. Data Loading
# ==========================================
def get_mnist_loaders(batch_size=60, test_batch_size=1000):
    """Downloads and returns MNIST train and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std dev of MNIST
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader

# ==========================================
# 3. Pruning Utilities (Global Pruning)
# ==========================================
def get_global_mask(model, pruning_rate=0.2, current_mask=None):
    """
    Creates a binary mask for the model parameters based on GLOBAL magnitude.
    Prunes 'pruning_rate' (e.g., 20%) of the smallest weights across ALL layers.
    """
    all_scores = []
    
    # 1. Gather all "alive" weights to find the global threshold
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if current_mask:
                # Only consider currently active weights for the percentile
                tensor = tensor[current_mask[name].cpu().numpy() == 1]
            all_scores.append(np.abs(tensor.flatten()))
    
    all_scores = np.concatenate(all_scores)
    # Calculate threshold on the surviving weights
    threshold = np.percentile(all_scores, pruning_rate * 100)
    
    new_mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            
            # Keep weights greater than or equal to threshold
            mask_val = np.where(np.abs(tensor) < threshold, 0, 1)
            
            # Ensure we respect the previous mask (once pruned, always pruned)
            if current_mask:
                 mask_val = mask_val * current_mask[name].cpu().numpy()
                 
            # CAST TO FLOAT32 TO AVOID DOUBLE/FLOAT MISMATCH
            new_mask[name] = torch.from_numpy(mask_val).float().to(param.device)
            
    return new_mask

def apply_mask(model, mask):
    """
    Multiplies the model weights by the mask to ensure pruned weights stay zero.
    """
    for name, param in model.named_parameters():
        if name in mask:
            param.data = param.data * mask[name]

def measure_sparsity(model, mask=None):
    """Returns the percentage of non-zero weights in the model."""
    all_weights = 0
    non_zero_weights = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            size = tensor.size
            all_weights += size
            
            if mask and name in mask:
                non_zero = torch.sum(mask[name]).item()
            else:
                non_zero = np.count_nonzero(tensor)
            
            non_zero_weights += non_zero
            
    return (non_zero_weights / all_weights) * 100

# ==========================================
# 4. Training Engine
# ==========================================
def train(model, device, train_loader, optimizer, criterion, epoch, mask=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Zero out gradients for pruned weights
        if mask:
             for name, param in model.named_parameters():
                if name in mask:
                    param.grad *= mask[name]

        optimizer.step()
        
        # Force weights to 0 (Double safety)
        if mask:
            apply_mask(model, mask)
            
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# ==========================================
# 5. Main Experiment Loop
# ==========================================
def run_experiment(epochs=20, rounds=20, prune_rate=0.2, batch_size=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: {epochs} Epochs | {rounds} Rounds | Global Pruning | SGD")

    # Data Loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # Initialize Model and save Initial Weights (W0)
    model = LotteryMLP().to(device)
    initial_state_dict = copy.deepcopy(model.state_dict())

    results = {
        "sparsity": [],
        "winning_ticket_acc": [],
        "random_init_acc": []
    }

    # Initial Mask (All 1s)
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = torch.ones_like(param).to(device)

    print(f"{'Round':<5} | {'Sparsity (%)':<12} | {'Win Ticket Acc':<15} | {'Random Init Acc':<15}")
    print("-" * 55)

    for round_idx in range(rounds + 1):
        # --- 1. Measure Sparsity ---
        sparsity = measure_sparsity(model, mask)
        results["sparsity"].append(sparsity)

        # --- 2. Train Winning Ticket (Reset to W0) ---
        model.load_state_dict(initial_state_dict) # Reset to W0
        apply_mask(model, mask) # Apply current mask
        
        # Using SGD with momentum as per LTH paper standard
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch, mask)
        
        _, wt_acc = test(model, device, test_loader, criterion)
        results["winning_ticket_acc"].append(wt_acc)

        # Preserve the trained weights for calculating the NEXT mask
        trained_weights = copy.deepcopy(model.state_dict())

        # --- 3. Train Control (Random Init) ---
        # Re-initialize with new random weights
        random_model = LotteryMLP().to(device)
        apply_mask(random_model, mask) # Apply SAME mask
        
        optimizer_rand = optim.SGD(random_model.parameters(), lr=0.01, momentum=0.9)
        
        for epoch in range(1, epochs + 1):
            train(random_model, device, train_loader, optimizer_rand, criterion, epoch, mask)
            
        _, rand_acc = test(random_model, device, test_loader, criterion)
        results["random_init_acc"].append(rand_acc)

        print(f"{round_idx:<5} | {sparsity:<12.2f} | {wt_acc:<15.2f} | {rand_acc:<15.2f}")

        # --- 4. Prune (Calculate Mask for NEXT round) ---
        # Calculate next mask based on the TRAINED weights of the winning ticket
        model.load_state_dict(trained_weights)
        if round_idx < rounds:
            mask = get_global_mask(model, pruning_rate=prune_rate, current_mask=mask)
                
    return results

# ==========================================
# 6. Visualization
# ==========================================
def plot_results(results):
    sparsities = results["sparsity"]
    wt_acc = results["winning_ticket_acc"]
    rand_acc = results["random_init_acc"]
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    plt.plot(sparsities, wt_acc, marker='o', label='Winning Ticket (Rewind to Init)', color='b', linewidth=2)
    plt.plot(sparsities, rand_acc, marker='s', label='Random Re-init (Control)', color='r', linestyle='--', linewidth=2)

    plt.title("Lottery Ticket Hypothesis: MNIST LeNet-300-100 (Improved)", fontsize=16)
    plt.xlabel("Weights Remaining (%)", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.gca().invert_xaxis() # 100% -> 0.1%
    
    ticks = [100, 50, 20, 10, 5, 2, 1]
    plt.xticks(ticks, ticks)
    plt.minorticks_off()
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('graph_improved.png') # Save plot automatically
    print("Graph saved to graph_improved.png")
    plt.show()

if __name__ == "__main__":
    results = run_experiment(epochs=20, rounds=20, prune_rate=0.2)
    plot_results(results)