
import random
from minigrad.engine import Value
from minigrad.neural_net import MLP
import matplotlib.pyplot as plt
#Test data
xs = [
    [2.0, 3.0, 1.0],
    [3.0, 1.0, 0.5],
    [0.5, 3.0, 1.0],
    [3.0, 1.0, 0.05],
]
xs = [[Value(v) for v in x] for x in xs]
ys = [Value(1.0), Value(0.0), Value(1.0), Value(0.0)]  # Convert to Value objects

#Random seed for reproducibility
random.seed(42)  # for reproducibility

# Training parameters
epochs = 50  # Increased epochs for better convergence
learning_rate = 0.5  # Reduced learning rate

# Initialize model
model = MLP(3, [4, 2, 1])
loss_plot = []
print("Training started...")
for epoch in range(epochs):
    # Zero gradients once per epoch
    model.zero_grad()
    
    # Forward pass for all samples and accumulate loss
    total_loss = Value(0.0)
    predictions = []
    
    for x, ygt in zip(xs, ys):
        ypred = model(x)
        predictions.append(ypred)
        loss = (ypred - ygt) ** 2
        total_loss = total_loss + loss
    
    # Average loss
    avg_loss = total_loss * (1.0 / len(xs))
    
    loss_plot.append(avg_loss.data)
    # Backward pass
    avg_loss.backward()
    
    # Update parameters
    for p in model.parameters():
        p.data += -learning_rate * p.grad
    
    # Print progress
    if (epoch + 1) % 20 == 0 or epoch < 10:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss.data:.6f}")

print("\nTraining completed!")
print("\nFinal predictions:")
for i, (x, ygt) in enumerate(zip(xs, ys)):
    pred = model(x)
    print(f"Sample {i+1}: Target={ygt.data:.1f}, Predicted={pred.data:.4f}")

# Test with a new sample
print(f"\nTest sample [3.0, 1.0, 0.5]: {model(xs[1]).data:.4f}")

# Plotting the loss over epochs
epochs_range = list(range(1, epochs + 1))
plt.plot(epochs_range, loss_plot);plt.grid()
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('test_training/loss_over_epochs.png', dpi=300, bbox_inches='tight')
plt.show()
