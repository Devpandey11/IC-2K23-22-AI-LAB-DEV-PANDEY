"""Lab 7: PCA/SVD, MNIST mini-project, FGSM and Grad-CAM.
Run on a laptop/Google Colab; MNIST downloads automatically on first run.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PCA and SVD demonstration on a small numeric matrix.
matrix = np.array([[2., 0.], [0., 2.], [3., 1.], [1., 3.]])
centered = matrix - matrix.mean(axis=0)
u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
print("SVD singular values:", singular_values)
pca = PCA(n_components=1)
print("PCA transformed data:\n", pca.fit_transform(matrix))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(x.flatten(1))

model = MNISTCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Mini-project training: one epoch to keep the practical demo short.
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
print("Completed one training epoch.")

model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    logits = model(images)
    print("Test batch accuracy:", (logits.argmax(1) == labels).float().mean().item())

# FGSM adversarial example: x_adv = x + epsilon * sign(gradient).
sample = images[:1].clone().detach().requires_grad_(True)
target = labels[:1]
model.zero_grad()
criterion(model(sample), target).backward()
epsilon = 0.15
adv = (sample + epsilon * sample.grad.sign()).clamp(0, 1).detach()
with torch.no_grad():
    print("Original prediction:", model(sample.detach()).argmax(1).item())
    print("FGSM prediction:", model(adv).argmax(1).item())

# Grad-CAM for the final convolution layer.
activations, gradients = [], []
def save_activation(module, inp, out):
    activations.append(out)
    out.register_hook(lambda grad: gradients.append(grad))

hook = model.conv.register_forward_hook(save_activation)
model.zero_grad()
out = model(sample.detach())
class_id = out.argmax(1).item()
model.zero_grad()
out[0, class_id].backward()
activation = activations[-1].detach()
gradient = gradients[-1].detach()
weights = gradient.mean(dim=(2, 3), keepdim=True)
cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
cam = torch.nn.functional.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
cam = cam[0, 0].cpu().numpy()
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
axes[0].imshow(sample.detach().cpu()[0, 0], cmap="gray"); axes[0].set_title("Original")
axes[1].imshow(adv.cpu()[0, 0], cmap="gray"); axes[1].set_title("FGSM")
axes[2].imshow(sample.detach().cpu()[0, 0], cmap="gray")
axes[2].imshow(cam, cmap="jet", alpha=0.45); axes[2].set_title("Grad-CAM")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig("lab7_results.png")
plt.show()
hook.remove()
