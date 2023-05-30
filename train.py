from torch import nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from neural_net_architecture import NeuralNet

writer = SummaryWriter()

# Path to your dataset
data_dir = 'dataset'

# Define the transforms to be applied to the images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),  # Resize the images to n*n pixels
    torchvision.transforms.ToTensor()  # Convert the images to tensors
])

# Load the data from the directory
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

# Split the data into training and validation sets
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = int(0.1 * len(dataset))

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
#
# Create a data loader for the training set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
# Create a data loader for the validation set
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
for batch_idx, (images, labels) in enumerate(val_loader):
    print(f"Batch {batch_idx+1}, images shape: {images.shape}, image label: {labels.shape}")



device = "cuda" if torch.cuda.is_available() else "cpu"

# for benchmark use a custom neural network model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = NeuralNet().to(device)


# model = NeuralNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-05)
#
# Define a summary writer for TensorBoard
writer = SummaryWriter()
num_epochs = 60
# Loop over the training data
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the training accuracy
        _, predicted = torch.max(outputs, 1)
        train_acc += (predicted == labels).sum().item()

        # Calculate the training loss
        train_loss += loss.item() * inputs.size(0)

    # Calculate the average training loss and accuracy
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)

    # Log the training loss and accuracy to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    # Loop over the validation data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Calculate the validation accuracy
            _, predicted = torch.max(outputs, 1)
            val_acc += (predicted == labels).sum().item()
            # Calculate the validation loss
            val_loss += loss.item() * inputs.size(0)

        # Calculate the average validation loss and accuracy
        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)

        # Log the validation loss and accuracy to TensorBoard
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

    # Print the training and validation loss and accuracy for each epoch
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, \
        Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

writer.close()

