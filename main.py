# pytorch version
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import FFNN

# SETTINGS
BATCH_SIZE = 64
EPOCHS = 10
MODEL_SHAPE = [512, 512]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """
    main function
    """
    
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    print(DEVICE)

    model = FFNN().to(DEVICE)
    print(f"device: {DEVICE}")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-----------------------------------")
        train(train_dataloader, model, loss_fn=loss_fn, optimizer=optimizer)
        test(test_dataloader, model=model, loss_fn=loss_fn)

    print('done')

    torch.save(model.state_dict(), "model.pth")
    print("saved")

def train(dataloader: DataLoader, model: FFNN, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test(dataloader:DataLoader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

main()
