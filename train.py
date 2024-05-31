import torch
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, save_path='model_d.pth'):
    model = model.to(device)
    writer = SummaryWriter('logs/train_d')
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, writer, epoch)
        print(f'Epoch {epoch + 1}, Test Accuracy: {test_accuracy}%')

        # 保存模型如果测试准确率有提高
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path} with accuracy {test_accuracy}%')

    writer.close()