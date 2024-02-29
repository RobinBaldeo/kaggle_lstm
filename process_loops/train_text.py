import torch


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for tokens, pos_tags, labels in train_loader:
        tokens, pos_tags, labels = tokens.to(device), pos_tags.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = model(tokens, pos_tags, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, valid_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for tokens, pos_tags, labels in valid_loader:
            tokens, pos_tags, labels = tokens.to(device), pos_tags.to(device), labels.to(device)
            loss = model(tokens, pos_tags, labels)
            total_loss += loss.item()
    return total_loss / len(valid_loader)


def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for tokens, pos_tags in data_loader:
            tokens, pos_tags = tokens.to(device), pos_tags.to(device)
            predictions = model(tokens, pos_tags)
            all_predictions.extend(predictions)
            # all_predictions.append(predictions)
    return all_predictions
