import torch

def train(model, optimizer, criterion, train_loader,num_epochs=1):
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            image, label = data
            label = torch.eye(6)[label]

            # Forward pass
            output = model(image)
            loss = criterion(output,label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                #print(label, output)