import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from flask import Flask, render_template, jsonify
import threading
import random
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

# Create directories if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variables for sharing data between threads
class TrainingState:
    def __init__(self):
        self.logs = {'loss': [], 'accuracy': []}
        self.is_training_complete = False
        self.test_samples = []

state = TrainingState()

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/get_logs')
def get_logs():
    return jsonify({
        'logs': state.logs,
        'training_complete': state.is_training_complete,
        'test_samples': state.test_samples
    })

# CNN Model definition
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_model():
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Training loop
    num_epochs = 10
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                accuracy = 100. * correct / total
                state.logs['loss'].append(running_loss / (batch_idx + 1))
                state.logs['accuracy'].append(accuracy)
                
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {running_loss/(batch_idx+1):.3f}, Accuracy: {accuracy:.2f}%')

    print("Training complete. Generating test samples...")
    
    # Test on random samples
    model.eval()
    with torch.no_grad():
        for i in range(10):
            idx = random.randint(0, len(test_dataset)-1)
            img, label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(dim=1).item()
            
            plt.figure(figsize=(2,2))
            plt.imshow(img.cpu().squeeze(), cmap='gray')
            plt.title(f'Pred: {pred}, True: {label}')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            state.test_samples.append({
                'image': base64.b64encode(image_png).decode(),
                'pred': pred,
                'true': label.item()
            })
    
    state.is_training_complete = True
    print("Test samples generated. You can view the results in the web interface.")

def main():
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(port=5000, debug=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start training in the main thread
    train_model()

if __name__ == '__main__':
    main()