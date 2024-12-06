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
from tqdm import tqdm
import time

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
    print(f"\n[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}\n")

    print("[STEP 1/5] Initializing model...")
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loading
    print("[STEP 2/5] Preparing datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)

    print(f"[INFO] Total training batches: {len(train_loader)}")
    print(f"[INFO] Batch size: 512")
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}\n")

    # Training loop
    num_epochs = 10
    print("[STEP 3/5] Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
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
            
            # Update progress bar every batch
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'accuracy': f'{accuracy:.2f}%'
            })
            
            # Update web interface every 10 batches
            if batch_idx % 10 == 0:
                state.logs['loss'].append(running_loss / (batch_idx + 1))
                state.logs['accuracy'].append(accuracy)

    training_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {training_time:.2f} seconds")

    print("\n[STEP 4/5] Evaluating model...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    final_accuracy = 100. * test_correct / test_total
    print(f"\n[INFO] Final Test Accuracy: {final_accuracy:.2f}%")

    print("\n[STEP 5/5] Generating test samples...")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(10), desc='Generating samples'):
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
    print("\n[INFO] Test samples generated. You can view the results in the web interface.")

def main():
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(port=5000, debug=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start training in the main thread
    train_model()

if __name__ == '__main__':
    main()