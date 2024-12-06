# MNIST CNN Training with Real-time Monitoring

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization using a Flask web server.

## Requirements

1. Install the required packages:
```
bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the CNN architecture
- `train.py`: Training script with Flask server integration
- `templates/monitor.html`: Web interface for monitoring training
- `static/style.css`: Styling for the web interface

## How to Run

1. Make sure all project files are in the correct directory structure
2. Run the training script: 
```
bash
python train.py
```

3. Open a web browser and navigate to `http://localhost:5000`
4. The training progress will be displayed in real-time
5. After training completes, 10 random test images with predictions will be shown

## Features

- Real-time loss and accuracy visualization
- CUDA support for GPU acceleration
- Web-based monitoring interface
- Random sample testing with visualization
- Automatic data downloading and preprocessing

## Notes

- Training progress is updated every 100 batches
- The web interface refreshes automatically every 5 seconds
- Test results are displayed after training completion
- The model uses CUDA if available, otherwise falls back to CPU