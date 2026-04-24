# ✍️ Interactive MNIST Digit Recognizer

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio">
  <img src="https://img.shields.io/badge/Google%20Colab-%23F9AB00.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
</div>

<br>

An end-to-end computer vision pipeline featuring a custom 5-layer Convolutional Neural Network (CNN) trained from scratch on the MNIST dataset, served via a real-time, interactive web canvas. 

Draw a digit using your mouse or trackpad, and the PyTorch backend will evaluate and predict the number stroke-by-stroke.

## Live UI Demonstrations

The model is highly accurate and evaluates the canvas in real-time as you draw. Here are a few examples of the UI correctly predicting drawn digits with high confidence:

| Predicting '0' | Predicting '1' |
| :---: | :---: |
| <img width="1656" height="680" alt="Screenshot 2026-04-24 220252" src="https://github.com/user-attachments/assets/869028c9-2920-424a-aa3b-60e7d1f0017a" />|<img width="1649" height="631" alt="Screenshot 2026-04-24 220019" src="https://github.com/user-attachments/assets/4f437447-86e3-4072-98d4-dd2b95b43b5c" />|

| Predicting '4' | Predicting '5' |
| :---: | :---: |
| <img width="1668" height="695" alt="Screenshot 2026-04-24 220150" src="https://github.com/user-attachments/assets/8fcef277-0beb-4bbf-a77a-60be08e347fc" />|<img width="1675" height="691" alt="Screenshot 2026-04-24 220210" src="https://github.com/user-attachments/assets/dc3729d1-fb5a-467f-8379-b18c49459e9d" />|

| Predicting '6' | Predicting '9' |
| :---: | :---: |
| <img width="1683" height="648" alt="Screenshot 2026-04-24 220229" src="https://github.com/user-attachments/assets/f64c5ea3-c994-4da1-b9b9-aa29a8983394" />| <img width="1677" height="670" alt="Screenshot 2026-04-24 215928" src="https://github.com/user-attachments/assets/bed3c37b-ffdd-4ccc-9d2f-80b3ebfb1d78" />|

##  Key Features
* **Custom Architecture:** Built entirely from scratch using PyTorch `nn.Module` (no pre-trained weights).
* **Real-Time Inference:** Continuous evaluation of the drawing canvas without needing to press a "submit" button.
* **Automated Preprocessing:** Handles canvas-to-tensor transformations, including dynamic resizing, grayscale conversion, color inversion, and standard deviation normalization.
* **Granular Metrics:** Tracks training loss and evaluates precision, recall, F1-scores, and confusion matrices.

##  Model Architecture
The network is a lightweight 5-layer CNN designed for rapid feature extraction and classification:

1. **Conv2D (32 filters, 3x3)** + ReLU + MaxPool2D
2. **Conv2D (64 filters, 3x3)** + ReLU + MaxPool2D
3. **Conv2D (64 filters, 3x3)** + ReLU
4. **Dense/Linear (64 units)** + ReLU
5. **Dense/Linear Output (10 units)** -> Maps to digits 0-9.

## Training & Performance Metrics

The model was trained for **15 epochs**, achieving excellent convergence and a highly accurate test set performance. 

| Training Loss | Confusion Matrix |
| :---: | :---: |
| <img width="851" height="575" alt="Screenshot 2026-04-24 220338" src="https://github.com/user-attachments/assets/2fc52a4e-d162-40ed-a5ec-c7938d2bcbc3" />|<img width="809" height="613" alt="Screenshot 2026-04-24 220354" src="https://github.com/user-attachments/assets/55f5a703-31d8-45a9-b76a-46314e6d17e3" />|
| *Smooth convergence approaching near-zero cross-entropy loss over 15 epochs.* | *Strong diagonal indicating high true-positive rates across all 10 digits.* |

## How to Run Locally

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch torchvision gradio numpy pillow matplotlib scikit-learn seaborn
```
1. Train the Model
Run the training script to download the MNIST data, train the CNN, and save the model weights to mnist_custom_cnn.pth.

```bash
python train.py
```
2. Launch the Web UI
Start the Gradio server. It will load the saved .pth weights and open a local port.

```bash
python app.py
```
Visit http://127.0.0.1:7860 in your browser to interact with the model.

☁️ Running in Google Colab
If you prefer not to install anything locally, you can run the entire pipeline in Google Colab:

Open the provided notebook.

Go to Runtime > Change runtime type and select T4 GPU.

Run all cells. The final cell will generate a public gradio.live link that you can open on your phone or share with friends.
