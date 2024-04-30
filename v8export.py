import torch
import torchvision
import tensorflow as tf

# Load PyTorch model
num_classes = "auto"  # Replace 10 with the actual number of output classes

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, num_classes)  # Replace num_classes with the number of output classes

# Load weights
weights_path = "./modelnamegoeshere.pt"
model.load_state_dict(torch.load(weights_path))
model.eval()

# Convert PyTorch model to TensorFlow format
dummy_input = torch.randn(1, 3, 320, 320)  # Example input shape (batch_size, channels, height, width)
tf_model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(3, 320, 320)), tf.keras.layers.Lambda(lambda x: model(x).detach().numpy())])
tf_model.save("tf_model")  # Save the TensorFlow model
