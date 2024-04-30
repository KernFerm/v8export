# PyTorch to TensorFlow.js Exporter

This tool converts a PyTorch model (.pt) to TensorFlow.js format.

# -> For Nvidia GPU ONLY!!! <-

## Installation

1. Install dependencies:

    ```bash
    pip install torch torchvision tensorflow
    ```

2. Clone the repository:

    ```bash
    git clone https://github.com/KernFerm/v8export.git
    ```

3. Navigate to the project directory:

    ```bash
    cd your_repo
    ```

## Usage

Run the Python script `export.py` to convert the PyTorch model to TensorFlow format and then convert it to TensorFlow.js format.

```bash```
python export.py --weights ./modelnamegoeshere.pt --include engine --half --img-size 320 320 --device 0



# Conversion Process
- Load PyTorch Model: The Python script loads the PyTorch model using torchvision and torch.
- Convert to TensorFlow: The PyTorch model is converted to TensorFlow format using TensorFlow's APIs.
- Save TensorFlow Model: The converted TensorFlow model is saved in the current directory.
- Convert to TensorFlow.js: Finally, the tfjs_converter tool is used to convert the TensorFlow model to TensorFlow.js format.


---


# Contributing
- Contributions are welcome! Please create a pull request with your changes.