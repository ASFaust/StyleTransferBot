# Style Transfer Bot

Style Transfer Bot is a Python script that pulls memes from the ShitpostBot API, applies style transfer to them using Torch, CV2, and NumPy, and posts the stylized images on Facebook. 

## Installation

To run Style Transfer Bot, you'll need to install the following packages:

* [Python 3.x](https://www.python.org/downloads/)
* [Torch](https://pytorch.org/get-started/locally/)
* [CV2](https://pypi.org/project/opencv-python/)
* [NumPy](https://numpy.org/install/)

You'll also need a Facebook API access token, which you can obtain by following these steps:

1. Create a new Facebook App and obtain an App ID and Secret Key.
2. Generate a user access token using the App ID and Secret Key.
3. Save the user access token in the root directory of the project as `token`.

## Usage

To run Style Transfer Bot, execute the following command in your terminal:

`python3 run.py`

