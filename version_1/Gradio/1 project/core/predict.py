import torch

class ImageClassifier():
    def __init__(self):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # define CNN architecture
        # load CNN architecture trained weights
        # index to label map
        # transformation
        pass
    def predict(self):
        # load image with PILLOW
        # prediction
        # label map -->  class
        # oepncv, write text on our input image
        # return class, output image
        pass
