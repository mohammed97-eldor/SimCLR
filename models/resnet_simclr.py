import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):

    def __init__(self, size, device="cpu", embedding_dim=64):
        super(ResNetSimCLR, self).__init__()

        self.embedding_dim = embedding_dim

        if size == 18:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) 
        elif size == 50:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif size == 101:
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            print("The passed model size is nopt available, therefore irt will be set to the defaults value and ResNet50 will be used!")
            self.model = models.resnet50(pretrained=False)

        self.dim_mlp = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.dim_mlp, 1000),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(1000, self.embedding_dim),
                                       torch.nn.ReLU())
        if device == "cpu" or not torch.cuda.is_available():
            if device != "cpu":
                print(f"device specified ({device}) is not accessible, check if cuda was written correctly, otherwise cuda is not available")
            self.device = "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            print("GPU is being utilized")
        else:
            print("Unexpected error occured when choosing the device therefore GPU will be used if available and CPU otherwise")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print("ResNet18:")
    print(ResNetSimCLR(18), "\n\n")
    print("----------------------------------------------------")
    print("ResNet50:")
    print(ResNetSimCLR(50), "\n\n")
    print("----------------------------------------------------")
    print("ResNet101:")
    print(ResNetSimCLR(101), "\n\n")
