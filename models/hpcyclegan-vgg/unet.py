import torch
import torch.nn as nn
from torchvision.models import resnet18

class UnetGenerator(nn.Module):
    def __init__(self, norm_layer = nn.BatchNorm2d):
        super(UnetGenerator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1)

        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
        self.norm2 = norm_layer(128)

        self.relu3 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
        self.norm3 = norm_layer(256)

        self.relu4 = nn.LeakyReLU(0.2, True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm4 = norm_layer(512)

        self.relu5 = nn.LeakyReLU(0.2, True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm5 = norm_layer(512)

        self.relu6 = nn.LeakyReLU(0.2, True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm6 = norm_layer(512)

        self.relu7 = nn.LeakyReLU(0.2, True)
        self.conv7 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm7 = norm_layer(512)

        self.relu8 = nn.LeakyReLU(0.2, True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
        self.relu9 = nn.ReLU(True)
        self.conv9 = nn.ConvTranspose2d(1024, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm9 = norm_layer(512)

        self.relu10 = nn.ReLU(True)
        self.conv10 = nn.ConvTranspose2d(1024, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm10 = norm_layer(512)

        self.relu11 = nn.ReLU(True)
        self.conv11 = nn.ConvTranspose2d(1024, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm11 = norm_layer(512)

        self.relu12 = nn.ReLU(True)
        self.conv12 = nn.ConvTranspose2d(1024, 512, kernel_size = 3, stride = 2, padding = 1)
        self.norm12 = norm_layer(512)

        self.relu13 = nn.ReLU(True)
        self.conv13 = nn.ConvTranspose2d(1024, 256, kernel_size = 3, stride = 2, padding = 1)
        self.norm13 = norm_layer(256)

        self.relu14 = nn.ReLU(True)
        self.conv14 = nn.ConvTranspose2d(512, 128, kernel_size = 3, stride = 2, padding = 1)
        self.norm14 = norm_layer(128)

        self.relu15 = nn.ReLU(True)
        self.conv15 = nn.ConvTranspose2d(256, 64, kernel_size = 3, stride = 2, padding = 1)
        self.norm15 = norm_layer(64)

        self.relu16 = nn.ReLU(True)
        self.conv16 = nn.ConvTranspose2d(128, 1, kernel_size = 3, stride = 2, padding = 1)
        self.tanh16 = nn.Tanh()

    def forward(self, x, parsing_feature):

        x = self.conv1(x)
        temp_1 = x

        x = self.relu2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        temp_2 = x

        x = self.relu3(x)
        x = self.conv3(x)
        x = self.norm3(x)
        temp_3 = x

        x = self.relu4(x)
        x = self.conv4(x)
        x = self.norm4(x)
        temp_4 = x

        x = self.relu5(x)
        x = self.conv5(x)
        x = self.norm5(x)
        temp_5 = x

        x = self.relu6(x)
        x = self.conv6(x)
        x = self.norm6(x)
        temp_6 = x

        x = self.relu7(x)
        x = self.conv7(x)
        x = self.norm7(x)
        temp_7 = x

        x = self.relu8(x)
        x = self.conv8(x)
        x = torch.cat([x, parsing_feature], 1)
        x = self.relu9(x)
        x = self.conv9(x)
        x = self.norm9(x)
        x = torch.cat([x, temp_7], 1)

        x = self.relu10(x)
        x = self.conv10(x)
        x = self.norm10(x)
        x = torch.cat([x, temp_6], 1)

        x = self.relu11(x)
        x = self.conv11(x)
        x = self.norm11(x)
        x = torch.cat([x, temp_5], 1)

        x = self.relu12(x)
        x = self.conv12(x)
        x = self.norm12(x)
        x = torch.cat([x, temp_4], 1)

        x = self.relu13(x)
        x = self.conv13(x)
        x = self.norm13(x)
        x = torch.cat([x, temp_3], 1)

        x = self.relu14(x)
        x = self.conv14(x)
        x = self.norm14(x)
        x = torch.cat([x, temp_2], 1)

        x = self.relu15(x)
        x = self.conv15(x)
        x = self.norm15(x)
        x = torch.cat([x, temp_1], 1)

        x = self.relu16(x)
        x = self.conv16(x)
        x = self.tanh16(x)

        return x

class UnetEncoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(UnetEncoder, self).__init__()
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar
# class UnetEncoder(nn.Module):
#     def __init__(self, norm_layer = nn.BatchNorm2d):
#         super(UnetEncoder, self).__init__()
    
#         self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1)
        
#         self.relu2 = nn.LeakyReLU(0.2, True)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
#         self.norm2 = norm_layer(128)
        
#         self.relu3 = nn.LeakyReLU(0.2, True)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
#         self.norm3 = norm_layer(256)
        
#         self.relu4 = nn.LeakyReLU(0.2, True)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1)
#         self.norm4 = norm_layer(512)
        
#         self.relu5 = nn.LeakyReLU(0.2, True)
#         self.conv5 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
#         self.norm5 = norm_layer(512)
        
#         self.relu6 = nn.LeakyReLU(0.2, True)
#         self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
#         self.norm6 = norm_layer(512)
        
#         self.relu7 = nn.LeakyReLU(0.2, True)
#         self.conv7 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
#         self.norm7 = norm_layer(512)
        
#         self.relu8 = nn.LeakyReLU(0.2, True)
#         self.conv8 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1)
    
#     def forward(self, x):
    
#         x = self.conv1(x)
#         x = self.relu2(x)
#         x = self.conv2(x)
#         x = self.norm2(x)
        
#         x = self.relu3(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
        
#         x = self.relu4(x)
#         x = self.conv4(x)
#         x = self.norm4(x)
        
#         x = self.relu5(x)
#         x = self.conv5(x)
#         x = self.norm5(x)
        
#         x = self.relu6(x)
#         x = self.conv6(x)
#         x = self.norm6(x)
        
#         x = self.relu7(x)
#         x = self.conv7(x)
#         x = self.norm7(x)
        
#         x = self.relu8(x)
#         x = self.conv8(x)
        
#         return x