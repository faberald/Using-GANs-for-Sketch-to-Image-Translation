import matplotlib.pyplot as plt
import csv

['epoch', 'i', 'loss_D_VAE', 'loss_D_LR', 'loss_GE', 'loss_pixel', 'loss_kl', 'loss_latent']
with open('loss.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    
print(data[0])

loss_D_VAE = []
loss_D_LR = []
loss_GE = []
loss_pixel = []
loss_kl = []
loss_latent = []
for i in data:
    # loss_D_VAE_1 = 0
    # loss_D_LR_1 = 0
    # loss_GE_1 = 0
    # loss_pixel_1 = 0
    # loss_kl_1 = 0
    # loss_latent_1 = 0 
    # for k in range(779):
    #     loss_D_VAE_1 += float(i[2])
    #     loss_D_LR_1 += float(i[3])
    #     loss_GE_1 += float(i[4])
    #     loss_pixel_1 += float(i[5])
    #     loss_kl_1 += float(i[6])
    #     loss_latent_1 += float(i[7])
    loss_D_VAE.append(float(i[2]))
    loss_D_LR.append(float(i[3]))
    loss_GE.append(float(i[4]))
    loss_pixel.append(float(i[5]))
    loss_kl.append(float(i[6]))
    loss_latent.append(float(i[7]))
        
plt.suptitle('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss_latent)
plt.show()