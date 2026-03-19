__author__ = 'Eric'

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import netron
from collections import OrderedDict

#=============================加载数据==================================

#查看当前环境是否有GPU，有则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#定义几个超参数
random_seed =456
learning_rate = 0.005
num_epoch = 1
batch_size =256

train_dataset = datasets.FashionMNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
#train_dataset = datasets.FashionMNIST(root='../data',train=True,transform=transforms.ToTensor(),download=False)
test_dataset = datasets.FashionMNIST(root='./data',train=False,transform=transforms.ToTensor())

#train_loder = DataLoader(dataset = train_dataset,batch_size= batch_size,shuffle= True)
#test_loder = DataLoader(datasets = test_dataset,batch_size=batch_size,shuffle=False)

m = len(train_dataset)

#把数据划分为训练数据和验证数据
train_data,val_data=random_split(train_dataset,[int(m-m*0.2),int(m*0.2)])

#The dataloders handle shuffing banching, etc..
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset = val_data,batch_size= batch_size)
test_loader = DataLoader(dataset = test_dataset,batch_size=batch_size)


#=============================1、网络构建==================================
#=============================构建Encoder==================================
class Encoder(nn.Module):
    def __init__(self,encoded_space_dim,fc2_input_dim):
        super().__init__()

        #构建Encoder
        #构建卷积网络
        self.encoder_cnn = nn.Sequential(
            #第一个卷积层
            nn.Conv2d(1,8,3,stride=2,padding=1),#inchannel,outchannel,kernelsize
            nn.ReLU(True),
            #第二个卷积层
            nn.Conv2d(8,16,3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #第三个卷积层
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU(True)
        )

        #定义一个展平层
        self.flatten = nn.Flatten(start_dim=1)

        #全连接网络
        self.encoder_lin = nn.Sequential(
            #第一个全连接层
            nn.Linear(3*3*32,fc2_input_dim),
            nn.ReLU(True),
            #第二个全连接层
            nn.Linear(fc2_input_dim,encoded_space_dim)

        )

    def forward(self,x):
        #图像输入到卷积网络
        x = self.encoder_cnn(x)
        #展平卷积网络的输出
        x = self.flatten(x)
        #使用全连接层
        x = self.encoder_lin(x)
        return x


#=============================构建Decoder==================================
class Decoder(nn.Module):
    def __init__(self,encoded_space_dim,fc2_input_dim):
        super().__init__()

        #定义全连接模块
        self.decoder_lin = nn.Sequential(
            #第一个全连接层
            nn.Linear(encoded_space_dim,fc2_input_dim),
            nn.ReLU(True),
            #第二个全连接层
            nn.Linear(fc2_input_dim,3*3*32),
            nn.ReLU(True)

        )

        #反展平
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32,3,3))

        #定义有多个转置卷积构成的模块
        self.decoder_conv = nn.Sequential(
            #第一个转置卷积层
            nn.ConvTranspose2d(32,16,3,stride=2,output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #第二个转置卷积层
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #第三个转置卷积层
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1,output_padding=1)

        )



    def forward(self,x):
        #应用全链接
        x = self.decoder_lin(x)
        #反展平
        x = self.unflatten(x)
        #应用转置卷积模块
        x = self.decoder_conv(x)
        #使用sigmoid激活函数，使得输出值在0-1之间
        x = torch.sigmoid(x)
        return x





#=============================检查模型结构==================================
#设置随机种子
torch.manual_seed(0)

#初始化编码器和解码器
d = 2

encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)

from torchsummary import summary
summary(encoder,(1,28,28),device="cpu")
summary(decoder,(2,),device="cpu")




#=============================定义损失函数==================================
loss_fn = torch.nn.MSELoss()

#定义学习率、优化器
lr = 0.001

params_to_optimize = [
    {'params' : encoder.parameters()} ,
    {'params' : decoder.parameters()}

]

optim = torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)

#检测是否有GPU设备，有则使用GPU，否则使用CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device:{device}')

#把实例化的编码器和解码器移动到指定位置上
encoder.to(device)
decoder.to(device)


#=============================2、模型训练与测试==================================
#=============================定义训练模型的函数==================================
def train_epoch(encoder,decoder,device,dataloder,loss_fn,optimizer):
    #Set train mode for both encoder and decoder
    encoder.train()
    decoder.train()
    train_loss = []

    #Iterate the dataloder(we don't need the label values, this is unsupervised learning)
    for image_batch,_ in dataloder:#with '_' we just ignore the labels(zhe second element of the dataloder tuple)
        #move tensor to the proper device
        image_batch = image_batch.to(device)
        #Encode data
        encoded_data = encoder(image_batch)
        #Decode data
        decoded_data = decoder(encoded_data)
        #Evaluates loss
        loss = loss_fn(decoded_data,image_batch)
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Print batch loss
        print('\t partical train loss(single batch): %f'%(loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

#=============================定义测试模型的函数==================================
def test_epoch(encoder,decoder,device,dataloader,loss_fn):
    #Set evaluation mode for both encoder and decoder
    encoder.eval()
    decoder.eval()

    with torch.no_grad():#no need to track the gradients
        #define the lists to store the output for each batch
        conc_out = []
        conc_label = []
        for image_batch,_ in dataloader:
            #Move tensor to the proper device
            image_batch = image_batch.to(device)
            #Encoder data
            encoded_data = encoder(image_batch)
            #Decoder data
            decoded_data = decoder(encoded_data)
            #Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        #Creates a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        #Evaluate global loss
        val_loss = loss_fn(conc_out,conc_label)

        return val_loss.data

# ============================ 缺失：训练循环 ============================
history = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epoch):
    print(f'\nEpoch {epoch+1}/{num_epoch}')

    # 训练
    train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
    # 验证
    val_loss = test_epoch(encoder, decoder, device, valid_loader, loss_fn)

    # 记录损失
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f'\nTrain Loss: {train_loss:.4f} \t Val Loss: {val_loss:.4f}')

#=============================3、可视化运行成果==================================
#=============================可视化训练和验证损失==================================
test_epoch(encoder,decoder,device,test_loader,loss_fn).item()#Plot losses
plt.figure(figsize=(10,8))
plt.semilogy(history['train_loss'],label='Train')
plt.semilogy(history['val_loss'],label='Vaild')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.show()


def plot_ae_outputs(encoder,decoder,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = test_dataset[i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(),cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original Image')
        ax = plt.subplot(2,n,i+1+n)
        plt.imshow(rec_img.cpu().squeeze().numpy(),cmap = 'gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Recontructured images')
    plt.show()

plot_ae_outputs(encoder,decoder)

#=============================可视化隐空间==================================
#定义从隐空间重构图像的函数
def plot_reconstructed(decoder,r0=(-5,10),r1=(-10,5),n=10):
    plt.figure(figsize=(20,8.5))
    w=28
    img = np.zeros((n*w,n*w))
    for i,y in enumerate(np.linspace(*r1,n)):
        for j,x in enumerate(np.linspace(*r0,n)):
            z = torch.Tensor([[x,y]]).to(device)
            x_hat=decoder(z)
            x_hat=x_hat.reshape(28,28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w,j*w:(j+1)*w]=x_hat
    plt.imshow(img,extent=[*r0,*r1],cmap='gist_gray')
    plt.show()

plot_reconstructed(decoder,r0=(-1,1),r1=(-1,1))

# -------------------------- 生成编码结果 --------------------------
def generate_encoded_df(encoder, dataloader, device):
    encoder.eval()
    encoded_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            encoded = encoder(x)
            encoded_list.append(encoded.cpu().numpy())
            labels_list.append(y.numpy())

    encoded_all = np.concatenate(encoded_list)
    labels_all = np.concatenate(labels_list)

    # 构建 DataFrame（你的隐空间是 2D，所以两列）
    df = pd.DataFrame({
        'Enc. Variable 0': encoded_all[:, 0],
        'Enc. Variable 1': encoded_all[:, 1],
        'label': labels_all
    })
    return df

# 生成编码后的 DataFrame
encoded_sample = generate_encoded_df(encoder, test_loader, device)


#=============================分析自编码器==================================
encoded_sample['Enc. Variable 0'].describe()

plt.figure(figsize=(17,9))
plt.scatter(encoded_sample['Enc. Variable 0'],encoded_sample['Enc. Variable 1'],c = encoded_sample.label,cmap='tab10')
plt.colorbar()
plt.show()


# ================================== 打印模型架构 ============================
# 4、Netron（交互式可视化，最直观）
#
# x = torch.randn(1,1, 28, 28).to(device)
#
# model = nn.Sequential(OrderedDict([
# ('encoder', encoder),
# ('decoder',decoder)
# ]))
#
# # 导出为ONNX格式
# torch.onnx.export(model, x, "model.onnx",
#                   input_names=['input'],#输入节点名称	在 Netron 中显示为 'input'
#                   output_names=['output'],#output_names	输出节点名称	在 Netron 中显示为 'output'
#                   #动态维度	指定哪些维度可变（这里是 batch 维度）
#                   #输入的第0维（batch）是动态的，可以变化
#                   #输出的第0维（batch）也是动态的
#                   dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
#                   )
#
# # 启动Netron查看（浏览器自动打开）
# netron.start("model.onnx", browse=True)#模型地址，打开浏览器
#
# # 加入阻塞，防止进程结束
# input("按回车键停止服务...")  # 或者 while True: pass