import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt

from functions import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# set path
data_path = "./data/"    # define UCF-101 RGB data path
action_name_path = './UCF101actions.pkl'
save_model_path = "./ResNetCRNN_ckpt/"

os.makedirs(save_model_path, exist_ok=True)

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
epochs = 120        # training epochs
batch_size = 40  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []

    for batch_idx, (X, y, is_new) in enumerate(train_loader):
        if is_new[0].item():
            h_n = c_n = None

        # distribute data to device
        X, y = X.to(device).squeeze(0), y.to(device).squeeze(0)

        optimizer.zero_grad()

        output, h_n, c_n = rnn_decoder(cnn_encoder(X), h_0=h_n, c_0=c_n)   # output has dim = (batch, number of classes)

        output = output.flatten()
        out_speed = output[1:] - output[:-1]

        pos_loss = F.mse_loss(output, y[:, 0])
        speed_loss = F.mse_loss(out_speed, y[1:, 1])

        loss = 0.5 * pos_loss + 0.5 * speed_loss

        losses.append((pos_loss.item(), speed_loss.item()))

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}]\tPosition Loss: {:.6f}, Speed Loss: {:.6f}'.format(
                epoch + 1, batch_idx, pos_loss.item(), speed_loss.item()))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    total_batches = 0
    with torch.no_grad():
        for batch_idx, (X, y, is_new) in enumerate(test_loader):
            if is_new[0].item():
                h_n = c_n = None

            X, y = X.to(device).squeeze(0), y.to(device).squeeze(0)

            output, h_n, c_n = rnn_decoder(cnn_encoder(X), h_0=h_n, c_0=c_n)   # output has dim = (batch, number of classes)

            output = output.flatten()
            out_speed = output[1:] - output[:-1]

            pos_loss = F.mse_loss(output, y[:, 0])
            speed_loss = F.mse_loss(out_speed, y[1:, 1])

            loss = 0.5 * pos_loss + 0.5 * speed_loss

            test_loss += loss.item()
            total_batches += 1
            # collect all y and y_pred in all batches
            # all_y.extend(y)
            # all_y_pred.extend(y_pred)

    test_loss /= total_batches

    # # compute accuracy
    # all_y = torch.stack(all_y, dim=0)
    # all_y_pred = torch.stack(all_y_pred, dim=0)
    # test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(total_batches, test_loss))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': 1}

# train, test split
# train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_set, valid_set = Dataset_CRNN(data_path, transform=transform), \
                       Dataset_CRNN(data_path, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p).to(device)

# Parallelize model to multiple GPUs
# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     cnn_encoder = nn.DataParallel(cnn_encoder)
#     rnn_decoder = nn.DataParallel(rnn_decoder)

#     # Combine all EncoderCNN + DecoderRNN parameters
#     crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
#                   list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
#                   list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

# elif torch.cuda.device_count() == 1:
# print("Using", torch.cuda.device_count(), "GPU!")
# Combine all EncoderCNN + DecoderRNN parameters
crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_ResNetCRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()
