import torch
import torch.nn as nn
import numpy as np
from myUNet_DF import UNet_choh_skip

N_RFCOIL = 16
N_INPUT_VERTICAL = 384
N_FREQ_ENCODING = 384
N_OUT_X = 384
N_OUT_Y = 384

N_HIDDEN_LRNN_1 = 12
N_HIDDEN_LRNN_2 = 12

N_UNET_DEPTH = 5

# ETER hybrid, fastmri 384x384
class ETER_hybrid_GRU_DFU(nn.Module):
	def __init__(self):
		super(ETER_hybrid_GRU_DFU, self).__init__()
		num_in_x = N_INPUT_VERTICAL
		num_in_y = N_FREQ_ENCODING
		n_coil = N_RFCOIL
		num_out_x = N_OUT_X
		num_out_y = N_OUT_Y
		input_size = num_in_y*n_coil*2
		num_layers = 1
		num_out1 = num_out_y*N_HIDDEN_LRNN_1
		num_in2 = num_in_x*N_HIDDEN_LRNN_1
		num_out2 = num_out_x*N_HIDDEN_LRNN_2
		num_feat_ch = int(num_out2*2/num_out_x) + N_RFCOIL*2
		n_hidden = N_HIDDEN_LRNN_2 + N_RFCOIL

		self.num_in_x = num_in_x
		self.num_in_y = num_in_y
		self.num_layers = num_layers
		self.num_out1 = num_out1
		self.num_out2 = num_out2
		self.num_out_x = num_out_x
		self.num_out_y = num_out_y

		self.gru_h = nn.GRU(input_size, num_out1, num_layers, batch_first=True, bidirectional=True)
		self.gru_v = nn.GRU(num_in2*2, num_out2, num_layers, batch_first=True, bidirectional=True)
		self.unet = UNet_choh_skip(in_channels=num_feat_ch, n_classes=1, depth=N_UNET_DEPTH, wf=6, batch_norm=False, up_mode='upconv', n_hidden=n_hidden)


	def forward(self, x, x_img):
		h_h0 = torch.zeros(self.num_layers*2, x.size(0), self.num_out1).cuda()
		h_v0 = torch.zeros(self.num_layers*2, x.size(0), self.num_out2).cuda()

		in_h = x.reshape([x.size(0), self.num_in_x, -1])
		out_h, _ = self.gru_h(in_h, h_h0)
		out_h = out_h.reshape([x.size(0), self.num_in_x, self.num_out_y,-1])
		out_h = out_h.permute(0, 2, 1, 3)
		out_h = out_h.reshape([x.size(0), self.num_out_y, -1])

		out_v, _ = self.gru_v(out_h, h_v0)
		out_v = out_v.reshape([x.size(0), self.num_out_y, self.num_out_x,-1])
		out_v = out_v.permute(0, 3, 2, 1)


		in_cnn = torch.cat((out_v, x_img), dim=1)
		out = self.unet(in_cnn)

		return out


