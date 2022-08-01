import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
from types import ModuleType
import matplotlib.pyplot as plt
import argparse

from model import ETER_hybrid_GRU_DFU
from myDataloader_fastmri_brain_random import choh_fastmri_brain_hybrid_ifft_random_acs32R4_test

from torch.utils.data import DataLoader
from torch.autograd import Variable




# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(device)




def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main(args):
	idx  = args.idx
	flag_cmap = args.cmap

	BATCH_SIZE = 4
	N_OUT_X = 384
	N_OUT_Y = 384

	model = ETER_hybrid_GRU_DFU()
	model = torch.load('tensors_R4.pt', map_location="cuda")
	model.eval()

	print(' number of params : {}'.format(get_n_params(model)))


	choh_data_test = choh_fastmri_brain_hybrid_ifft_random_acs32R4_test(num_total_set=1)


	testloader = DataLoader(choh_data_test, batch_size=BATCH_SIZE, shuffle=False)
	print(testloader)
	total_step = len(testloader)

	criterion = nn.MSELoss()

	inputs = []
	results = []
	refs = []

	with torch.no_grad():
		print('\n  start inferece')
		for i_batch, sample_batched in enumerate(testloader):
			data_in = sample_batched['data'].type(torch.cuda.FloatTensor)
			data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor)
			data_ref = sample_batched['label'].type(torch.cuda.FloatTensor)

			out = model(data_in, data_in_img)

			results = np.append( results, out.cpu().detach().numpy() )
			refs = np.append( refs, data_ref.cpu().detach().numpy() )

			loss = criterion(out, data_ref)


	results = np.reshape(results, [len(choh_data_test), N_OUT_X, N_OUT_Y])
	refs = np.reshape(refs, [len(choh_data_test), N_OUT_X, N_OUT_Y])


	flag_while = True
	while flag_while:

		plt.figure()

		img_pred = np.squeeze( results[idx,:,:] )
		img_truth = np.squeeze( refs[idx,:,:] )

		plt.subplot(3,1,1)
		plt.imshow(img_pred, aspect='equal', cmap=flag_cmap)
		plt.title('img_pred')
		plt.colorbar()

		plt.subplot(3,1,2)
		plt.imshow(img_truth, aspect='equal', cmap=flag_cmap)
		plt.title('img_truth')
		plt.colorbar()

		plt.subplot(3,1,3)
		plt.imshow(np.abs(img_truth-img_pred), aspect='equal', cmap=flag_cmap)
		plt.title('diff')
		plt.colorbar()
		plt.show()


		try:
			print(' ')
			x = int(input("Enter a number (idx): "))
		except ValueError:
			print('    choh, not a number, end')
			flag_while = False
			break
		idx = x






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='    choh, get array idx for display')
	parser.add_argument('-i', '--idx', type=int, default=3, help='array idx for display')
	parser.add_argument('-c','--cmap', type=str, default='viridis', help='colormap for display')
	args = parser.parse_args()

	main(args)
