import torch 
import torch.utils.data as data
import numpy as np
from phasing_model import *
from crystallographic_operations import generate_reflection_array, setup_absent, get_centers_twoarray, phaseerror_numba
from training_backend import init_3D_and_1D_indices, reflections_1D_to_3D, phases_1D_to_3D

def read_exampledata(datafile):

    f = open(datafile, 'r')
    file_string = f.read()
    f.close()

    data = ['refcode'+x for x in file_string.split('refcode') if x]
    
    amplitudes     = np.zeros((len(data), 1205))
    phases_target  = np.zeros((len(data), 1205))
    phases_predict = np.zeros((len(data), 1205))

    N = 0
    for crystal in data:
        crystal_lines = crystal.split('\n')
        M = 0
        for line in crystal_lines:
            if 'refcode' not in line and 'hkl' not in line:
                line = line.split()
                if len(line) > 0:
                    amplitudes[N][M]     = float(line[3])
                    phases_target[N][M]  = float(line[4])
                    phases_predict[N][M] = float(line[5])
                    M += 1
        N += 1

    return amplitudes, phases_target, phases_predict

N_cycle_max     = 3
mixed_precision = True 
max_index       = 10
hkl_shape       = [1 + max_index*2, 1 + max_index, 1 + max_index]

# model definition
model_args = {
     'max_index' : 10,
       'filters' : 96,
   'kernel_size' : 3,
     'cnn_depth' : 6,
           'dim' : 1024,
       'dim_exp' : 2048,
 'dim_token_exp' : 512,
     'mlp_depth' : 8,
   'reflections' : 1205,
}

bin_size = 180.0
offset   = bin_size / 2
bin_nr   = int(360 / bin_size)

reflections_1Dto3D_indices, _ = init_3D_and_1D_indices(max_index, hkl_shape)

hkl_shape_torch = torch.tensor(hkl_shape,dtype=torch.int)

reflection_array = generate_reflection_array(max_index)
absent_array     = torch.FloatTensor(setup_absent(max_index))

model = PhAINeuralNetwork(**model_args)

# load trained parameters 
state = torch.load('./PhAI_model.pth', weights_only = True)
model.load_state_dict(state)

data_amplitudes, data_phases_target, data_phases_predict = read_exampledata('example_data.txt')

translations_sg14 = torch.zeros(8,3)
translations_sg14[0] = torch.FloatTensor([1/2,0  ,0  ])
translations_sg14[1] = torch.FloatTensor([0  ,0  ,0  ])
translations_sg14[2] = torch.FloatTensor([0  ,1/2,0  ])
translations_sg14[3] = torch.FloatTensor([0  ,0  ,1/2])
translations_sg14[4] = torch.FloatTensor([0  ,1/2,1/2])
translations_sg14[5] = torch.FloatTensor([1/2,0  ,1/2])
translations_sg14[6] = torch.FloatTensor([1/2,1/2,0  ])
translations_sg14[7] = torch.FloatTensor([1/2,1/2,1/2])
translations_sg14 = translations_sg14.numpy().reshape(1,8,3)


for n in range(0, data_amplitudes.shape[0]):
    amplitudes = torch.FloatTensor(data_amplitudes[n]).reshape(1,1205)

    amplitudes = reflections_1D_to_3D(amplitudes, reflections_1Dto3D_indices, hkl_shape_torch)

    output_phases = torch.zeros((1, hkl_shape_torch[0], hkl_shape_torch[1], hkl_shape_torch[2]))

    last = False
    for i in range(0, N_cycle_max):
        if i == N_cycle_max-1:
            last = True
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                if last == False:
                    output_phases = model(amplitudes, output_phases)

                    output_phases = output_phases.permute(0,2,1)
                    output_phases = phases_1D_to_3D(output_phases, reflections_1Dto3D_indices, hkl_shape_torch, offset, bin_size, bin_nr)
                else:
                    output_phases = model(amplitudes, output_phases)

                    output_phases = output_phases.permute(0,2,1)

                    # convert phases to float
                    output_phases = torch.argmax(output_phases, dim=2)
                    output_phases = offset + (output_phases*bin_size) - 180.00 - (bin_size/2)

