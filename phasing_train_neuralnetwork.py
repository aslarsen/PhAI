import torch 
import torch.utils.data as data
import numpy as np
from phasing_model import *
from phasing_dataset import powderdataset_reader_zarr, PhasingDatasetZarr, setup_formfactors_looptable, samples_splitter
from crystallographic_operations import generate_reflection_array, setup_absent
from training_backend import phasecalc_torch, phaseloss_crossentropy, apply_mask, init_3D_and_1D_indices, reflections_1D_to_3D, phases_1D_to_3D, make_predictions

data_training_file = 'sg14_max10_fulldataset.zarr'
data_location = './'
datafile_trainingset  = data_location + data_training_file

validation_split    = 0.05 # ratio of validation data compared to training data
allowed_spacegroups = [14]
allowed_elements    = [ i for i in range(0,200) ] 
max_index           = 10
max_datasamples     = 999999999
form_factor_file    = 'formfactors.dat'

hkl_shape           = [1 + max_index*2, 1 + max_index, 1 + max_index]

learning_rate   = 0.0005
batch_size      = 212
weight_decay    = 1e-6
pytorch_workers = 2
epochs          = 999999
N_cycle_max     = 3
mixed_precision = True 
print_step      = 30   # print loss every print_step
validation_step = 1000 # make validation predictions every validation_step

params = {'batch_size': batch_size,
             'shuffle': True,
         'num_workers': pytorch_workers,
           'drop_last': False}

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

reflection_array = generate_reflection_array(max_index)
absent_array     = torch.FloatTensor(setup_absent(max_index))

print ('reading data set', flush=True)
samples = powderdataset_reader_zarr(datafile_trainingset, validation_split, allowed_spacegroups, allowed_elements, min_volume = 0.0, max_volume = 1000.0, max_length = 10.0, max_datasamples = max_datasamples)

training_samples, validation_samples = samples_splitter(samples, validation_split)

print ('init dataloaders', flush=True)
training_set       = PhasingDatasetZarr(datafile_trainingset, training_samples, reflection_array, max_asym_atoms = 100, max_total_atoms = 400)
training_generator = data.DataLoader(training_set, **params)

validation_set       = PhasingDatasetZarr(datafile_trainingset, validation_samples, reflection_array, max_asym_atoms = 100, max_total_atoms = 400)
validation_generator = data.DataLoader(validation_set, **params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print ('init network', flush=True)
model = PhAINeuralNetwork(**model_args)
model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

average_weights = np.load('average_weights.npy')
average_weights = (torch.FloatTensor(average_weights)*absent_array).to(device)

# setup origin translations
translations_sg14 = torch.zeros(8,3)
translations_sg14[0] = torch.FloatTensor([1/2,0  ,0  ])
translations_sg14[1] = torch.FloatTensor([0  ,0  ,0  ])
translations_sg14[2] = torch.FloatTensor([0  ,1/2,0  ])
translations_sg14[3] = torch.FloatTensor([0  ,0  ,1/2])
translations_sg14[4] = torch.FloatTensor([0  ,1/2,1/2])
translations_sg14[5] = torch.FloatTensor([1/2,0  ,1/2])
translations_sg14[6] = torch.FloatTensor([1/2,1/2,0  ])
translations_sg14[7] = torch.FloatTensor([1/2,1/2,1/2])
translations_sg14 = translations_sg14.to(device, non_blocking=True)

reflection_array = torch.FloatTensor(reflection_array).to(device, non_blocking=True)

reflections_1Dto3D_indices, _ = init_3D_and_1D_indices(max_index, hkl_shape)
reflections_1Dto3D_indices    = reflections_1Dto3D_indices.to(device, non_blocking=True)
hkl_shape_torch = torch.tensor(hkl_shape,dtype=torch.int).to(device, non_blocking=True)

print ('init form factors',flush=True)
max_magnitude = 75.0
step_size = 0.0001
formfactor_magnitude_library, mag_count = setup_formfactors_looptable(form_factor_file, max_magnitude, step_size)
formfactor_magnitude_library = torch.FloatTensor(formfactor_magnitude_library).to(device, non_blocking=True)

scaler    = torch.cuda.amp.GradScaler(enabled=mixed_precision)

print ('start training', flush=True)
step = 0
for epoch in range(1, epochs):
    for batch_uvw, batch_elements, batch_occupancy, batch_usio, batch_reciprocal_lattice, batch_resolution_mask, in training_generator:
        step += 1
        optimizer.zero_grad()
        batch_uvw       = batch_uvw.to(device, non_blocking=True)
        batch_elements  = batch_elements.to(device, non_blocking=True)
        batch_occupancy = batch_occupancy.to(device, non_blocking=True)
        batch_usio      = batch_usio.to(device, non_blocking=True)
        batch_resolution_mask = batch_resolution_mask.to(device, non_blocking=True)
        batch_reciprocal_lattice = batch_reciprocal_lattice.to(device, non_blocking=True)

        N_cycle = torch.randint(low=1, high=N_cycle_max+1, size=(1,), device=device)[0]
        batch_amplitudes = torch.zeros((batch_uvw.shape[0], reflection_array.shape[0]),device=device)
        batch_phases     = torch.zeros((batch_uvw.shape[0], reflection_array.shape[0]),device=device)

        batch_amplitudes, batch_phases = phasecalc_torch(batch_amplitudes, batch_phases, reflection_array, batch_reciprocal_lattice, batch_elements, batch_uvw, batch_occupancy, batch_usio, 
                                                         batch_elements.shape[1], step_size, mag_count, 4, formfactor_magnitude_library)

        batch_amplitudes = apply_mask(batch_amplitudes, batch_resolution_mask)

        batch_weights    = torch.clone(average_weights).expand(batch_phases.shape[0], batch_phases.shape[1])

        output_phases    = torch.zeros((batch_phases.shape[0], hkl_shape_torch[0], hkl_shape_torch[1], hkl_shape_torch[2]),device=device)

        batch_amplitudes = reflections_1D_to_3D(batch_amplitudes, reflections_1Dto3D_indices, hkl_shape_torch)

        last = False
        for i in range(0, N_cycle):
            if i == N_cycle-1:
                last = True
 
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                if last == False:
                    with torch.no_grad():
 
                        output_phases = model(batch_amplitudes, output_phases)
 
                        output_phases = output_phases.permute(0,2,1) 
                        output_phases = phases_1D_to_3D(output_phases, reflections_1Dto3D_indices, hkl_shape_torch, offset, bin_size, bin_nr)
                else:
                    with torch.enable_grad():
                        output_phases = model(batch_amplitudes, output_phases)
 
                        output_phases = output_phases.permute(0,2,1) 
                        loss          = phaseloss_crossentropy(output_phases, batch_phases, batch_weights, reflection_array, translations_sg14, bin_size, offset, bin_nr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % print_step == 0:
            loss_line = "step: %10i loss: %10.7f" % (step, loss)
            print (loss_line)

        if step % validation_step == 0:
            val_loss, val_output_phases, val_target_phases, val_amplitudes = make_predictions(device, mixed_precision, model, validation_generator, translations_sg14, \
                                                                                              N_cycle_max, formfactor_magnitude_library, reflection_array, \
                                                                                              step_size, mag_count, average_weights, reflections_1Dto3D_indices, \
                                                                                              hkl_shape_torch, offset, bin_size, bin_nr)
            val_line = "step: %10i validation_loss: %10.7f" % (step, val_loss)
            print (val_line) 


