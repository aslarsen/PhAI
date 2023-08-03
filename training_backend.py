import torch 
import math
import numpy as np
from crystallographic_operations import get_centers_twoarray

@torch.jit.script
def phasecalc_torch(amplitudes, phases, hkl, reciprocal_lattices, elements, uvw, occupancy, uiso,
                    max_element_nr: int, step_size: float, mag_count: int, batch_step_input: int,
                    formfactor_magnitude_library_torch):
    """
    Function that calculates phases and amplitudes of a batch of crystals, all the crystals must have to same number of
    atoms given by the max_element_nr parameter. Crystals with fewer atoms will have the contribution of the absent
    atoms zeroed by the setting the occupancy of the abent atoms to zero.


    Args: 
        amplitudes and phases: zero initialized torch tensors                                           size = batch_nr, reflection_nr 

        hkl: miller indicies of reflections to calculate                                                size = reflection_nr, 3

        reciprocal_lattices: reciprocal lattices of the crystals                                        size = batch_nr, 3, 3

        elements: elements numbers, absent atoms should have element number 0                           size = batch_nr, max_element_nr

        uvw: tensor of fractional cordinatees, absent atoms should have [0.0, 0.0, 0.0] as cordinate    size = batch_nr, max_element_nr, 3

        occupancy: tensor of occupancies, absnet atoms should have the value 0.0                        size = batch_nr, max_element_nr

        uiso: tensor of uiso parameters, absent atoms should have 0.0 as value                          size = batch_nr, max_element_nr

        max_element_nr:                     maximum number of atoms/elements in a crystal
        step_size:                          step size used in the form factor lookup table
        mag_count:                          size of the form factor lookup table
        batch_step_input:                   size of simultaneous calculations in the loop

        formfactor_magnitude_library_torch: form factor lookup table

    """
    batch_nr      = amplitudes.shape[0]
    reflection_nr = hkl.shape[0]
    steps = batch_nr // batch_step_input
    extra_steps = 0
    if batch_nr % batch_step_input != 0:
        extra_steps += 1

    for i in range(0,steps+extra_steps):
        if i >= steps:
            BATCH_STEP = batch_nr-batch_step_input*steps
            BATCH_STEP_0 = i*batch_step_input
            BATCH_STEP_1 = i*batch_step_input + (batch_nr-batch_step_input*steps)
        else:
            BATCH_STEP = batch_step_input
            BATCH_STEP_0 = i*batch_step_input
            BATCH_STEP_1 = (i+1)*batch_step_input

        Q    = torch.matmul(hkl, reciprocal_lattices[BATCH_STEP_0:BATCH_STEP_1])
        Qmag = torch.sqrt(torch.sum(Q ** 2, dim=2, keepdim=False))
        Qmag_indicies = torch.div(Qmag, step_size).long()
        Qmag_indicies = Qmag_indicies.reshape(BATCH_STEP, reflection_nr, 1).expand(BATCH_STEP, reflection_nr, max_element_nr)

        elements_batch = elements[BATCH_STEP_0:BATCH_STEP_1].reshape(BATCH_STEP, 1, max_element_nr).expand(BATCH_STEP, reflection_nr, max_element_nr)

        magnitude_indicies = ((elements_batch * mag_count) + Qmag_indicies).long()
        magnitude_indicies = magnitude_indicies.reshape(BATCH_STEP*reflection_nr*max_element_nr)

        # get form factors 
        ff = torch.index_select(formfactor_magnitude_library_torch, 0, magnitude_indicies)
        ff = ff.reshape(BATCH_STEP, reflection_nr, max_element_nr)

        sin_values = torch.sin(2.0*math.pi*torch.matmul(hkl, torch.transpose(uvw[BATCH_STEP_0:BATCH_STEP_1],1,2)))
        cos_values = torch.cos(2.0*math.pi*torch.matmul(hkl, torch.transpose(uvw[BATCH_STEP_0:BATCH_STEP_1],1,2)))

        occupancy_batch    = occupancy[BATCH_STEP_0:BATCH_STEP_1].reshape(BATCH_STEP, 1, max_element_nr).expand(BATCH_STEP, reflection_nr, max_element_nr)
        dw_factors         = torch.exp(-0.5*((Qmag**2).reshape(BATCH_STEP, reflection_nr, 1).expand(BATCH_STEP, reflection_nr, max_element_nr))*uiso[BATCH_STEP_0:BATCH_STEP_1].reshape(BATCH_STEP, 1, max_element_nr))

        SF = dw_factors*occupancy_batch*ff*torch.complex(cos_values,sin_values)
        SF = torch.sum(SF, dim=2, keepdim=False)

        # get amplitudes and phases
        amplitudes[BATCH_STEP_0:BATCH_STEP_1] = torch.sqrt(torch.real(SF * torch.conj(SF)))
        phases[BATCH_STEP_0:BATCH_STEP_1]     = torch.angle(SF)

    # clean data
    eps = 0.01
    elements_to_zero = torch.where(amplitudes > eps, 1.0, 0.0)
    amplitudes       = amplitudes*elements_to_zero

    amplitudes      /= amplitudes.max(1, keepdim=True)[0]

    phases           = phases*elements_to_zero
    phases           = torch.rad2deg(phases)

    return amplitudes, phases


def phaseloss_crossentropy(output, target, weights, hkl, translations, bin_size: float, offset: float, bin_nr: int):
    """
    Calculates the crossentropy phaseloss between output of the neural network and the targets
    Args:
        output: phase predictions as logits 
        target: target phases angles in degrees
        weights: weight per reflection of each sample
        hkl: tensor of reflections with miller indicies 
        translations: tensor of possible origin translations
        bin_size, offset and bin_nr: converts an angle in degree to the either 0 or 1 
         
    """

    weights        = weights.reshape(-1,1,weights.shape[1],1)
    target         = target.reshape(-1,1,target.shape[1]).repeat(1,translations.shape[0],1)
    output         = output.reshape(-1,1,output.shape[1],output.shape[2]).repeat(1,translations.shape[0],1,1)

    shifted_phases = target - 2 * 180.0*torch.einsum('ij,kj->ki', hkl, translations)
    shifted_phases = ((shifted_phases+180) % 360) - 180
    shifted_phases[shifted_phases == -180.0] = 180.0

    shifted_phases = torch.div(shifted_phases+180.0+offset, bin_size, rounding_mode='trunc').long()
    shifted_phases[shifted_phases == bin_nr] = 0
    shifted_phases = shifted_phases.reshape(shifted_phases.shape[0], shifted_phases.shape[1], shifted_phases.shape[2], 1)

    output        = output.type(torch.float32)
    p             = -torch.gather(output,3,shifted_phases) + torch.logsumexp(output,3,keepdim=True)
    p             = torch.sum((p*weights[:,:,0:hkl.shape[0],:]) / torch.sum(weights[:,:,0:hkl.shape[0],:], dim=2, keepdim=True),  dim=2, keepdim=True)
    min_errors, _ = torch.min(p, dim=1)
    loss          = torch.mean(min_errors)

    return loss

@torch.jit.script
def apply_mask(amplitudes, mask):
    amplitudes  = amplitudes*mask
    amplitudes /= amplitudes.max(1, keepdim=True)[0]
    return amplitudes

def init_3D_and_1D_indices(max_index, hkl_size):

    reflections_3Dto1D_indices = []

    reflection_indices = torch.zeros((hkl_size[0], hkl_size[1], hkl_size[2]), dtype=torch.int32)
    inside_sphere_reflection_count = 0
    reflection_count               = 0
    reflections_3Dto1D_indices = []
    empty_value = 99999
    for h in range(-max_index, max_index+1):
        for k in range(0, max_index+1):
            for l in range(0, max_index+1):
                if not(h==0 and k==0 and l==0):
                    if  math.sqrt(h**2+k**2+l**2) <= max_index:
                        reflection_indices[h+max_index][k][l] = inside_sphere_reflection_count
                        inside_sphere_reflection_count += 1
                        reflections_3Dto1D_indices.append(reflection_count)
                    else:
                        reflection_indices[h+max_index][k][l] = empty_value
                else:
                    reflection_indices[h+max_index][k][l] = empty_value
                reflection_count += 1
    total_reflections = inside_sphere_reflection_count

    reflections_3Dto1D_indices = torch.FloatTensor(reflections_3Dto1D_indices).long()
    reflections_1Dto3D_indices = torch.flatten(reflection_indices)
    reflections_1Dto3D_indices[reflections_1Dto3D_indices==empty_value] = total_reflections

    return reflections_1Dto3D_indices, reflections_3Dto1D_indices

@torch.jit.script
def reflections_1D_to_3D(data, indices, hkls):
    batch_size = data.shape[0]
    data = torch.nn.functional.pad(data, (0,1), mode='constant', value=0.0)
    data = torch.index_select(data, 1, indices)
    data = data.reshape(batch_size, hkls[0], hkls[1], hkls[2])
    return data

@torch.jit.script
def phases_1D_to_3D(phases, indices, hkls, offset: float, bin_size: float, bin_nr: int):
    batch_size = phases.shape[0]

    phases = torch.argmax(phases, dim=2)
    phases[phases == bin_nr] = 0
    phases = offset + (phases*bin_size) - 180.00 - (bin_size/2)
    phases = torch.nn.functional.pad(phases, (0,1), mode='constant', value=0.0)

    phases = torch.index_select(phases, 1, torch.flatten(indices))
    phases = phases.reshape(batch_size, hkls[0], hkls[1], hkls[2])

    return phases

def make_predictions(device, mixed_precision, model, generator, translations_sg14, N_cycle_max, formfactor_magnitude_library, reflection_array, step_size, mag_count, average_weights, reflections_1Dto3D_indices, hkl_shape_torch, offset, bin_size, bin_nr):
    predict_amplitudes    = []
    predict_target_phases = []
    predict_output_phases = []
    predict_loss       = []
    for batch_uvw, batch_elements, batch_occupancy, batch_usio, batch_reciprocal_lattice, batch_resolution_mask, in generator:
        batch_uvw       = batch_uvw.to(device, non_blocking=True)
        batch_elements  = batch_elements.to(device, non_blocking=True)
        batch_occupancy = batch_occupancy.to(device, non_blocking=True)
        batch_usio      = batch_usio.to(device, non_blocking=True)
        batch_resolution_mask = batch_resolution_mask.to(device, non_blocking=True)
        batch_reciprocal_lattice = batch_reciprocal_lattice.to(device, non_blocking=True)

        batch_amplitudes = torch.zeros((batch_uvw.shape[0], reflection_array.shape[0]),device=device)
        batch_phases     = torch.zeros((batch_uvw.shape[0], reflection_array.shape[0]),device=device)

        batch_amplitudes, batch_phases = phasecalc_torch(batch_amplitudes, batch_phases, reflection_array, batch_reciprocal_lattice, batch_elements, batch_uvw, batch_occupancy, batch_usio,
                                                         batch_elements.shape[1], step_size, mag_count, 4, formfactor_magnitude_library)

        batch_amplitudes = apply_mask(batch_amplitudes, batch_resolution_mask)

        batch_weights    = torch.clone(batch_amplitudes)

        output_phases    = torch.zeros((batch_phases.shape[0], hkl_shape_torch[0], hkl_shape_torch[1], hkl_shape_torch[2]),device=device)

        batch_amplitudes = reflections_1D_to_3D(batch_amplitudes, reflections_1Dto3D_indices, hkl_shape_torch)

        last = False
        for i in range(0, N_cycle_max):
            if i == N_cycle_max-1:
                last = True

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=mixed_precision):
                    if last == False:
                        output_phases = model(batch_amplitudes, output_phases)

                        output_phases = output_phases.permute(0,2,1)
                        output_phases = phases_1D_to_3D(output_phases, reflections_1Dto3D_indices, hkl_shape_torch, offset, bin_size, bin_nr)
                    else:
                        output_phases = model(batch_amplitudes, output_phases)

                        output_phases = output_phases.permute(0,2,1)
                        loss          = phaseloss_crossentropy(output_phases, batch_phases, batch_weights, reflection_array, translations_sg14, bin_size, offset, bin_nr)

        predict_loss.append(loss.item())
        predict_amplitudes.append(batch_amplitudes.cpu().detach())
        predict_target_phases.append(batch_phases.cpu().detach())
        predict_output_phases.append(output_phases.cpu().detach())

    predict_loss = np.mean(predict_loss)

    predict_output_phases = np.array(torch.cat(predict_output_phases, 0).numpy())
    predict_target_phases = np.array(torch.cat(predict_target_phases, 0).numpy())
    predict_amplitudes    = np.array(torch.cat(predict_amplitudes, 0).numpy())

    # convert output phases from the binary format to -180 and 0.0
    predict_output_phases = get_centers_twoarray(predict_output_phases.astype(np.float32), bin_size = 180.0)

    return predict_loss, predict_output_phases, predict_target_phases, predict_amplitudes

