import torch 
import numpy as np
import numba
import math

@numba.jit(nopython=True, fastmath = True)
def get_unit_cell_parameters(lattice_a, lattice_b, lattice_c):

    a_length = np.linalg.norm(lattice_a)
    b_length = np.linalg.norm(lattice_b)
    c_length = np.linalg.norm(lattice_c)

    alpha = np.rad2deg(math.acos(np.dot(lattice_b,lattice_c)/(b_length*c_length)))
    beta  = np.rad2deg(math.acos(np.dot(lattice_a,lattice_c)/(a_length*c_length)))
    gamma = np.rad2deg(math.acos(np.dot(lattice_b,lattice_a)/(b_length*a_length)))

    return a_length, b_length, c_length, alpha, beta, gamma

@numba.jit(nopython=True)
def sample_uiso_simple_uniform(elements, lower, upper):
    uiso = np.zeros(len(elements))
    for i in range(0, len(elements)):
        uiso[i] = np.random.uniform(lower,upper)
    return uiso


@numba.jit(nopython=True, fastmath = True)
def d_spacing(hkl, cellparam):
    h = hkl[0]
    k = hkl[1]
    l = hkl[2]
    a = cellparam[0]
    b = cellparam[1]
    c = cellparam[2]
    al = cellparam[3]*(math.pi/180.0)
    be = cellparam[4]*(math.pi/180.0)
    ga = cellparam[5]*(math.pi/180.0)

    # Triclinic general case
    s11 = b**2 * c**2 * (math.sin(al))**2
    s22 = a**2 * c**2 * (math.sin(be))**2
    s33 = a**2 * b**2 * (math.sin(ga))**2

    s12 = a * b * c**2 * (math.cos(al)*math.cos(be) - math.cos(ga))
    s23 = a**2 * b * c * (math.cos(be)*math.cos(ga) - math.cos(al))
    s31 = a * b**2 * c * (math.cos(ga)*math.cos(al) - math.cos(be))

    vol_sq = a**2 * b**2 * c**2 * (1 - (math.cos(al))**2 - (math.cos(be))**2 - (math.cos(ga))**2 + 2 * (math.cos(al)) * (math.cos(be)) * (math.cos(ga)))

    temp_d = (1/vol_sq) * (s11*h**2 + s22*k**2 + s33*l**2 + 2*s12*h*k + 2*s23*k*l + 2*s31*l*h)

    return math.sqrt(1/temp_d) #returns d_spacing

@numba.jit(nopython=True, fastmath = True)
def get_resolution_mask_ovoid(hkl_list, unitcell, resolution):
    resolution_mask = np.zeros((1205),dtype=np.int32)

    max_sin_th_over_lambda = 1.0 / (2.0*resolution)
    for i in range(0, len(hkl_list)):
        if 1.0 / (2.0*d_spacing(hkl_list[i], unitcell)) <= max_sin_th_over_lambda:
            resolution_mask[i] = 1
        else:
            resolution_mask[i] = 0

    return resolution_mask

@numba.jit(nopython=True, fastmath = True)
def generate_symmetric_molecules_numba(asymmetric_unit, elements, asymmetric_occupancy,  asymmetric_uiso, rotations, translations, lattice_a, lattice_b, lattice_c):
    count   = 0
    atom_nr = len(asymmetric_unit)*len(rotations)
    atoms_fractional = np.zeros((atom_nr, 3), dtype=np.float32)
    atoms_elements   = np.zeros(atom_nr, dtype=np.int16)
    occupancy        = np.zeros(atom_nr, dtype=np.float32)
    uiso             = np.zeros(atom_nr, dtype=np.float32)

    for n in range(0, len(rotations)):
        for i in range(0, len(asymmetric_unit)):

            # new
            atoms_fractional[count][0] = rotations[n][0][0] * asymmetric_unit[i][0] + rotations[n][0][1] * asymmetric_unit[i][1] + rotations[n][0][2] * asymmetric_unit[i][2] + translations[n][0]
            atoms_fractional[count][1] = rotations[n][1][0] * asymmetric_unit[i][0] + rotations[n][1][1] * asymmetric_unit[i][1] + rotations[n][1][2] * asymmetric_unit[i][2] + translations[n][1]
            atoms_fractional[count][2] = rotations[n][2][0] * asymmetric_unit[i][0] + rotations[n][2][1] * asymmetric_unit[i][1] + rotations[n][2][2] * asymmetric_unit[i][2] + translations[n][2]
            atoms_elements[count]      = elements[i]
            occupancy[count]           = asymmetric_occupancy[i]
            uiso[count]                = asymmetric_uiso[i]

            count += 1
    return atoms_fractional, atoms_elements, occupancy, uiso

@numba.jit(nopython=True, fastmath = True)
def add_missing_data(amplitudes):
    max_index = 10

    indicies = np.zeros((21,11,11), dtype=np.int32)

    N = 0
    for h in range(-max_index, max_index+1):
        for k in range(0, max_index+1):
            for l in range(0, max_index+1):
                if not(h==0 and k==0 and l==0):
                    if math.sqrt(h**2+k**2+l**2) <= max_index:
                        #if h < 0 and l == 0:
                        indicies[h+max_index][k][l] = N
                        N += 1
    N = 0
    for h in range(-max_index, max_index+1):
        for k in range(0, max_index+1):
            for l in range(0, max_index+1):
                if not(h==0 and k==0 and l==0):
                    if math.sqrt(h**2+k**2+l**2) <= max_index:
                        if h < 0 and l == 0:
                            amplitudes[N] = amplitudes[indicies[(2*max_index)-(h+max_index)][k][l]]
                        N += 1

    return amplitudes

def generate_reflection_array(max_index):
    hkl_array = []
    N = 0
    for h in range(-max_index, max_index+1):
        for k in range(0, max_index+1):
            for l in range(0, max_index+1):
                if not(h==0 and k==0 and l==0):
                    if math.sqrt(h**2+k**2+l**2) <= max_index:
                        hkl_array.append([h,k,l])

                        N += 1
    hkl_array = np.array(hkl_array, dtype=np.float32)
    return hkl_array

def setup_absent(max_index):
    absent_array = []
    for h in range(-max_index, max_index+1):
        for k in range(0, max_index+1):
            for l in range(0, max_index+1):
                if not(h==0 and k==0 and l==0):
                    if math.sqrt(h**2+k**2+l**2) <= max_index:
                        num = 1.0
                        if k == 0 and l % 2 != 0:
                            num = 0.0
                        if h == 0 and k % 2 != 0 and l == 0:
                            num = 0.0
                        if h == 0 and k == 0 and l % 2 != 0:
                            num = 0.0
                        absent_array.append(num)
    absent_array = np.array(absent_array)
    return absent_array

@numba.jit(nopython=True, fastmath = True)
def get_bin_center(anglebin, bin_size):
    offset   = bin_size / 2
    m = offset + (anglebin*bin_size) - 180.0 - bin_size/2
    return m

@numba.jit(nopython=True, fastmath = True)
def get_centers_twoarray(bins, bin_size):
    bin_nr = int(360.0 / bin_size)
    centers = np.zeros((bins.shape[0], bins.shape[1]), dtype=np.float32)
    for b in range(0, bins.shape[0]):
        for i in range(0, bins.shape[1]):
            predicted_class = np.argmax(bins[b][i])
            if predicted_class == bin_nr:
                predicted_class = 0
            center = get_bin_center(predicted_class, bin_size)
            centers[b][i] = center
    return centers

@numba.jit(nopython=True, fastmath = True)
def phaseloss_numba(output, target, hkl, weights, batch_translations, translations_nr, loss_type = 'L1', use_weights = False):
    batch_nr        = target.shape[0]
    reflections_nr  = hkl.shape[0]
    errors     = np.zeros((batch_nr, translations_nr))

    shifted_phases = np.zeros((batch_nr, translations_nr, reflections_nr), dtype=np.float32)
    for i in range(0, batch_nr):
        for j in range(0, translations_nr):
            translation_errors = np.zeros((reflections_nr))
            for n in range(0, reflections_nr):
                shifted_phases[i][j][n] = target[i][n] - 2 * 180.0*np.dot(hkl[n], batch_translations[i][j])
                shifted_phases[i][j][n] = ((shifted_phases[i][j][n]+180) % 360) - 180.0
                if shifted_phases[i][j][n] == -180.0:
                    shifted_phases[i][j][n] = 180.0

                if loss_type == 'L1':
                    translation_errors[n] = np.abs(np.abs(np.abs(output[i][n] - shifted_phases[i][j][n]) - 180.0) - 180.0)
                elif loss_type == 'MSE':
                    translation_errors[n] = (np.abs(np.abs(output[i][n] - shifted_phases[i][j][n]) - 180.0) - 180.0) ** 2
                if use_weights == True:
                    translation_errors[n] = translation_errors[n]*weights[i][n]
            if use_weights == False:
                errors[i][j] = np.mean(translation_errors)
            else:
                errors[i][j] = np.sum(translation_errors) / np.sum(weights[i])

    best_shifted_phases = np.zeros((batch_nr, reflections_nr))
    best_error          = np.zeros((batch_nr))
    best_translations   = np.zeros((batch_nr,3))
    for i in range(0, batch_nr):
        min_index              = np.argmin(errors[i])
        best_error[i]          = errors[i][min_index]
        best_shifted_phases[i] = shifted_phases[i][min_index]
        best_translations[i]   = batch_translations[i][min_index]

    loss = np.nanmean(best_error)
    return loss, best_error, best_shifted_phases, best_translations

