import zarr
import numpy as np
import numba
from torch.utils.data import Dataset
import torch
from crystallographic_operations import get_unit_cell_parameters, sample_uiso_simple_uniform, generate_symmetric_molecules_numba, get_resolution_mask_ovoid
from random import uniform

@numba.jit(nopython=True, fastmath = True)
def check_empty_data(spacegroups):
    nonzero = False
    for i in range(0, len(spacegroups)):
        if spacegroups[i] != 0:
            nonzero = True
    return nonzero

@numba.jit(nopython=True, fastmath = True)
def check_elements(elements, allowed_elements):
    good_structure = True
    for i in range(0, len(elements)):
        good = False
        for n in range(0, len(allowed_elements)):
            if elements[i] == allowed_elements[n]:
                good = True
        if good == False:
            good_structure = False
            break
    return good_structure

@numba.jit(nopython=True, fastmath = True)
def check_sample(data_elements, data_spacegroups, data_volumes, data_lattice, allowed_spacegroups, allowed_elements, min_volume, max_volume, max_length, found_nr, max_datasamples):
    sample_check_nr = len(data_elements)
    good_samples = np.zeros((sample_check_nr), dtype=np.int64)
    N = 0
    for i in range(0, sample_check_nr):
        if (data_spacegroups[i] == allowed_spacegroups).any() == True:
            #if data_volumes[i] > min_volume and data_volumes[i] < max_volume:
            a, b, c, alpha, beta, gamma = get_unit_cell_parameters(data_lattice[i][0], data_lattice[i][1], data_lattice[i][2])
            if a < max_length and b < max_length and c < max_length:
                if check_elements(data_elements[i], allowed_elements) == True:
                    #print (data_elements[i])
                    if (found_nr + N) >= max_datasamples:
                        break
                    else:
                        good_samples[N] = i
                        N += 1
    good_samples = good_samples[0:N]
    return good_samples, N


def powderdataset_reader_zarr(datafile_name, validation_split, allowed_spacegroups, allowed_elements, min_volume = 0.0, max_volume = 1000.0,  max_length = 10.0, max_datasamples = 999999999):

    zarr_file   = zarr.open(datafile_name, mode='r')

    data_length = len(zarr_file.data.spacegroup_number)
    part_size = 500000
    found_nr = 0

    if data_length < part_size:
        part_size = data_length

    total_parts = data_length // part_size
    if data_length % part_size > 0:
        total_parts += 1
    allowed_spacegroups = np.array(allowed_spacegroups, dtype=np.int16)
    allowed_elements    = np.array(allowed_elements, dtype=np.int16)

    samples  = []
    for i in range(0, total_parts):
        start = i*part_size
        end   = (i+1)*part_size
        if i == total_parts-1: # last part
            end = data_length

        elements           = zarr_file.data.asymmetric_elements[start:end]
        spacegroups        = zarr_file.data.spacegroup_number[start:end]
        volumes            = zarr_file.data.volume[start:end]
        lattice            = zarr_file.data.lattice[start:end]

        good_samples, this_found_nr = check_sample(elements, spacegroups, volumes, lattice, allowed_spacegroups, allowed_elements, min_volume, max_volume, max_length, found_nr, max_datasamples)
        good_samples += start
        samples.extend(good_samples)

        found_nr += this_found_nr

        if found_nr >= max_datasamples:
            break
        if check_empty_data(spacegroups) == False:
            break

    return samples

def samples_splitter(samples, split_ratio):

    permutations = np.random.permutation(len(samples))

    samples  = np.array(samples)[permutations]

    validation_split_int = int(len(samples)*split_ratio)

    validation_samples = samples[0:validation_split_int]
    training_samples   = samples[validation_split_int:]

    return training_samples, validation_samples

@numba.jit(nopython=True, fastmath = True)
def setup_selection(samples, samples_nr, max_asym_atoms):

    sample_select_uvw     = np.zeros((samples_nr, max_asym_atoms, 3), dtype=np.int32)
    asym_select_uvw       = np.zeros((max_asym_atoms,3), dtype=np.int32)
    asym_select           = np.zeros((max_asym_atoms), dtype=np.int32)
    sample_select         = np.zeros((samples_nr, max_asym_atoms), dtype=np.int32)
    sample_select_1d      = np.zeros((samples_nr, max_asym_atoms), dtype=np.int32)
    sample_select_lattice = np.zeros((samples_nr, 3, 3), dtype=np.int32)
    lattice_select        = np.zeros((3,3), dtype=np.int32)

    for i in range(0, samples_nr):
        for j in range(0, max_asym_atoms):
            #sample_select[i][j] = samples[i]
            sample_select_uvw[i][j][0] = samples[i]
            sample_select_uvw[i][j][1] = samples[i]
            sample_select_uvw[i][j][2] = samples[i]

    for i in range(0, samples_nr):
        for j in range(0, 3):
            #sample_select[i][j] = samples[i]
            sample_select_lattice[i][j][0] = samples[i]
            sample_select_lattice[i][j][1] = samples[i]
            sample_select_lattice[i][j][2] = samples[i]

    for j in range(0, max_asym_atoms):
        asym_select_uvw[j][0] = j
        asym_select_uvw[j][1] = j
        asym_select_uvw[j][2] = j

    for i in range(0, max_asym_atoms):
        asym_select[i] = i

    for i in range(0, samples_nr):
        for j in range(0, max_asym_atoms):
            sample_select[i][j] = samples[i]

    for j in range(0, 3):
        lattice_select[j][0] = j
        lattice_select[j][1] = j
        lattice_select[j][2] = j

    return sample_select_uvw, asym_select_uvw, sample_select, asym_select, sample_select_lattice, lattice_select

@numba.jit(nopython=True, fastmath = True)
def setup_phasing_data(uvw, elements, occupancy, uiso, max_atoms):
    uvw_new       = np.zeros((max_atoms,3),dtype=np.float32)
    elements_new  = np.zeros((max_atoms),dtype=np.int32)
    occupancy_new = np.zeros((max_atoms),dtype=np.float32)
    uiso_new      = np.zeros((max_atoms),dtype=np.float32)

    uvw_new[0:len(uvw)]             = uvw
    elements_new[0:len(elements)]   = elements
    occupancy_new[0:len(occupancy)] = occupancy
    uiso_new[0:len(occupancy)]      = uiso

    return uvw_new, elements_new, occupancy_new, uiso_new

@numba.jit(nopython=True, fastmath = True)
def remove_random_reflections(resolution_mask, fraction_to_remove):

    total_reflections        = len(resolution_mask)
    present_reflections      = np.count_nonzero(resolution_mask)
    reflections_nr_to_remove = int(present_reflections*fraction_to_remove)
    reflections_nr_to_keep   = present_reflections - reflections_nr_to_remove

    keep         = np.ones(reflections_nr_to_keep,dtype=np.int32)
    remove       = np.zeros(reflections_nr_to_remove,dtype=np.int32)
    remove_array = np.concatenate((keep, remove))

    np.random.shuffle(remove_array)
    N = 0
    for i in range(0, total_reflections):
        if resolution_mask[i] != 0:
            resolution_mask[i] = remove_array[N]
            N += 1
    return resolution_mask


class PhasingDatasetZarr(Dataset):
    def __init__(self, datafile_name, samples, hkl_array, max_asym_atoms = 100, max_total_atoms = 400, resolution_sample = [0.0, 2.0], remove_fraction = 0.15, uiso_range = [0.005,0.06]):
        self.datafile_name       = datafile_name

        self.root                  = zarr.open(datafile_name, mode='r')

        self.samples             = samples
        self.samples_nr          = len(self.samples)
        self.resolution_sample   = resolution_sample
        self.remove_fraction     = remove_fraction
        self.uiso_range          = uiso_range 

        self.max_asym_atoms      = max_asym_atoms
        self.max_total_atoms     = max_total_atoms

        self.uvw, self.elements, self.occupancy, self.atom_nr, self.lattice, self.reciprocal_lattice, self.spacegroup, self.refcodes = self.load_data_in_parts_fancy(self.root, self.samples, self.samples_nr, self.max_asym_atoms)

        self.subset = []
        self.hkl_subset = []

        self.hkl_array = hkl_array

        self.sym_rotations = np.array([[[ 1.,  0.,  0.],
                                        [ 0.,  1.,  0.],
                                        [ 0.,  0.,  1.]],
        
                                       [[-1.,  0.,  0.],
                                        [ 0., -1.,  0.],
                                        [ 0.,  0., -1.]],
        
                                       [[-1.,  0.,  0.],
                                        [ 0.,  1.,  0.],
                                        [ 0.,  0., -1.]],
        
                                       [[ 1.,  0.,  0.],
                                        [ 0., -1.,  0.],
                                        [ 0.,  0.,  1.]]])

        self.sym_translations = np.array([[0.,  0. , 0. ],
                                          [0.,  0. , 0. ],
                                          [0.,  0.5, 0.5],
                                          [0.,  0.5, 0.5]])


    def load_data_in_parts_fancy(self, zarr_file, samples, samples_nr, max_asym_atoms):

        atom_select   = np.array([0,1,2])
        sample_select_uvw, asym_select_uvw, sample_select, asym_select, sample_select_lattice, lattice_select = setup_selection(samples, samples_nr, max_asym_atoms)

        uvw                = zarr_file.data.asymmetric_uvw.get_coordinate_selection((sample_select_uvw, asym_select_uvw, atom_select))
        occupancy          = zarr_file.data.asymmetric_occupancy.get_coordinate_selection((sample_select, asym_select))
        elements           = zarr_file.data.asymmetric_elements.get_coordinate_selection((sample_select, asym_select))
        atom_nr            = zarr_file.data.asymmetric_size.get_coordinate_selection((samples))
        lattice            = zarr_file.data.lattice.get_coordinate_selection((sample_select_lattice, lattice_select, atom_select))
        reciprocal_lattice = zarr_file.data.lattice_reciprocal.get_coordinate_selection((sample_select_lattice, lattice_select, atom_select))
        spacegroups        = zarr_file.data.spacegroup_number.get_coordinate_selection((samples))
        refcodes           = zarr_file.data.refcode.get_coordinate_selection((samples))

        return uvw, elements, occupancy, atom_nr, lattice, reciprocal_lattice, spacegroups, refcodes


    def __len__(self):
        return self.samples_nr

    def __getitem__(self, idx):
        data_id = idx

        spacegroup        = self.spacegroup[data_id]
        refcode           = self.refcodes[data_id]

        asymmetric_uiso = sample_uiso_simple_uniform(self.elements[data_id][0:self.atom_nr[data_id]], self.uiso_range[0], self.uiso_range[1])

        total_uvw, total_elements, total_occupancy, total_usio = generate_symmetric_molecules_numba(self.uvw[data_id][0:self.atom_nr[data_id]],
                                                                                               self.elements[data_id][0:self.atom_nr[data_id]],
                                                                                              self.occupancy[data_id][0:self.atom_nr[data_id]],
                                                                                                                               asymmetric_uiso,
                                                                                                                            self.sym_rotations,
                                                                                                                         self.sym_translations,
                                                                                                                      self.lattice[data_id][0],
                                                                                                                      self.lattice[data_id][1],
                                                                                                                      self.lattice[data_id][2])

        total_uvw, total_elements, total_occupancy, total_usio = setup_phasing_data(total_uvw, total_elements, total_occupancy, total_usio, max_atoms = self.max_total_atoms)
        total_uvw       = torch.FloatTensor(total_uvw)
        total_elements  = torch.FloatTensor(total_elements)
        total_occupancy = torch.FloatTensor(total_occupancy)
        total_usio      = torch.FloatTensor(total_usio)
        reciprocal_lattice = torch.FloatTensor(self.reciprocal_lattice[data_id])

        resolution      = uniform(self.resolution_sample[0], self.resolution_sample[1])
        resolution_mask = get_resolution_mask_ovoid(self.hkl_array, get_unit_cell_parameters(self.lattice[data_id][0], self.lattice[data_id][1], self.lattice[data_id][2]), resolution)

        remove_fraction = uniform(0.0,self.remove_fraction)
        resolution_mask = remove_random_reflections(resolution_mask, remove_fraction)

        resolution_mask = torch.FloatTensor(resolution_mask).bool()

        return total_uvw, total_elements, total_occupancy, total_usio, reciprocal_lattice, resolution_mask


@numba.jit(nopython=True, fastmath = True)
def setup_all_element_formfactors(ELEMENT_NR, a_formfactors_array, b_formfactors_array, c_formfactors_array, max_magnitude, step_size = 0.001):
    magnitudes            = np.linspace(0.0, max_magnitude, 1+int((max_magnitude - 0.0) / step_size))
    magnitudes_segment_nr = len(magnitudes)
    formfactor_magnitude_library = np.zeros((ELEMENT_NR+1, magnitudes_segment_nr),dtype=np.float32) # index 0 just empty 
    for i in range(0, ELEMENT_NR+1):
        for n in range(0, magnitudes_segment_nr):
            a1 = a_formfactors_array[i][0]
            a2 = a_formfactors_array[i][1]
            a3 = a_formfactors_array[i][2]
            a4 = a_formfactors_array[i][3]

            b1 = b_formfactors_array[i][0]
            b2 = b_formfactors_array[i][1]
            b3 = b_formfactors_array[i][2]
            b4 = b_formfactors_array[i][3]

            c  = c_formfactors_array[i]

            Qmag = magnitudes[n]

            f = a1 * np.exp(-b1 * (Qmag / (4 * np.pi)) ** 2) + \
                a2 * np.exp(-b2 * (Qmag / (4 * np.pi)) ** 2) + \
                a3 * np.exp(-b3 * (Qmag / (4 * np.pi)) ** 2) + \
                a4 * np.exp(-b4 * (Qmag / (4 * np.pi)) ** 2) + c
            formfactor_magnitude_library[i][n] = f
    return formfactor_magnitude_library, magnitudes


def setup_formfactors_looptable(formfactor_file, max_magnitude = 75.0, step_size = 0.0001):
    ELEMENT_NR = 118

    a_formfactors_dict = {}
    b_formfactors_dict = {}
    c_formfactors_dict = {}
    a_formfactors_array = np.zeros((ELEMENT_NR+1,4),dtype=np.float32) # index 0 is empty 
    b_formfactors_array = np.zeros((ELEMENT_NR+1,4),dtype=np.float32)
    c_formfactors_array = np.zeros((ELEMENT_NR+1),dtype=np.float32)
    f = open('formfactors.dat','r')
    for line in f:
        if '#' not in line:
            line = line.split()
            element_nr = int(line[0])

            a1 = float(line[3])
            b1 = float(line[4])
            a2 = float(line[5])
            b2 = float(line[6])
            a3 = float(line[7])
            b3 = float(line[8])
            a4 = float(line[9])
            b4 = float(line[10])
            c  = float(line[11])
            a_formfactors_dict.update({element_nr:np.array([a1,a2,a3,a4])})
            b_formfactors_dict.update({element_nr:np.array([b1,b2,b3,b4])})
            c_formfactors_dict.update({element_nr:c})
            a_formfactors_array[element_nr] = np.array([a1,a2,a3,a4])
            b_formfactors_array[element_nr] = np.array([b1,b2,b3,b4])
            c_formfactors_array[element_nr] = c

    formfactor_magnitude_library, magnitudes = setup_all_element_formfactors(ELEMENT_NR, a_formfactors_array, b_formfactors_array, c_formfactors_array, max_magnitude = max_magnitude, step_size = step_size)
    mag_count = int(max_magnitude / step_size) + 1

    formfactor_magnitude_library_final = formfactor_magnitude_library[0]
    for i in range(1, len(formfactor_magnitude_library)):
        formfactor_magnitude_library_final = np.concatenate((formfactor_magnitude_library_final, formfactor_magnitude_library[i]))

    return formfactor_magnitude_library_final, mag_count
