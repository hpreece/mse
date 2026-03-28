import os
import numpy as np
import ctypes
import ast
import copy
import time

"""
# Multiple Stellar Evolution (MSE) -- A Population Synthesis Code for Multiple-Star Systems #

MSE is code that models the long-term evolution of hierarchical multiple-star systems (binaries, triples, quadruples, and higher-order systems) from the main sequence until remnant phase. It takes into account gravitational dynamical evolution, stellar evolution (using the `sse` tracks), and binary interactions (such as mass transfer and common-envelope evolution).  It includes routines for external perturbations from flybys in the field, or (to limited extent) encounters in dense stellar systems such as galactic nuclei. 

C++ and Fortran compilers are required, as well as Python (2/3) for the Python interface. Make sure to first compile the code using `make`. Please modify the Makefile according to your installation (`CXX` and `FC` should be correctly assigned).  

The script `test_mse.py` can be used to test the installation. The script `run_system.py` is useful for quickly running a system. 

See the user guide (doc/doc.pdf) for more detailed information.

"""


class MSE(object):

    def __init__(self):
        self.__CONST_G = 4.0*np.pi**2
        self.__CONST_C = 63239.72638679138
        self.__CONST_M_SUN = 1.0
        self.__CONST_R_SUN = 0.004649130343817401
        self.__CONST_L_SUN = 0.0002710404109745588
        self.__CONST_KM_PER_S = 0.210862
        self.__CONST_PER_PC3 = 1.14059e-16
        self.__CONST_PARSEC = 206201.0
        self.__CONST_MJUP = 0.000954248

        self.__system_index = 0
        self.__relative_tolerance = 1.0e-10
        self.__absolute_tolerance_eccentricity_vectors = 1.0e-8
        self.__absolute_tolerance_spin_vectors = 1.0e-3
        self.__absolute_tolerance_angular_momentum_vectors = 1.0e-2

        self.__wall_time_max_s = 3.6e4

        self.__include_quadrupole_order_terms = True
        self.__include_octupole_order_binary_pair_terms = True
        self.__include_octupole_order_binary_triplet_terms = True
        self.__include_hexadecupole_order_binary_pair_terms = True
        self.__include_dotriacontupole_order_binary_pair_terms = True
        self.__include_double_averaging_corrections = False

        self.__MSTAR_gbs_tolerance_default = 1.0e-10
        self.__MSTAR_gbs_tolerance_kick = 1.0e-8
        self.__MSTAR_collision_tolerance = 1.0e-10
        self.__MSTAR_output_time_tolerance = 1.0e-6
        self.__MSTAR_include_PN_acc_10 = True
        self.__MSTAR_include_PN_acc_20 = True
        self.__MSTAR_include_PN_acc_25 = True
        self.__MSTAR_include_PN_acc_30 = True
        self.__MSTAR_include_PN_acc_35 = True
        self.__MSTAR_include_PN_acc_SO = True
        self.__MSTAR_include_PN_acc_SS = True
        self.__MSTAR_include_PN_acc_Q = True
        self.__MSTAR_include_PN_spin_SO = True
        self.__MSTAR_include_PN_spin_SS = True
        self.__MSTAR_include_PN_spin_Q = True
        
        self.__nbody_analysis_fractional_semimajor_axis_change_parameter = 0.01
        self.__nbody_analysis_fractional_integration_time = 0.1  # [P5.1] paper default=0.1; was 0.05
        self.__nbody_analysis_minimum_integration_time = 1.0e1
        self.__nbody_analysis_maximum_integration_time = 1.0e5
        self.__nbody_dynamical_instability_direct_integration_time_multiplier = 1.5
        self.__nbody_semisecular_direct_integration_time_multiplier = 1.0e2
        self.__nbody_supernovae_direct_integration_time_multiplier = 1.5
        self.__nbody_other_direct_integration_time_multiplier = 1.5
        
        self.__effective_radius_multiplication_factor_for_collisions_stars = 1.0
        self.__effective_radius_multiplication_factor_for_collisions_compact_objects = 1.0e2
        
        self.__binary_evolution_CE_energy_flag = 0
        self.__binary_evolution_CE_spin_flag = 0  # [P5.1] paper default=0 (spins unaffected); was 1
        self.__binary_evolution_mass_transfer_timestep_parameter = 0.05
        self.__binary_evolution_CE_recombination_fraction = 1.0
        self.__binary_evolution_use_eCAML_model = False
        self.__binary_evolution_mass_transfer_model = 0
        self.__chandrasekhar_mass = 1.44
        self.__eddington_accretion_factor = 10.0
        self.__nova_accretion_factor = 1.0e-3
        self.__alpha_wind_accretion = 1.5
        self.__beta_wind_accretion = 0.125
        self.__NS_model = 0
        self.__ECSNe_model = 0
        self.__defining_upper_mass_for_sdB_formation = 0.65
        self.__binary_evolution_SNe_Ia_single_degenerate_model = 0
        self.__binary_evolution_SNe_Ia_double_degenerate_model = 0
        self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision = 0.9
        self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO = 0.9

        self.__triple_mass_transfer_primary_star_accretion_efficiency_no_disk = 0.1
        self.__triple_mass_transfer_secondary_star_accretion_efficiency_no_disk = 0.1
        self.__triple_mass_transfer_primary_star_accretion_efficiency_disk = 0.6
        self.__triple_mass_transfer_secondary_star_accretion_efficiency_disk = 0.3
        self.__triple_mass_transfer_inner_binary_alpha_times_lambda = 5.0

        self.__particles_committed = False
        self.model_time = 0.0
        self.time_step = 0.0
        self.relative_energy_error = 0.0
        self.state = 0
        self.CVODE_flag = 0
        self.CVODE_error_code = 0
        self.integration_flag = 0
        self.__stop_after_root_found = False
        self.__random_seed = 0
        self._log_cache = None
        
        self.__verbose_flag = 0 ### 0: no verbose output in C++; > 0: verbose output, with increasing verbosity (>1 will slow down the code considerably)
        
        self.enable_tides = True
        self.enable_root_finding = True
        self.enable_VRR = False
        
        self.__include_flybys = True
        self.__log_mstar_transitions = True
        self.__flybys_correct_for_gravitational_focussing = True
        self.__flybys_include_secular_encounters = False
        self.__flybys_reference_binary = -1
        self.__flybys_velocity_distribution = 0
        self.__flybys_mass_distribution = 0
        self.__flybys_mass_distribution_lower_value = 0.1
        self.__flybys_mass_distribution_upper_value = 100.0
        self.__flybys_encounter_sphere_radius = 1.0e5
        self.__flybys_stellar_density = 0.1*self.__CONST_PER_PC3 ### density at infinity
        self.__flybys_stellar_relative_velocity_dispersion = 30.0*self.__CONST_KM_PER_S

        __current_dir__ = os.path.dirname(os.path.realpath(__file__))
        lib_path = os.path.join(__current_dir__, 'libmse.so')

        if not os.path.isfile(lib_path):
            print('Library libmse.so does not exist -- trying to compile')
            os.system('make')
        
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.init_lib()
        self.particles = []

    def init_lib(self):
        self.lib.add_particle.argtypes = (ctypes.POINTER(ctypes.c_int),ctypes.c_bool,ctypes.c_bool)
        self.lib.add_particle.restype = ctypes.c_int
        
        self.lib.delete_particle.argtypes = (ctypes.c_int,)
        self.lib.delete_particle.restype = ctypes.c_int

        self.lib.set_children.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_int)
        self.lib.set_children.restype = ctypes.c_int

        self.lib.get_children.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int))
        self.lib.get_children.restype = ctypes.c_int

        self.lib.get_number_of_particles.argtypes = ()
        self.lib.get_number_of_particles.restype = ctypes.c_int

        self.lib.get_internal_index_in_particlesMap.argtypes = (ctypes.c_int,)
        self.lib.get_internal_index_in_particlesMap.restype = ctypes.c_int

        self.lib.get_is_binary.argtypes = (ctypes.c_int,)
        self.lib.get_is_binary.restype = ctypes.c_bool

        self.lib.get_is_bound.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_bool))
        self.lib.get_is_bound.restype = ctypes.c_int

        self.lib.set_mass.argtypes = (ctypes.c_int,ctypes.c_double)
        self.lib.set_mass.restype = ctypes.c_int

        self.lib.get_mass.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double))
        self.lib.get_mass.restype = ctypes.c_int

        self.lib.set_mass_transfer_terms.argtypes = (ctypes.c_int,ctypes.c_bool)
        self.lib.set_mass_transfer_terms.restype = ctypes.c_int

        self.lib.get_mass_transfer_terms.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_bool))
        self.lib.get_mass_transfer_terms.restype = ctypes.c_int

        self.lib.get_mass_dot.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double))
        self.lib.get_mass_dot.restype = ctypes.c_int

        self.lib.set_radius.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.c_double)
        self.lib.set_radius.restype = ctypes.c_int

        self.lib.get_radius.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_radius.restype = ctypes.c_int

        self.lib.set_spin_vector.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_spin_vector.restype = ctypes.c_int

        self.lib.get_spin_vector.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_spin_vector.restype = ctypes.c_int

        self.lib.set_stellar_evolution_properties.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int)
        self.lib.set_stellar_evolution_properties.restype = ctypes.c_int

        # C42 fix: added missing m_dot_accretion_SD (double*) param
        self.lib.get_stellar_evolution_properties.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_stellar_evolution_properties.restype = ctypes.c_int

        ### kicks ###
        self.lib.set_kick_properties.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_bool,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_kick_properties.restype = ctypes.c_int

        self.lib.get_kick_properties.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_kick_properties.restype = ctypes.c_int


        ### binary evolution ###
        self.lib.set_binary_evolution_properties.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_binary_evolution_properties.restype = ctypes.c_int

        self.lib.get_binary_evolution_properties.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_binary_evolution_properties.restype = ctypes.c_int


        ### orbital elements ###
        # C44 fix: last param sample_orbital_phase_randomly is bool in C, was c_int
        self.lib.set_orbital_elements.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_bool)
        self.lib.set_orbital_elements.restype = ctypes.c_int

        self.lib.get_orbital_elements.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),\
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_orbital_elements.restype = ctypes.c_int

        self.lib.get_inclination_relative_to_parent_interface.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double))
        self.lib.get_inclination_relative_to_parent_interface.restype = ctypes.c_int

        self.lib.get_relative_position_and_velocity.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_relative_position_and_velocity.restype = ctypes.c_int

        self.lib.get_absolute_position_and_velocity.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_absolute_position_and_velocity.restype = ctypes.c_int
        
        self.lib.set_integration_method.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_bool)
        self.lib.set_integration_method.restype = ctypes.c_int

        self.lib.set_PN_terms.argtypes = (ctypes.c_int,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool)
        self.lib.set_PN_terms.restype = ctypes.c_int

        self.lib.get_PN_terms.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool))
        self.lib.get_PN_terms.restype = ctypes.c_int

        self.lib.set_tides_terms.argtypes = (ctypes.c_int,ctypes.c_bool,ctypes.c_int,ctypes.c_bool,ctypes.c_bool,ctypes.c_double,ctypes.c_bool)
        self.lib.set_tides_terms.restype = ctypes.c_int

        self.lib.get_tides_terms.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_bool))
        self.lib.get_tides_terms.restype = ctypes.c_int

        self.lib.set_root_finding_terms.argtypes = (ctypes.c_int,ctypes.c_bool,ctypes.c_bool,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_bool,ctypes.c_bool,ctypes.c_double,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_double);
        self.lib.set_root_finding_terms.restype = ctypes.c_int

        self.lib.set_root_finding_state.argtypes = (ctypes.c_int,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,ctypes.c_bool)
        self.lib.set_root_finding_state.restype = ctypes.c_int

        self.lib.get_root_finding_state.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool),ctypes.POINTER(ctypes.c_bool))
        self.lib.get_root_finding_state.restype = ctypes.c_int

        self.lib.set_constants.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_constants.restype = ctypes.c_int

        self.__set_constants_in_code()

        self.lib.set_parameters.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double, \
            ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,\
            ctypes.c_bool,ctypes.c_bool,ctypes.c_bool,\
            ctypes.c_bool,ctypes.c_int,ctypes.c_bool, ctypes.c_int,ctypes.c_int, ctypes.c_bool, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, \
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_bool, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, \
            ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, \
            ctypes.c_bool, \
            ctypes.c_double, \
            ctypes.c_int, ctypes.c_int, \
            ctypes.c_int, \
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, \
            ctypes.c_bool)
        self.lib.set_parameters.restype = ctypes.c_int

        self.__set_parameters_in_code() 


        self.lib.evolve_interface.argtypes = (ctypes.c_double,ctypes.c_double, \
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int))
        self.lib.evolve_interface.restype = ctypes.c_int

        self.lib.set_external_particle_properties.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_external_particle_properties.restype = ctypes.c_int

        self.lib.apply_external_perturbation_assuming_integrated_orbits_interface.argtypes = ()
        self.lib.apply_external_perturbation_assuming_integrated_orbits_interface.restype = ctypes.c_int

        self.lib.apply_external_perturbation_assuming_integrated_orbits_single_perturber_interface.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
        self.lib.apply_external_perturbation_assuming_integrated_orbits_single_perturber_interface.restype = ctypes.c_int

        self.lib.apply_user_specified_instantaneous_perturbation_interface.argtypes = ()
        self.lib.apply_user_specified_instantaneous_perturbation_interface.restype = ctypes.c_int

        self.lib.set_instantaneous_perturbation_properties.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_instantaneous_perturbation_properties.restype = ctypes.c_int

        self.lib.set_VRR_properties.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double)
        self.lib.set_VRR_properties.restype = ctypes.c_int

        self.lib.reset_interface.argtypes = ()
        self.lib.reset_interface.restype = ctypes.c_int
        
        self.lib.set_random_seed.argtypes = (ctypes.c_int,)
        self.lib.set_random_seed.restype = ctypes.c_int

        self.lib.set_verbose_flag.argtypes = (ctypes.c_int,)
        self.lib.set_verbose_flag.restype = ctypes.c_int

        self.lib.initialize_code_interface.argtypes = ()
        self.lib.initialize_code_interface.restype = ctypes.c_int
        
        
        ### logging ###
        self.lib.get_size_of_log_data.argtypes = ()
        self.lib.get_size_of_log_data.restype = ctypes.c_int
        
        # C4/C41 fix: added missing eccentric_collision (int*) and eccentricity (double*) params
        self.lib.get_log_entry_properties.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double))
        self.lib.get_log_entry_properties.restype = ctypes.c_int
        
        self.lib.get_internal_index_in_particlesMap_log.argtypes = (ctypes.c_int,ctypes.c_int)
        self.lib.get_internal_index_in_particlesMap_log.restype = ctypes.c_int

        self.lib.get_is_binary_log.argtypes = (ctypes.c_int,ctypes.c_int)
        self.lib.get_is_binary_log.restype = ctypes.c_bool


        # C43 fix: added missing WD_He_layer_mass (double*) and m_dot_accretion_SD (double*) params
        self.lib.get_body_properties_from_log_entry.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_body_properties_from_log_entry.restype = ctypes.c_int
        
        self.lib.get_binary_properties_from_log_entry.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int), \
                        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), \
                        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
                        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.get_binary_properties_from_log_entry.restype = ctypes.c_int

        self.lib.write_final_log_entry_interface.argtypes = (ctypes.c_double, ctypes.c_int)
        self.lib.write_final_log_entry_interface.restype = ctypes.c_int
       
        ### tests ###
        self.lib.unit_tests_interface.argtypes = (ctypes.c_int,)
        self.lib.unit_tests_interface.restype = ctypes.c_int

        self.lib.determine_compact_object_merger_properties_interface.argtypes = ( ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.c_double, ctypes.c_double, ctypes.c_double, \
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), \
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), \
            ctypes.POINTER(ctypes.c_double) )
        self.lib.determine_compact_object_merger_properties_interface.restype = ctypes.c_int

        self.lib.sample_from_3d_maxwellian_distribution_interface.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double) )
        self.lib.sample_from_3d_maxwellian_distribution_interface.restype = ctypes.c_int

        self.lib.sample_from_normal_distribution_interface.argtypes = (ctypes.c_double, ctypes.c_double)
        self.lib.sample_from_normal_distribution_interface.restype = ctypes.c_double

        self.lib.sample_from_kroupa_93_imf_interface.argtypes = ()
        self.lib.sample_from_kroupa_93_imf_interface.restype = ctypes.c_double

        self.lib.sample_spherical_coordinates_unit_vectors_from_isotropic_distribution_interface.argtypes = (ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.sample_spherical_coordinates_unit_vectors_from_isotropic_distribution_interface.restype = ctypes.c_int

        self.lib.test_kick_velocity.argtypes = (ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double))
        self.lib.test_kick_velocity.restype = ctypes.c_int

        self.lib.test_flybys_perturber_sampling.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double, \
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), \
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
        self.lib.test_flybys_perturber_sampling.restype = ctypes.c_int

        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_accumulation_efficiency.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))
        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_accumulation_efficiency.restype = ctypes.c_int

        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_explosion.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_bool))
        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_explosion.restype = ctypes.c_int

        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_white_dwarf_hydrogen_accretion_boundaries.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_white_dwarf_hydrogen_accretion_boundaries.restype = ctypes.c_int
        
     ###############
    
    def add_particle(self,particle):
        index = ctypes.c_int(0)

        self.lib.add_particle(ctypes.byref(index), particle.is_binary, particle.is_external)
        particle.index = index.value
        if particle.is_binary==False:
            flag = self.lib.set_mass(particle.index,particle.mass)

        self.particles.append(particle)

    def add_particles(self,particles):
        for index,particle in enumerate(particles):
            self.add_particle(particle)
        
        ### All particles need to be added individually to the code first before calling __update_particles_in_code, since the latter includes reference to particles' children ###
        flag = self.__update_particles_in_code(self.particles)


    def delete_particle(self,particle):
        flag = self.lib.delete_particle(particle.index)
        if flag==-1:
            raise RuntimeError('Could not delete particle with index {0}'.format(particle.index))
        self.particles.remove(particle)

    def commit_particles(self):
        self.__set_random_seed()
        
        flag = self.__update_particles_in_code()
        
        self.lib.initialize_code_interface()
        
        self.__update_particles_from_code()

        end_time,initial_hamiltonian,state,CVODE_flag,CVODE_error_code,integration_flag = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0)
        error_code = self.lib.evolve_interface(0.0,0.0,ctypes.byref(end_time),ctypes.byref(initial_hamiltonian), \
            ctypes.byref(state),ctypes.byref(CVODE_flag),ctypes.byref(CVODE_error_code),ctypes.byref(integration_flag))

        self.initial_hamiltonian = initial_hamiltonian.value
        
        self.__particles_committed = True

    def evolve_model(self,end_time):
        
        if end_time is None:
            raise RuntimeError('End time not specified in evolve_model!')
        if self.__particles_committed == False:
            self.commit_particles()
        
        flag = self.__update_particles_in_code()

        ### get initial system structure ###
        orbits = [p for p in self.particles if p.is_binary == True]
        bodies_old = [b.index for b in self.particles if b.is_binary == False]
        children1_old = [o.child1.index for o in orbits]
        children2_old = [o.child2.index for o in orbits]

        ### integrate system of ODEs ###
        start_time = self.model_time

        output_time,hamiltonian,state,CVODE_flag,CVODE_error_code,integration_flag = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(self.integration_flag)

        error_code = self.lib.evolve_interface(start_time,end_time,ctypes.byref(output_time),ctypes.byref(hamiltonian), \
            ctypes.byref(state),ctypes.byref(CVODE_flag),ctypes.byref(CVODE_error_code),ctypes.byref(integration_flag))
        output_time,hamiltonian,state,CVODE_flag,CVODE_error_code,integration_flag = output_time.value,hamiltonian.value,state.value,CVODE_flag.value,CVODE_error_code.value,integration_flag.value

        self._invalidate_log_cache()

        ### compute energy error ###
        self.hamiltonian = hamiltonian
        if self.initial_hamiltonian == 0.0:
            self.relative_energy_error = 0.0
        else:
            self.relative_energy_error = np.fabs( (self.initial_hamiltonian - self.hamiltonian)/self.initial_hamiltonian )

        ### update model time ###
        self.model_time = output_time

        if (error_code != 0):  # [H11] check evolve_interface return value (0=success, non-zero=error)
            print('MSE: evolve_interface returned non-zero code {0}'.format(error_code))

        self.error_code = error_code
        self.CVODE_flag = CVODE_flag
        self.CVODE_error_code = CVODE_error_code
        self.state = state
        self.integration_flag = integration_flag

        #if self.integration_flag == 0: 
        self.__copy_particle_structure_from_code()

        self.__update_particles_from_code()

        ### check if the system structure changed (including changes in bodies) ###
        orbits = [p for p in self.particles if p.is_binary == True]
        
        children1 = [o.child1.index for o in orbits]
        children2 = [o.child2.index for o in orbits]
        bodies = [b.index for b in self.particles if b.is_binary == False]            
        
        self.structure_change = False
        if bodies != bodies_old or children1 != children1_old or children2 != children2_old:
            self.structure_change = True

        return self.state,self.structure_change,self.CVODE_flag,self.CVODE_error_code

    def apply_external_perturbation_assuming_integrated_orbits(self):
        self.__update_particles_in_code()
        self.lib.apply_external_perturbation_assuming_integrated_orbits_interface()
        self.__update_particles_from_code()

    def apply_external_perturbation_assuming_integrated_orbits_single_perturber(self, M_per, e_per, Q_per, e_vec_unit_per_x, e_vec_unit_per_y, e_vec_unit_per_z, h_vec_unit_per_x, h_vec_unit_per_y, h_vec_unit_per_z):
        self.__update_particles_in_code()
        self.lib.apply_external_perturbation_assuming_integrated_orbits_single_perturber_interface(M_per, e_per, Q_per, e_vec_unit_per_x, e_vec_unit_per_y, e_vec_unit_per_z, h_vec_unit_per_x, h_vec_unit_per_y, h_vec_unit_per_z)
        self.__update_particles_from_code()
        
    def apply_user_specified_instantaneous_perturbation(self):
        self.__update_particles_in_code(set_instantaneous_perturbation_properties=True)
        self.lib.apply_user_specified_instantaneous_perturbation_interface()
        self.__update_particles_from_code()

    def __update_particle_in_code(self,particle,set_instantaneous_perturbation_properties=False):
        flag = 0
        if particle.is_binary == False:
            flag = self.lib.set_mass(particle.index,particle.mass)

        if self.enable_tides == False:
            particle.include_tidal_friction_terms = False
            particle.tides_method = 1
            particle.include_tidal_bulges_precession_terms = False
            particle.include_rotation_precession_terms = False

        flag += self.lib.set_tides_terms(particle.index,particle.include_tidal_friction_terms,particle.tides_method,particle.include_tidal_bulges_precession_terms,particle.include_rotation_precession_terms, \
            particle.minimum_eccentricity_for_tidal_precession,particle.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary)
            
        if self.enable_root_finding == False:
            particle.check_for_secular_breakdown = False
            particle.check_for_dynamical_instability = False
            particle.check_for_physical_collision_or_orbit_crossing = False
            particle.check_for_RLOF_at_pericentre = False
            particle.check_for_GW_condition = False
            particle.check_for_entering_LISA_band = False
            
        flag += self.lib.set_root_finding_terms(particle.index,particle.check_for_secular_breakdown,particle.check_for_dynamical_instability,particle.dynamical_instability_criterion,particle.dynamical_instability_central_particle,particle.dynamical_instability_K_parameter, \
                particle.check_for_physical_collision_or_orbit_crossing,particle.check_for_minimum_periapse_distance,particle.check_for_minimum_periapse_distance_value,particle.check_for_RLOF_at_pericentre,particle.check_for_RLOF_at_pericentre_use_sepinsky_fit,particle.check_for_GW_condition,particle.check_for_entering_LISA_band,particle.check_for_entering_LISA_band_critical_GW_frequency)
        flag += self.lib.set_root_finding_state(particle.index,particle.secular_breakdown_has_occurred,particle.dynamical_instability_has_occurred, \
                particle.physical_collision_or_orbit_crossing_has_occurred,particle.minimum_periapse_distance_has_occurred,particle.RLOF_at_pericentre_has_occurred,particle.GW_condition_has_occurred,particle.entering_LISA_band_has_occurred)

        if self.enable_VRR == True:
            flag += self.lib.set_VRR_properties(particle.index,particle.VRR_model,particle.VRR_include_mass_precession,particle.VRR_mass_precession_rate, \
                particle.VRR_Omega_vec_x,particle.VRR_Omega_vec_y,particle.VRR_Omega_vec_z, \
                particle.VRR_eta_20_init,particle.VRR_eta_a_22_init,particle.VRR_eta_b_22_init,particle.VRR_eta_a_21_init,particle.VRR_eta_b_21_init, \
                particle.VRR_eta_20_final,particle.VRR_eta_a_22_final,particle.VRR_eta_b_22_final,particle.VRR_eta_a_21_final,particle.VRR_eta_b_21_final,particle.VRR_initial_time,particle.VRR_final_time)

        flag += self.lib.set_binary_evolution_properties(particle.index,particle.dynamical_mass_transfer_low_mass_donor_timescale,particle.dynamical_mass_transfer_WD_donor_timescale,particle.compact_object_disruption_mass_loss_timescale, \
            particle.common_envelope_alpha, particle.common_envelope_lambda, particle.common_envelope_timescale, particle.triple_common_envelope_alpha)

        flag += self.lib.set_PN_terms(particle.index,particle.include_pairwise_1PN_terms,particle.include_pairwise_25PN_terms,particle.include_spin_orbit_1PN_terms,particle.exclude_1PN_precession_in_case_of_isolated_binary)
        
        if particle.is_external==False:
            
            if particle.is_binary==True:
                flag += self.lib.set_children(particle.index,particle.child1.index,particle.child2.index)
                flag += self.lib.set_orbital_elements(particle.index,particle.a, particle.e, particle.TA, particle.INCL, particle.AP, particle.LAN, particle.sample_orbital_phase_randomly)
                flag += self.lib.set_integration_method(particle.index,particle.integration_method,particle.KS_use_perturbing_potential)
            else:
                flag += self.lib.set_radius(particle.index,particle.radius,particle.radius_dot)
                flag += self.lib.set_mass_transfer_terms(particle.index,particle.include_mass_transfer_terms)
                flag += self.lib.set_spin_vector(particle.index,particle.spin_vec_x,particle.spin_vec_y,particle.spin_vec_z)
                flag += self.lib.set_stellar_evolution_properties(particle.index,particle.stellar_type,particle.object_type,particle.sse_initial_mass,particle.metallicity,particle.sse_time_step,particle.epoch,particle.age, \
                    particle.convective_envelope_mass,particle.convective_envelope_radius,particle.core_mass,particle.core_radius,particle.luminosity,particle.apsidal_motion_constant,particle.gyration_radius,particle.tides_viscous_time_scale,particle.tides_viscous_time_scale_prescription)
                flag += self.lib.set_kick_properties(particle.index,particle.kick_distribution,particle.include_WD_kicks,particle.kick_distribution_sigma_km_s_NS,particle.kick_distribution_sigma_km_s_BH,particle.kick_distribution_sigma_km_s_WD, \
                    particle.kick_distribution_2_m_NS,particle.kick_distribution_4_m_NS,particle.kick_distribution_4_m_ej,particle.kick_distribution_5_v_km_s_NS,particle.kick_distribution_5_v_km_s_BH,particle.kick_distribution_5_sigma,particle.kick_distribution_sigma_km_s_NS_ECSN)

                if set_instantaneous_perturbation_properties==True:
                    flag += self.lib.set_instantaneous_perturbation_properties(particle.index,particle.instantaneous_perturbation_delta_mass, \
                        particle.instantaneous_perturbation_delta_X,particle.instantaneous_perturbation_delta_Y,particle.instantaneous_perturbation_delta_Z, \
                        particle.instantaneous_perturbation_delta_VX,particle.instantaneous_perturbation_delta_VY,particle.instantaneous_perturbation_delta_VZ)
                        
        else:
            flag += self.lib.set_external_particle_properties(particle.index, particle.external_t_ref, particle.e, particle.external_r_p, particle.INCL, particle.AP, particle.LAN)
    
        return flag

    def __update_particles_in_code(self,set_instantaneous_perturbation_properties=False):
        flag = 0
        for index,particle in enumerate(self.particles):
            if particle.is_binary==True:
                flag += self.lib.set_children(particle.index,particle.child1.index,particle.child2.index)

        # [H14] Do not reset flag to 0 here — accumulate errors from set_children
        for index,particle in enumerate(self.particles):
            flag += self.__update_particle_in_code(particle,set_instantaneous_perturbation_properties=set_instantaneous_perturbation_properties)
        return flag

    def __update_particle_from_code(self,particle):
        mass = ctypes.c_double(0.0)
        flag = self.lib.get_mass(particle.index,ctypes.byref(mass))
        particle.mass = mass.value

        include_tidal_friction_terms,tides_method,include_tidal_bulges_precession_terms,include_rotation_precession_terms,minimum_eccentricity_for_tidal_precession,exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = ctypes.c_bool(True),ctypes.c_int(0),ctypes.c_bool(True),ctypes.c_bool(True),ctypes.c_double(0.0),ctypes.c_bool(True)
        flag += self.lib.get_tides_terms(particle.index,ctypes.byref(include_tidal_friction_terms),ctypes.byref(tides_method),ctypes.byref(include_tidal_bulges_precession_terms),ctypes.byref(include_rotation_precession_terms),ctypes.byref(minimum_eccentricity_for_tidal_precession),ctypes.byref(exclude_rotation_and_bulges_precession_in_case_of_isolated_binary))
        particle.include_tidal_friction_terms = include_tidal_friction_terms.value
        particle.tides_method = tides_method.value
        particle.include_tidal_bulges_precession_terms = include_tidal_bulges_precession_terms.value
        particle.include_rotation_precession_terms = include_rotation_precession_terms.value
        particle.minimum_eccentricity_for_tidal_precession = minimum_eccentricity_for_tidal_precession.value
        particle.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = exclude_rotation_and_bulges_precession_in_case_of_isolated_binary.value

        include_pairwise_1PN_terms,include_pairwise_25PN_terms, include_spin_orbit_1PN_terms, exclude_1PN_precession_in_case_of_isolated_binary = ctypes.c_bool(True),ctypes.c_bool(True),ctypes.c_bool(True),ctypes.c_bool(True)
        flag += self.lib.get_PN_terms(particle.index,ctypes.byref(include_pairwise_1PN_terms),ctypes.byref(include_pairwise_25PN_terms),ctypes.byref(include_spin_orbit_1PN_terms),ctypes.byref(exclude_1PN_precession_in_case_of_isolated_binary))
        particle.include_pairwise_1PN_terms = include_pairwise_1PN_terms.value
        particle.include_pairwise_25PN_terms = include_pairwise_25PN_terms.value
        particle.include_spin_orbit_1PN_terms = include_spin_orbit_1PN_terms.value
        particle.exclude_1PN_precession_in_case_of_isolated_binary = exclude_1PN_precession_in_case_of_isolated_binary.value

        dynamical_mass_transfer_low_mass_donor_timescale,dynamical_mass_transfer_WD_donor_timescale,compact_object_disruption_mass_loss_timescale,common_envelope_alpha,common_envelope_lambda,common_envelope_timescale,triple_common_envelope_alpha = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
        flag += self.lib.get_binary_evolution_properties(particle.index,ctypes.byref(dynamical_mass_transfer_low_mass_donor_timescale),ctypes.byref(dynamical_mass_transfer_WD_donor_timescale),ctypes.byref(compact_object_disruption_mass_loss_timescale), \
            ctypes.byref(common_envelope_alpha), ctypes.byref(common_envelope_lambda), ctypes.byref(common_envelope_timescale), ctypes.byref(triple_common_envelope_alpha))
        particle.dynamical_mass_transfer_low_mass_donor_timescale = dynamical_mass_transfer_low_mass_donor_timescale.value
        particle.dynamical_mass_transfer_WD_donor_timescale = dynamical_mass_transfer_WD_donor_timescale.value
        particle.compact_object_disruption_mass_loss_timescale = compact_object_disruption_mass_loss_timescale.value
        particle.common_envelope_alpha = common_envelope_alpha.value
        particle.common_envelope_lambda = common_envelope_lambda.value
        particle.common_envelope_timescale = common_envelope_timescale.value
        particle.triple_common_envelope_alpha = triple_common_envelope_alpha.value

        if self.enable_root_finding == True:
            secular_breakdown_has_occurred,dynamical_instability_has_occurred,physical_collision_or_orbit_crossing_has_occurred,minimum_periapse_distance_has_occurred,RLOF_at_pericentre_has_occurred,GW_condition_has_occurred,entering_LISA_band_has_occurred = ctypes.c_bool(False),ctypes.c_bool(False),ctypes.c_bool(False),ctypes.c_bool(False),ctypes.c_bool(False),ctypes.c_bool(False),ctypes.c_bool(False)
            flag += self.lib.get_root_finding_state(particle.index,ctypes.byref(secular_breakdown_has_occurred),ctypes.byref(dynamical_instability_has_occurred), \
                ctypes.byref(physical_collision_or_orbit_crossing_has_occurred),ctypes.byref(minimum_periapse_distance_has_occurred),ctypes.byref(RLOF_at_pericentre_has_occurred),ctypes.byref(GW_condition_has_occurred),ctypes.byref(entering_LISA_band_has_occurred))
            particle.secular_breakdown_has_occurred = secular_breakdown_has_occurred.value
            particle.dynamical_instability_has_occurred = dynamical_instability_has_occurred.value
            particle.physical_collision_or_orbit_crossing_has_occurred = physical_collision_or_orbit_crossing_has_occurred.value
            particle.minimum_periapse_distance_has_occurred = minimum_periapse_distance_has_occurred.value
            particle.RLOF_at_pericentre_has_occurred = RLOF_at_pericentre_has_occurred.value
            particle.GW_condition_has_occurred = GW_condition_has_occurred.value
            particle.entering_LISA_band_has_occurred = entering_LISA_band_has_occurred.value

        if particle.is_binary==True:
            a,e,TA,INCL,AP,LAN = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
            flag += self.lib.get_orbital_elements(particle.index,ctypes.byref(a),ctypes.byref(e),ctypes.byref(TA),ctypes.byref(INCL),ctypes.byref(AP),ctypes.byref(LAN))
            particle.a = a.value
            particle.e = e.value
            particle.TA = TA.value
            particle.INCL = INCL.value
            particle.AP = AP.value
            particle.LAN = LAN.value
            
            INCL_parent = ctypes.c_double(0.0)
            flag += self.lib.get_inclination_relative_to_parent_interface(particle.index,ctypes.byref(INCL_parent))
            particle.INCL_parent = INCL_parent.value
            
            x,y,z,vx,vy,vz = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
            flag += self.lib.get_relative_position_and_velocity(particle.index,ctypes.byref(x),ctypes.byref(y),ctypes.byref(z),ctypes.byref(vx),ctypes.byref(vy),ctypes.byref(vz))
            particle.x = x.value
            particle.y = y.value
            particle.z = z.value
            particle.vx = vx.value
            particle.vy = vy.value
            particle.vz = vz.value
            
        else:
            X,Y,Z,VX,VY,VZ = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
            flag = self.lib.get_absolute_position_and_velocity(particle.index,ctypes.byref(X),ctypes.byref(Y),ctypes.byref(Z),ctypes.byref(VX),ctypes.byref(VY),ctypes.byref(VZ))
            particle.X = X.value
            particle.Y = Y.value
            particle.Z = Z.value
            particle.VX = VX.value
            particle.VY = VY.value
            particle.VZ = VZ.value

            if particle.is_external==False:
                is_bound = ctypes.c_bool(True)
                flag += self.lib.get_is_bound(particle.index,ctypes.byref(is_bound))
                particle.is_bound = is_bound.value
                
                radius,radius_dot = ctypes.c_double(0.0),ctypes.c_double(0.0)
                flag += self.lib.get_radius(particle.index,ctypes.byref(radius),ctypes.byref(radius_dot))
                particle.radius = radius.value
                particle.radius_dot = radius_dot.value

                stellar_type,object_type,sse_initial_mass,metallicity,sse_time_step,epoch,age,convective_envelope_mass,convective_envelope_radius,core_mass,core_radius,luminosity,apsidal_motion_constant,gyration_radius,tides_viscous_time_scale,roche_lobe_radius_pericenter,WD_He_layer_mass,m_dot_accretion_SD = ctypes.c_int(0),ctypes.c_int(0), \
                    ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
                flag += self.lib.get_stellar_evolution_properties(particle.index,ctypes.byref(stellar_type),ctypes.byref(object_type),ctypes.byref(sse_initial_mass),ctypes.byref(metallicity),ctypes.byref(sse_time_step), \
                    ctypes.byref(epoch),ctypes.byref(age),ctypes.byref(convective_envelope_mass),ctypes.byref(convective_envelope_radius),ctypes.byref(core_mass),ctypes.byref(core_radius),ctypes.byref(luminosity),ctypes.byref(apsidal_motion_constant),ctypes.byref(gyration_radius),ctypes.byref(tides_viscous_time_scale),ctypes.byref(roche_lobe_radius_pericenter),ctypes.byref(WD_He_layer_mass),ctypes.byref(m_dot_accretion_SD))
                particle.stellar_type = stellar_type.value
                particle.object_type = object_type.value
                particle.sse_initial_mass = sse_initial_mass.value
                particle.metallicity = metallicity.value
                particle.sse_time_step = sse_time_step.value
                particle.epoch = epoch.value
                particle.age = age.value
                particle.convective_envelope_mass = convective_envelope_mass.value
                particle.convective_envelope_radius = convective_envelope_radius.value
                particle.core_mass = core_mass.value
                particle.core_radius = core_radius.value
                particle.luminosity = luminosity.value
                particle.apsidal_motion_constant = apsidal_motion_constant.value
                particle.gyration_radius = gyration_radius.value
                particle.tides_viscous_time_scale = tides_viscous_time_scale.value
                particle.roche_lobe_radius_pericenter = roche_lobe_radius_pericenter.value
                particle.WD_He_layer_mass = WD_He_layer_mass.value
                particle.m_dot_accretion_SD = m_dot_accretion_SD.value
                
                kick_distribution,include_WD_kicks,kick_distribution_sigma_km_s_NS,kick_distribution_sigma_km_s_BH,kick_distribution_sigma_km_s_WD,kick_distribution_2_m_NS,kick_distribution_4_m_NS,kick_distribution_4_m_ej,kick_distribution_5_v_km_s_NS,kick_distribution_5_v_km_s_BH,kick_distribution_5_sigma,kick_distribution_sigma_km_s_NS_ECSN \
                    = ctypes.c_int(0),ctypes.c_bool(False),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
                flag += self.lib.get_kick_properties(particle.index,ctypes.byref(kick_distribution),ctypes.byref(include_WD_kicks),ctypes.byref(kick_distribution_sigma_km_s_NS),ctypes.byref(kick_distribution_sigma_km_s_BH),ctypes.byref(kick_distribution_sigma_km_s_WD), \
                ctypes.byref(kick_distribution_2_m_NS),ctypes.byref(kick_distribution_4_m_NS),ctypes.byref(kick_distribution_4_m_ej),ctypes.byref(kick_distribution_5_v_km_s_NS),ctypes.byref(kick_distribution_5_v_km_s_BH),ctypes.byref(kick_distribution_5_sigma),ctypes.byref(kick_distribution_sigma_km_s_NS_ECSN))
                particle.kick_distribution = kick_distribution.value
                particle.include_WD_kicks = include_WD_kicks.value
                particle.kick_distribution_sigma_km_s_NS = kick_distribution_sigma_km_s_NS.value
                particle.kick_distribution_sigma_km_s_BH = kick_distribution_sigma_km_s_BH.value
                particle.kick_distribution_sigma_km_s_WD = kick_distribution_sigma_km_s_WD.value
                particle.kick_distribution_2_m_NS = kick_distribution_2_m_NS.value
                particle.kick_distribution_4_m_NS = kick_distribution_4_m_NS.value
                particle.kick_distribution_4_m_ej = kick_distribution_4_m_ej.value
                particle.kick_distribution_5_v_km_s_NS = kick_distribution_5_v_km_s_NS.value
                particle.kick_distribution_5_v_km_s_BH = kick_distribution_5_v_km_s_BH.value
                particle.kick_distribution_5_sigma = kick_distribution_5_sigma.value
                particle.kick_distribution_sigma_km_s_NS_ECSN = kick_distribution_sigma_km_s_NS_ECSN.value  # [C48/H13] fix typo: ECN → ECSN

                mass_dot = ctypes.c_double(0.0)
                flag = self.lib.get_mass_dot(particle.index,ctypes.byref(mass_dot))
                particle.mass_dot = mass_dot.value

                spin_vec_x,spin_vec_y,spin_vec_z = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
                flag += self.lib.get_spin_vector(particle.index,ctypes.byref(spin_vec_x),ctypes.byref(spin_vec_y),ctypes.byref(spin_vec_z))
                particle.spin_vec_x = spin_vec_x.value
                particle.spin_vec_y = spin_vec_y.value
                particle.spin_vec_z = spin_vec_z.value
                
                include_mass_transfer_terms = ctypes.c_bool(True)
                flag += self.lib.get_mass_transfer_terms(particle.index,ctypes.byref(include_mass_transfer_terms))
                particle.include_mass_transfer_terms = include_mass_transfer_terms.value

        return flag
        
    def __update_particles_from_code(self):
        
        flag = 0
        for index,particle in enumerate(self.particles):
            flag += self.__update_particle_from_code(particle)
        return flag

    def __copy_particle_structure_from_code(self):
        self.particles = []
        N_particles = self.lib.get_number_of_particles()

        for i in range(N_particles):
            
            internal_index = self.lib.get_internal_index_in_particlesMap(i)
            is_binary = self.lib.get_is_binary(internal_index)

            mass = ctypes.c_double(0.0)
            flag = self.lib.get_mass(internal_index,ctypes.byref(mass))
            mass = mass.value

            child1_index,child2_index = -1,-1
            if is_binary==True:
                child1,child2 = ctypes.c_int(0),ctypes.c_int(0)
                self.lib.get_children(internal_index,ctypes.byref(child1),ctypes.byref(child2))
                child1_index = child1.value
                child2_index = child2.value

            #print("isb",N_particles,"i",i,"internal_index",internal_index,"mass",mass,"is_binary",is_binary,"child1_index",child1_index,"child2_index",child2_index)#,child1,child2)
            p = Particle(is_binary=is_binary,mass=mass,child1=None,child2=None,a=0.0,e=0.0,INCL=0.0,AP=0.0,LAN=0.0) ### orbital elements should be updated later
            p.index = internal_index

            p.child1_index = child1_index
            p.child2_index = child2_index
            self.particles.append(p)

        binaries = [x for x in self.particles if x.is_binary == True]

        for i,p in enumerate(self.particles):
            if p.is_binary == True:
                i1 = [j for j in range(N_particles) if self.particles[j].index == p.child1_index][0]
                i2 = [j for j in range(N_particles) if self.particles[j].index == p.child2_index][0]
                #print("i1",i1,"i2",i2,"i",i,"p.child1_index",p.child1_index,"p.child2_index",p.child2_index)
                p.child1 = self.particles[i1]
                p.child2 = self.particles[i2]

    def __set_constants_in_code(self):
        self.lib.set_constants(self.__CONST_G,self.__CONST_C,self.__CONST_M_SUN,self.__CONST_R_SUN,self.__CONST_L_SUN,self.__CONST_KM_PER_S,self.__CONST_PER_PC3,self.__CONST_MJUP)


    def __set_parameters_in_code(self):
        self.lib.set_parameters(self.__relative_tolerance,self.__absolute_tolerance_eccentricity_vectors,self.__absolute_tolerance_spin_vectors,self.__absolute_tolerance_angular_momentum_vectors,self.__include_quadrupole_order_terms, \
            self.__include_octupole_order_binary_pair_terms,self.__include_octupole_order_binary_triplet_terms, \
            self.__include_hexadecupole_order_binary_pair_terms,self.__include_dotriacontupole_order_binary_pair_terms, self.__include_double_averaging_corrections, \
            self.__include_flybys, self.__flybys_reference_binary, self.__flybys_correct_for_gravitational_focussing, self.__flybys_velocity_distribution, self.__flybys_mass_distribution, self.__flybys_include_secular_encounters, \
            self.__flybys_mass_distribution_lower_value, self.__flybys_mass_distribution_upper_value, self.__flybys_encounter_sphere_radius, \
            self.__flybys_stellar_density, self.__flybys_stellar_relative_velocity_dispersion, \
            self.__binary_evolution_CE_energy_flag, self.__binary_evolution_CE_spin_flag, self.__binary_evolution_mass_transfer_timestep_parameter, self.__binary_evolution_CE_recombination_fraction, self.__binary_evolution_use_eCAML_model, \
            self.__MSTAR_gbs_tolerance_default, self.__MSTAR_gbs_tolerance_kick, self.__MSTAR_collision_tolerance, self.__MSTAR_output_time_tolerance, \
            self.__nbody_analysis_fractional_semimajor_axis_change_parameter,self.__nbody_analysis_fractional_integration_time,self.__nbody_analysis_minimum_integration_time,self.__nbody_analysis_maximum_integration_time, \
            self.__nbody_dynamical_instability_direct_integration_time_multiplier,self.__nbody_semisecular_direct_integration_time_multiplier,self.__nbody_supernovae_direct_integration_time_multiplier,self.__nbody_other_direct_integration_time_multiplier, \
            self.__chandrasekhar_mass,self.__eddington_accretion_factor,self.__nova_accretion_factor,self.__alpha_wind_accretion,self.__beta_wind_accretion, \
            self.__triple_mass_transfer_primary_star_accretion_efficiency_no_disk,self.__triple_mass_transfer_secondary_star_accretion_efficiency_no_disk,self.__triple_mass_transfer_primary_star_accretion_efficiency_disk,self.__triple_mass_transfer_secondary_star_accretion_efficiency_disk,self.__triple_mass_transfer_inner_binary_alpha_times_lambda, \
            self.__effective_radius_multiplication_factor_for_collisions_stars, self.__effective_radius_multiplication_factor_for_collisions_compact_objects, \
            self.__MSTAR_include_PN_acc_10,self.__MSTAR_include_PN_acc_20,self.__MSTAR_include_PN_acc_25,self.__MSTAR_include_PN_acc_30,self.__MSTAR_include_PN_acc_35,self.__MSTAR_include_PN_acc_SO,self.__MSTAR_include_PN_acc_SS,self.__MSTAR_include_PN_acc_Q,self.__MSTAR_include_PN_spin_SO,self.__MSTAR_include_PN_spin_SS,self.__MSTAR_include_PN_spin_Q, \
            self.__stop_after_root_found, \
            self.__wall_time_max_s, \
            self.__NS_model, self.__ECSNe_model, \
            self.__system_index, \
            self.__binary_evolution_mass_transfer_model, self.__binary_evolution_SNe_Ia_single_degenerate_model, self.__binary_evolution_SNe_Ia_double_degenerate_model, self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision, self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO, \
            self.__defining_upper_mass_for_sdB_formation, \
            self.__log_mstar_transitions)

    def reset(self):
        self.__init__()
        self.lib.reset_interface()
        ctypes._reset_cache()
        
    def __set_random_seed(self):
        self.lib.set_random_seed(self.random_seed)

    def __set_verbose_flag(self):
        self.lib.set_verbose_flag(self.verbose_flag)

    ### Logging ###
    def __get_log(self):
        N_log = self.lib.get_size_of_log_data()
        #print("get_log",N_log)
        log = []
        for index_log in range(N_log):
            entry = {}
            particles = []
            #N_particles = self.lib.get_number_of_particles()
            
            time = ctypes.c_double(0.0)
            N_particles,event_flag,integration_flag,index1,index2,binary_index,kick_speed_km_s,SNe_type,SNe_info,eccentric_collision,eccentricity = ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_double(0.0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0),ctypes.c_double(0.0)
            flag = self.lib.get_log_entry_properties(index_log,ctypes.byref(time),ctypes.byref(event_flag),ctypes.byref(integration_flag),ctypes.byref(N_particles),ctypes.byref(index1),ctypes.byref(index2),ctypes.byref(binary_index),ctypes.byref(kick_speed_km_s),ctypes.byref(SNe_type),ctypes.byref(SNe_info),ctypes.byref(eccentric_collision),ctypes.byref(eccentricity))
            entry.update({'time':time.value,'event_flag':event_flag.value,'integration_flag':integration_flag.value,'index1':index1.value,'index2':index2.value,'binary_index':binary_index.value,'N_particles':N_particles.value,'kick_speed_km_s':kick_speed_km_s.value,"SNe_type":SNe_type.value,"SNe_info":SNe_info.value,"eccentric_collision":eccentric_collision.value,"eccentricity":eccentricity.value})
            #print("log i ",index_log,"N",N_log,"integration_flag",integration_flag.value)
            for index_particle in range(N_particles.value):
                internal_index = self.lib.get_internal_index_in_particlesMap_log(index_log,index_particle)

                is_binary = self.lib.get_is_binary_log(index_log,internal_index)
                
                append = False
                if is_binary == False:
                    parent,stellar_type,object_type = ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0)
                    mass,radius,core_mass,sse_initial_mass,convective_envelope_mass,epoch,age,core_radius,convective_envelope_radius,luminosity,ospin,X,Y,Z,VX,VY,VZ,metallicity,spin_vec_x,spin_vec_y,spin_vec_z,WD_He_layer_mass,m_dot_accretion_SD = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
                    flag = self.lib.get_body_properties_from_log_entry(index_log,internal_index,ctypes.byref(parent),ctypes.byref(mass),ctypes.byref(radius),ctypes.byref(stellar_type), \
                        ctypes.byref(core_mass),ctypes.byref(sse_initial_mass),ctypes.byref(convective_envelope_mass), \
                        ctypes.byref(epoch),ctypes.byref(age), \
                        ctypes.byref(core_radius),ctypes.byref(convective_envelope_radius),ctypes.byref(luminosity),ctypes.byref(ospin), \
                        ctypes.byref(X), ctypes.byref(Y), ctypes.byref(Z), ctypes.byref(VX), ctypes.byref(VY), ctypes.byref(VZ), \
                        ctypes.byref(object_type),ctypes.byref(metallicity), \
                        ctypes.byref(spin_vec_x),ctypes.byref(spin_vec_y),ctypes.byref(spin_vec_z), \
                        ctypes.byref(WD_He_layer_mass),ctypes.byref(m_dot_accretion_SD))

                    p = Particle(is_binary=is_binary,mass=mass.value,radius=radius.value,stellar_type=stellar_type.value,core_mass=core_mass.value,sse_initial_mass=sse_initial_mass.value, \
                        convective_envelope_mass=convective_envelope_mass.value, epoch=epoch.value, age=age.value, core_radius=core_radius.value, convective_envelope_radius=convective_envelope_radius.value, luminosity=luminosity.value, metallicity=metallicity.value, WD_He_layer_mass=WD_He_layer_mass.value, m_dot_accretion_SD=m_dot_accretion_SD.value)
                    p.index = internal_index
                    p.parent = parent.value
                    p.ospin = ospin.value
                    p.object_type = object_type.value

                    p.X = X.value
                    p.Y = Y.value
                    p.Z = Z.value
                    p.VX = VX.value
                    p.VY = VY.value
                    p.VZ = VZ.value
                    
                    p.spin_vec_x = spin_vec_x.value
                    p.spin_vec_y = spin_vec_y.value
                    p.spin_vec_z = spin_vec_z.value
                    
                    append = True
                elif integration_flag.value == 0:
                    parent,child1,child2 = ctypes.c_int(0),ctypes.c_int(0),ctypes.c_int(0)
                    mass,a,e,TA,INCL,AP,LAN,h_vec_x,h_vec_y,h_vec_z,e_vec_x,e_vec_y,e_vec_z = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
                    flag = self.lib.get_binary_properties_from_log_entry(index_log,internal_index,ctypes.byref(parent),ctypes.byref(child1),ctypes.byref(child2), \
                        ctypes.byref(mass), ctypes.byref(a),ctypes.byref(e),ctypes.byref(TA),ctypes.byref(INCL), ctypes.byref(AP), ctypes.byref(LAN), \
                        ctypes.byref(h_vec_x),ctypes.byref(h_vec_y),ctypes.byref(h_vec_z), \
                        ctypes.byref(e_vec_x),ctypes.byref(e_vec_y),ctypes.byref(e_vec_z))

                    p = Particle(is_binary=is_binary,mass=mass.value,child1=child1.value,child2=child2.value,a=a.value,e=e.value,TA=TA.value,INCL=INCL.value,AP=AP.value,LAN=LAN.value)
                    p.index = internal_index
                    p.parent = parent.value
                    p.child1_index=child1.value
                    p.child2_index=child2.value
                    p.mass = mass.value
                    
                    p.h_vec_x = h_vec_x.value
                    p.h_vec_y = h_vec_y.value
                    p.h_vec_z = h_vec_z.value

                    p.e_vec_x = e_vec_x.value
                    p.e_vec_y = e_vec_y.value
                    p.e_vec_z = e_vec_z.value
                    
                    append = True

                if append==True:
                    particles.append(p)

            for i,p in enumerate(particles):
                if p.is_binary == True:
                    i1 = [j for j in range(N_particles.value) if particles[j].index == p.child1_index][0]
                    i2 = [j for j in range(N_particles.value) if particles[j].index == p.child2_index][0]
                    #print("i1",i1,"i2",i2,"i",i,"p.child1_index",p.child1_index,"p.child2_index",p.child2_index)
                    p.child1 = particles[i1]
                    p.child2 = particles[i2]

            entry.update({'N_particles':len(particles)})
            entry.update({'particles':particles})
            log.append(entry)
        #print("log done")
        return log

    def write_final_log_entry(self):
        self.lib.write_final_log_entry_interface(self.model_time, self.integration_flag)
        self._invalidate_log_cache()

    @property
    def log(self):
        if self._log_cache is None:
            self._log_cache = self.__get_log()
        return self._log_cache

    def _invalidate_log_cache(self):
        self._log_cache = None


    ### Tests ###
    def unit_tests(self,mode):
        return self.lib.unit_tests_interface(mode)

    def determine_compact_object_merger_properties(self,m1,m2,chi1,chi2,spin_vec_1_unit,spin_vec_2_unit,h_vec_unit,e_vec_unit):
        v_recoil_vec_x,v_recoil_vec_y,v_recoil_vec_z = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
        alpha_vec_final_x,alpha_vec_final_y,alpha_vec_final_z = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
        M_final = ctypes.c_double(0.0)
        self.lib.determine_compact_object_merger_properties_interface( m1,m2,chi1,chi2,spin_vec_1_unit[0],spin_vec_1_unit[1],spin_vec_1_unit[2],spin_vec_2_unit[0],spin_vec_2_unit[1],spin_vec_2_unit[2], \
            h_vec_unit[0],h_vec_unit[1],h_vec_unit[2],e_vec_unit[0], e_vec_unit[1], e_vec_unit[2], \
            ctypes.byref(v_recoil_vec_x),ctypes.byref(v_recoil_vec_y),ctypes.byref(v_recoil_vec_z), \
            ctypes.byref(alpha_vec_final_x),ctypes.byref(alpha_vec_final_y),ctypes.byref(alpha_vec_final_z), \
            ctypes.byref(M_final) )
        v_recoil_vec = np.array( [v_recoil_vec_x.value,v_recoil_vec_y.value,v_recoil_vec_z.value] )
        alpha_vec_final = np.array( [alpha_vec_final_x.value,alpha_vec_final_y.value,alpha_vec_final_z.value] )
        M_final = M_final.value

        return v_recoil_vec,alpha_vec_final,M_final
        
    def test_sample_from_3d_maxwellian_distribution(self,sigma):
        vx,vy,vz = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
        self.lib.sample_from_3d_maxwellian_distribution_interface(sigma,ctypes.byref(vx),ctypes.byref(vy),ctypes.byref(vz))
        vx,vy,vz = vx.value, vy.value, vz.value
        return vx,vy,vz

    def test_sample_from_normal_distribution(self,mu,sigma):
        v = self.lib.sample_from_normal_distribution_interface(mu,sigma)
        return v

    def test_sample_from_kroupa_93_imf(self):
        m = self.lib.sample_from_kroupa_93_imf_interface()
        return m

    def test_sample_spherical_coordinates_unit_vectors_from_isotropic_distribution(self):
        r_hat_vec_x,r_hat_vec_y,r_hat_vec_z,theta_hat_vec_x,theta_hat_vec_y,theta_hat_vec_z,phi_hat_vec_x,phi_hat_vec_y,phi_hat_vec_z = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0), \
            ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0), ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
        self.lib.sample_spherical_coordinates_unit_vectors_from_isotropic_distribution_interface(ctypes.byref(r_hat_vec_x),ctypes.byref(r_hat_vec_y),ctypes.byref(r_hat_vec_z), \
            ctypes.byref(theta_hat_vec_x),ctypes.byref(theta_hat_vec_y),ctypes.byref(theta_hat_vec_z), \
            ctypes.byref(phi_hat_vec_x),ctypes.byref(phi_hat_vec_y),ctypes.byref(phi_hat_vec_z))

        r_hat_vec = np.array( [r_hat_vec_x.value, r_hat_vec_y.value, r_hat_vec_z.value] )
        theta_hat_vec = np.array( [theta_hat_vec_x.value, theta_hat_vec_y.value, theta_hat_vec_z.value] )
        phi_hat_vec = np.array( [phi_hat_vec_x.value, phi_hat_vec_y.value, phi_hat_vec_z.value] )
        return r_hat_vec,theta_hat_vec,phi_hat_vec

    def test_kick_velocity(self,kick_distribution,m):
        kw,v = ctypes.c_int(0),ctypes.c_double(0.0)
        self.lib.test_kick_velocity(kick_distribution,m,ctypes.byref(kw),ctypes.byref(v))
        return kw.value,v.value

    def test_flybys_perturber_sampling(self,R_enc,n_star,sigma_rel,M_int):
        M_per,b_vec_x,b_vec_y,b_vec_z,V_vec_x,V_vec_y,V_vec_z = ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0),ctypes.c_double(0.0)
        self.lib.test_flybys_perturber_sampling(R_enc,n_star,sigma_rel,M_int,ctypes.byref(M_per),ctypes.byref(b_vec_x),ctypes.byref(b_vec_y),ctypes.byref(b_vec_z),ctypes.byref(V_vec_x),ctypes.byref(V_vec_y),ctypes.byref(V_vec_z)) 
        b_vec = np.array( [ b_vec_x.value,b_vec_y.value,b_vec_z.value] )
        V_vec = np.array( [ V_vec_x.value,V_vec_y.value,V_vec_z.value] )
        return M_per.value,b_vec,V_vec

    def test_binary_evolution_SNe_Ia_single_degenerate_model_1_accumulation_efficiency(self, M_WD, accretion_rate, luminosity):
        eta,WD_accretion_mode = ctypes.c_double(0.0), ctypes.c_int(0)
        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_accumulation_efficiency(M_WD, accretion_rate, luminosity, ctypes.byref(eta), ctypes.byref(WD_accretion_mode))
        return eta.value, WD_accretion_mode.value

    def test_binary_evolution_SNe_Ia_single_degenerate_model_1_explosion(self, M_WD, accretion_rate, M_He, luminosity):
        explosion = ctypes.c_bool(False)
        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_explosion(M_WD, accretion_rate, M_He, luminosity, ctypes.byref(explosion))
        return explosion.value

    def test_binary_evolution_SNe_Ia_single_degenerate_model_1_white_dwarf_hydrogen_accretion_boundaries(self, M_WD):
        m_dot_lower,m_dot_upper = ctypes.c_double(0.0), ctypes.c_double(0.0)
        self.lib.test_binary_evolution_SNe_Ia_single_degenerate_model_1_white_dwarf_hydrogen_accretion_boundaries(M_WD, ctypes.byref(m_dot_lower), ctypes.byref(m_dot_upper))
        return m_dot_lower.value,m_dot_upper.value


    ### Constants ###
    @property
    def CONST_G(self):
        return self.__CONST_G

    @CONST_G.setter
    def CONST_G(self, value):
        self.__CONST_G = value
        self.__set_constants_in_code()
        
    @property
    def CONST_C(self):
        return self.__CONST_C

    @CONST_C.setter
    def CONST_C(self, value):
        self.__CONST_C = value
        self.__set_constants_in_code()

    @property
    def CONST_M_SUN(self):
        return self.__CONST_M_SUN

    @CONST_M_SUN.setter
    def CONST_M_SUN(self, value):
        self.__CONST_M_SUN = value
        self.__set_constants_in_code()

    @property
    def CONST_L_SUN(self):
        return self.__CONST_L_SUN

    @CONST_L_SUN.setter
    def CONST_L_SUN(self, value):
        self.__CONST_L_SUN = value
        self.__set_constants_in_code()

    @property
    def CONST_R_SUN(self):
        return self.__CONST_R_SUN

    @CONST_R_SUN.setter
    def CONST_R_SUN(self, value):
        self.__CONST_R_SUN = value
        self.__set_constants_in_code()

    @property
    def CONST_KM_PER_S(self):
        return self.__CONST_KM_PER_S

    @CONST_KM_PER_S.setter
    def CONST_KM_PER_S(self, value):
        self.__CONST_KM_PER_S = value
        self.__set_constants_in_code()

    @property
    def CONST_PER_PC3(self):
        return self.__CONST_PER_PC3

    @CONST_PER_PC3.setter
    def CONST_PER_PC3(self, value):
        self.__CONST_PER_PC3 = value
        self.__set_constants_in_code()

    @property
    def CONST_PARSEC(self):
        return self.__CONST_PARSEC

    @CONST_PARSEC.setter
    def CONST_PARSEC(self, value):
        self.__CONST_PARSEC = value
        self.__set_constants_in_code()

    @property
    def CONST_MJUP(self):
        return self.__CONST_MJUP

    @CONST_MJUP.setter
    def CONST_MJUP(self, value):
        self.__CONST_MJUP = value
        self.__set_constants_in_code()

    ##################
    ### Parameters ###
    ##################
    
    @property
    def relative_tolerance(self):
        return self.__relative_tolerance
    @relative_tolerance.setter
    def relative_tolerance(self, value):
        self.__relative_tolerance = value
        self.__set_parameters_in_code()

    @property
    def wall_time_max_s(self):
        return self.__wall_time_max_s
    @wall_time_max_s.setter
    def wall_time_max_s(self, value):
        self.__wall_time_max_s = value
        self.__set_parameters_in_code()

    @property
    def absolute_tolerance_eccentricity_vectors(self):
        return self.__absolute_tolerance_eccentricity_vectors
    @absolute_tolerance_eccentricity_vectors.setter
    def absolute_tolerance_eccentricity_vectors(self, value):
        self.__absolute_tolerance_eccentricity_vectors = value
        self.__set_parameters_in_code()

    @property
    def absolute_tolerance_spin_vectors(self):
        return self.__absolute_tolerance_spin_vectors
    @absolute_tolerance_spin_vectors.setter
    def absolute_tolerance_spin_vectors(self, value):
        self.__absolute_tolerance_spin_vectors = value
        self.__set_parameters_in_code()

    @property
    def absolute_tolerance_angular_momentum_vectors(self):
        return self.__absolute_tolerance_angular_momentum_vectors
    @absolute_tolerance_angular_momentum_vectors.setter
    def absolute_tolerance_angular_momentum_vectors(self, value):
        self.__absolute_tolerance_angular_momentum_vectors = value
        self.__set_parameters_in_code()

    @property
    def include_quadrupole_order_terms(self):
        return self.__include_quadrupole_order_terms
    @include_quadrupole_order_terms.setter
    def include_quadrupole_order_terms(self, value):
        self.__include_quadrupole_order_terms = value
        self.__set_parameters_in_code()

    @property
    def include_octupole_order_binary_pair_terms(self):
        return self.__include_octupole_order_binary_pair_terms
    @include_octupole_order_binary_pair_terms.setter
    def include_octupole_order_binary_pair_terms(self, value):
        self.__include_octupole_order_binary_pair_terms = value
        self.__set_parameters_in_code()

    @property
    def include_octupole_order_binary_triplet_terms(self):
        return self.__include_octupole_order_binary_triplet_terms
    @include_octupole_order_binary_triplet_terms.setter
    def include_octupole_order_binary_triplet_terms(self, value):
        self.__include_octupole_order_binary_triplet_terms = value
        self.__set_parameters_in_code()

    @property
    def include_hexadecupole_order_binary_pair_terms(self):
        return self.__include_hexadecupole_order_binary_pair_terms
    @include_hexadecupole_order_binary_pair_terms.setter
    def include_hexadecupole_order_binary_pair_terms(self, value):
        self.__include_hexadecupole_order_binary_pair_terms = value
        self.__set_parameters_in_code()

    @property
    def include_dotriacontupole_order_binary_pair_terms(self):
        return self.__include_dotriacontupole_order_binary_pair_terms
    @include_dotriacontupole_order_binary_pair_terms.setter
    def include_dotriacontupole_order_binary_pair_terms(self, value):
        self.__include_dotriacontupole_order_binary_pair_terms = value
        self.__set_parameters_in_code()

    @property
    def include_double_averaging_corrections(self):
        return self.__include_double_averaging_corrections
    @include_double_averaging_corrections.setter
    def include_double_averaging_corrections(self, value):
        self.__include_double_averaging_corrections = value
        self.__set_parameters_in_code()

#    @property
#    def include_VRR(self):
#        return self.__include_VRR
#    @include_VRR.setter
#    def include_VRR(self, value):
#        self.__include_VRR = value
#        self.__set_parameters_in_code()


    @property
    def random_seed(self):
        return self.__random_seed
    @random_seed.setter
    def random_seed(self, value):
        self.__random_seed = value
        self.__set_random_seed()

    @property
    def verbose_flag(self):
        return self.__verbose_flag
    @verbose_flag.setter
    def verbose_flag(self, value):
        self.__verbose_flag = value
        self.__set_verbose_flag()

    @property
    def stop_after_root_found(self):
        return self.__stop_after_root_found
    @stop_after_root_found.setter
    def stop_after_root_found(self, value):
        self.__stop_after_root_found = value
        self.__set_parameters_in_code()


    ### Flybys ###
    @property
    def include_flybys(self):
        return self.__include_flybys
    @include_flybys.setter
    def include_flybys(self, value):
        self.__include_flybys = value
        self.__set_parameters_in_code()

    @property
    def log_mstar_transitions(self):
        return self.__log_mstar_transitions
    @log_mstar_transitions.setter
    def log_mstar_transitions(self, value):
        self.__log_mstar_transitions = value
        self.__set_parameters_in_code()

    @property
    def flybys_reference_binary(self):
        return self.__flybys_reference_binary
    @flybys_reference_binary.setter
    def flybys_reference_binary(self, value):
        self.__flybys_reference_binary = value
        self.__set_parameters_in_code()
        
    @property
    def flybys_correct_for_gravitational_focussing(self):
        return self.__flybys_correct_for_gravitational_focussing
    @flybys_correct_for_gravitational_focussing.setter
    def flybys_correct_for_gravitational_focussing(self, value):
        self.__flybys_correct_for_gravitational_focussing = value
        self.__set_parameters_in_code()

    @property
    def flybys_velocity_distribution(self):
        return self.__flybys_velocity_distribution
    @flybys_velocity_distribution.setter
    def flybys_velocity_distribution(self, value):
        self.__flybys_velocity_distribution = value
        self.__set_parameters_in_code()

    @property
    def flybys_mass_distribution(self):
        return self.__flybys_mass_distribution
    @flybys_mass_distribution.setter
    def flybys_mass_distribution(self, value):
        self.__flybys_mass_distribution = value
        self.__set_parameters_in_code()

    @property
    def flybys_mass_distribution_lower_value(self):
        return self.__flybys_mass_distribution_lower_value
    @flybys_mass_distribution_lower_value.setter
    def flybys_mass_distribution_lower_value(self, value):
        self.__flybys_mass_distribution_lower_value = value
        self.__set_parameters_in_code()

    @property
    def flybys_mass_distribution_upper_value(self):
        return self.__flybys_mass_distribution_upper_value
    @flybys_mass_distribution_upper_value.setter
    def flybys_mass_distribution_upper_value(self, value):
        self.__flybys_mass_distribution_upper_value = value
        self.__set_parameters_in_code()

    @property
    def flybys_encounter_sphere_radius(self):
        return self.__flybys_encounter_sphere_radius
    @flybys_encounter_sphere_radius.setter
    def flybys_encounter_sphere_radius(self, value):
        self.__flybys_encounter_sphere_radius = value
        self.__set_parameters_in_code()

    @property
    def flybys_stellar_density(self):
        return self.__flybys_stellar_density
    @flybys_stellar_density.setter
    def flybys_stellar_density(self, value):
        self.__flybys_stellar_density = value
        self.__set_parameters_in_code()
        
    @property
    def flybys_stellar_relative_velocity_dispersion(self):
        return self.__flybys_stellar_relative_velocity_dispersion
    @flybys_stellar_relative_velocity_dispersion.setter
    def flybys_stellar_relative_velocity_dispersion(self, value):
        self.__flybys_stellar_relative_velocity_dispersion = value
        self.__set_parameters_in_code()

    
    @property
    def flybys_include_secular_encounters(self):
        return self.__flybys_include_secular_encounters
    @flybys_include_secular_encounters.setter
    def flybys_include_secular_encounters(self, value):
        self.__flybys_include_secular_encounters = value
        self.__set_parameters_in_code()


    ### N-body ###
    @property
    def MSTAR_gbs_tolerance_default(self):
        return self.__MSTAR_gbs_tolerance_default
    @MSTAR_gbs_tolerance_default.setter
    def MSTAR_gbs_tolerance_default(self, value):
        self.__MSTAR_gbs_tolerance_default = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_gbs_tolerance_kick(self):
        return self.__MSTAR_gbs_tolerance_kick
    @MSTAR_gbs_tolerance_kick.setter
    def MSTAR_gbs_tolerance_kick(self, value):
        self.__MSTAR_gbs_tolerance_kick = value
        self.__set_parameters_in_code()
        
    @property
    def MSTAR_collision_tolerance(self):
        return self.__MSTAR_collision_tolerance
    @MSTAR_collision_tolerance.setter
    def MSTAR_collision_tolerance(self, value):
        self.__MSTAR_collision_tolerance = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_output_time_tolerance(self):
        return self.__MSTAR_output_time_tolerance
    @MSTAR_output_time_tolerance.setter
    def MSTAR_output_time_tolerance(self, value):
        self.__MSTAR_output_time_tolerance = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_10(self):
        return self.__MSTAR_include_PN_acc_10
    @MSTAR_include_PN_acc_10.setter
    def MSTAR_include_PN_acc_10(self, value):
        self.__MSTAR_include_PN_acc_10 = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_20(self):
        return self.__MSTAR_include_PN_acc_20
    @MSTAR_include_PN_acc_20.setter
    def MSTAR_include_PN_acc_20(self, value):
        self.__MSTAR_include_PN_acc_20 = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_25(self):
        return self.__MSTAR_include_PN_acc_25
    @MSTAR_include_PN_acc_25.setter
    def MSTAR_include_PN_acc_25(self, value):
        self.__MSTAR_include_PN_acc_25 = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_30(self):
        return self.__MSTAR_include_PN_acc_30
    @MSTAR_include_PN_acc_30.setter
    def MSTAR_include_PN_acc_30(self, value):
        self.__MSTAR_include_PN_acc_30 = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_35(self):
        return self.__MSTAR_include_PN_acc_35
    @MSTAR_include_PN_acc_35.setter
    def MSTAR_include_PN_acc_35(self, value):
        self.__MSTAR_include_PN_acc_35 = value
        self.__set_parameters_in_code()
        
    @property
    def MSTAR_include_PN_acc_SO(self):
        return self.__MSTAR_include_PN_acc_SO
    @MSTAR_include_PN_acc_SO.setter
    def MSTAR_include_PN_acc_SO(self, value):
        self.__MSTAR_include_PN_acc_SO = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_SS(self):
        return self.__MSTAR_include_PN_acc_SS
    @MSTAR_include_PN_acc_SS.setter
    def MSTAR_include_PN_acc_SS(self, value):
        self.__MSTAR_include_PN_acc_SS = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_acc_Q(self):
        return self.__MSTAR_include_PN_acc_Q
    @MSTAR_include_PN_acc_Q.setter
    def MSTAR_include_PN_acc_Q(self, value):
        self.__MSTAR_include_PN_acc_Q = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_spin_SO(self):
        return self.__MSTAR_include_PN_spin_SO
    @MSTAR_include_PN_spin_SO.setter
    def MSTAR_include_PN_spin_SO(self, value):
        self.__MSTAR_include_PN_spin_SO = value
        self.__set_parameters_in_code()

    @property
    def MSTAR_include_PN_spin_SS(self):
        return self.__MSTAR_include_PN_spin_SS
    @MSTAR_include_PN_spin_SS.setter
    def MSTAR_include_PN_spin_SS(self, value):
        self.__MSTAR_include_PN_spin_SS = value
        self.__set_parameters_in_code()
        
    @property
    def MSTAR_include_PN_spin_Q(self):
        return self.__MSTAR_include_PN_spin_Q
    @MSTAR_include_PN_spin_Q.setter
    def MSTAR_include_PN_spin_Q(self, value):
        self.__MSTAR_include_PN_spin_Q = value
        self.__set_parameters_in_code()
        

    @property
    def nbody_analysis_fractional_semimajor_axis_change_parameter(self):
        return self.__nbody_analysis_fractional_semimajor_axis_change_parameter
    @nbody_analysis_fractional_semimajor_axis_change_parameter.setter
    def nbody_analysis_fractional_semimajor_axis_change_parameter(self, value):
        self.__nbody_analysis_fractional_semimajor_axis_change_parameter = value
        self.__set_parameters_in_code()

    @property
    def nbody_analysis_fractional_integration_time(self):
        return self.__nbody_analysis_fractional_integration_time
    @nbody_analysis_fractional_integration_time.setter
    def nbody_analysis_fractional_integration_time(self, value):
        self.__nbody_analysis_fractional_integration_time = value
        self.__set_parameters_in_code()

    @property
    def nbody_analysis_minimum_integration_time(self):
        return self.__nbody_analysis_minimum_integration_time
    @nbody_analysis_minimum_integration_time.setter
    def nbody_analysis_minimum_integration_time(self, value):
        self.__nbody_analysis_minimum_integration_time = value
        self.__set_parameters_in_code()

    @property
    def nbody_analysis_maximum_integration_time(self):
        return self.__nbody_analysis_maximum_integration_time
    @nbody_analysis_maximum_integration_time.setter
    def nbody_analysis_maximum_integration_time(self, value):
        self.__nbody_analysis_maximum_integration_time = value
        self.__set_parameters_in_code()

    @property
    def nbody_dynamical_instability_direct_integration_time_multiplier(self):
        return self.__nbody_dynamical_instability_direct_integration_time_multiplier
    @nbody_dynamical_instability_direct_integration_time_multiplier.setter
    def nbody_dynamical_instability_direct_integration_time_multiplier(self, value):
        self.__nbody_dynamical_instability_direct_integration_time_multiplier = value
        self.__set_parameters_in_code()

    @property
    def nbody_semisecular_direct_integration_time_multiplier(self):
        return self.__nbody_semisecular_direct_integration_time_multiplier
    @nbody_semisecular_direct_integration_time_multiplier.setter
    def nbody_semisecular_direct_integration_time_multiplier(self, value):
        self.__nbody_semisecular_direct_integration_time_multiplier = value
        self.__set_parameters_in_code()

    @property
    def nbody_supernovae_direct_integration_time_multiplier(self):
        return self.__nbody_supernovae_direct_integration_time_multiplier
    @nbody_supernovae_direct_integration_time_multiplier.setter
    def nbody_supernovae_direct_integration_time_multiplier(self, value):
        self.__nbody_supernovae_direct_integration_time_multiplier = value
        self.__set_parameters_in_code()

    @property
    def nbody_other_direct_integration_time_multiplier(self):
        return self.__nbody_other_direct_integration_time_multiplier
    @nbody_other_direct_integration_time_multiplier.setter
    def nbody_other_direct_integration_time_multiplier(self, value):
        self.__nbody_other_direct_integration_time_multiplier = value
        self.__set_parameters_in_code()

    @property
    def effective_radius_multiplication_factor_for_collisions_stars(self):
        return self.__effective_radius_multiplication_factor_for_collisions_stars
    @effective_radius_multiplication_factor_for_collisions_stars.setter
    def effective_radius_multiplication_factor_for_collisions_stars(self, value):
        self.__effective_radius_multiplication_factor_for_collisions_stars = value
        self.__set_parameters_in_code()

    @property
    def effective_radius_multiplication_factor_for_collisions_compact_objects(self):
        return self.__effective_radius_multiplication_factor_for_collisions_compact_objects
    @effective_radius_multiplication_factor_for_collisions_compact_objects.setter
    def effective_radius_multiplication_factor_for_collisions_compact_objects(self, value):
        self.__effective_radius_multiplication_factor_for_collisions_compact_objects = value
        self.__set_parameters_in_code()

    ### Binary evolution ###
    @property
    def binary_evolution_CE_energy_flag(self):
        return self.__binary_evolution_CE_energy_flag
    @binary_evolution_CE_energy_flag.setter
    def binary_evolution_CE_energy_flag(self, value):
        self.__binary_evolution_CE_energy_flag = value
        self.__set_parameters_in_code()
        
    @property
    def binary_evolution_CE_spin_flag(self):
        return self.__binary_evolution_CE_spin_flag
    @binary_evolution_CE_spin_flag.setter
    def binary_evolution_CE_spin_flag(self, value):
        self.__binary_evolution_CE_spin_flag = value
        self.__set_parameters_in_code()
        
    @property
    def binary_evolution_CE_recombination_fraction(self):
        return self.__binary_evolution_CE_recombination_fraction
    @binary_evolution_CE_recombination_fraction.setter
    def binary_evolution_CE_recombination_fraction(self, value):
        self.__binary_evolution_CE_recombination_fraction = value
        self.__set_parameters_in_code()

    @property
    def binary_evolution_mass_transfer_timestep_parameter(self):
        return self.__binary_evolution_mass_transfer_timestep_parameter
    @binary_evolution_mass_transfer_timestep_parameter.setter
    def binary_evolution_mass_transfer_timestep_parameter(self, value):
        self.__binary_evolution_mass_transfer_timestep_parameter = value
        self.__set_parameters_in_code()

    @property
    def binary_evolution_use_eCAML_model(self):
        return self.__binary_evolution_use_eCAML_model
    @binary_evolution_use_eCAML_model.setter
    def binary_evolution_use_eCAML_model(self, value):
        self.__binary_evolution_use_eCAML_model = value
        self.__set_parameters_in_code()


    @property
    def chandrasekhar_mass(self):
        return self.__chandrasekhar_mass
    @chandrasekhar_mass.setter
    def chandrasekhar_mass(self, value):
        self.__chandrasekhar_mass = value
        self.__set_parameters_in_code()

    @property
    def eddington_accretion_factor(self):
        return self.__eddington_accretion_factor
    @eddington_accretion_factor.setter
    def eddington_accretion_factor(self, value):
        self.__eddington_accretion_factor = value
        self.__set_parameters_in_code()

    @property
    def nova_accretion_factor(self):
        return self.__nova_accretion_factor
    @nova_accretion_factor.setter
    def nova_accretion_factor(self, value):
        self.__nova_accretion_factor = value
        self.__set_parameters_in_code()

    @property
    def alpha_wind_accretion(self):
        return self.__alpha_wind_accretion
    @alpha_wind_accretion.setter
    def alpha_wind_accretion(self, value):
        self.__alpha_wind_accretion = value
        self.__set_parameters_in_code()

    @property
    def beta_wind_accretion(self):
        return self.__beta_wind_accretion
    @beta_wind_accretion.setter
    def beta_wind_accretion(self, value):
        self.__beta_wind_accretion = value
        self.__set_parameters_in_code()

    @property
    def defining_upper_mass_for_sdB_formation(self):
        return self.__defining_upper_mass_for_sdB_formation
    @defining_upper_mass_for_sdB_formation.setter
    def defining_upper_mass_for_sdB_formation(self, value):
        self.__defining_upper_mass_for_sdB_formation = value
        self.__set_parameters_in_code()

    @property
    def binary_evolution_mass_transfer_model(self):
        return self.__binary_evolution_mass_transfer_model
    @binary_evolution_mass_transfer_model.setter
    def binary_evolution_mass_transfer_model(self, value):
        self.__binary_evolution_mass_transfer_model = value
        self.__set_parameters_in_code()

    @property
    def binary_evolution_SNe_Ia_single_degenerate_model(self):
        return self.__binary_evolution_SNe_Ia_single_degenerate_model
    @binary_evolution_SNe_Ia_single_degenerate_model.setter
    def binary_evolution_SNe_Ia_single_degenerate_model(self, value):
        self.__binary_evolution_SNe_Ia_single_degenerate_model = value
        self.__set_parameters_in_code()
        
    @property
    def binary_evolution_SNe_Ia_double_degenerate_model(self):
        return self.__binary_evolution_SNe_Ia_double_degenerate_model
    @binary_evolution_SNe_Ia_double_degenerate_model.setter
    def binary_evolution_SNe_Ia_double_degenerate_model(self, value):
        self.__binary_evolution_SNe_Ia_double_degenerate_model = value
        self.__set_parameters_in_code()

    @property
    def binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision(self):
        return self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision
    @binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision.setter
    def binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision(self, value):
        self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_eccentricity_for_eccentric_collision = value
        self.__set_parameters_in_code()
        
    @property
    def binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO(self):
        return self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO
    @binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO.setter
    def binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO(self, value):
        self.__binary_evolution_SNe_Ia_double_degenerate_model_minimum_primary_mass_CO_CO = value
        self.__set_parameters_in_code()
        
    
    ### Triple evolution ###
    @property
    def triple_mass_transfer_primary_star_accretion_efficiency_no_disk(self):
        return self.__triple_mass_transfer_primary_star_accretion_efficiency_no_disk
    @triple_mass_transfer_primary_star_accretion_efficiency_no_disk.setter
    def triple_mass_transfer_primary_star_accretion_efficiency_no_disk(self, value):
        self.__triple_mass_transfer_primary_star_accretion_efficiency_no_disk = value
        self.__set_parameters_in_code()

    @property
    def triple_mass_transfer_secondary_star_accretion_efficiency_no_disk(self):
        return self.__triple_mass_transfer_secondary_star_accretion_efficiency_no_disk
    @triple_mass_transfer_secondary_star_accretion_efficiency_no_disk.setter
    def triple_mass_transfer_secondary_star_accretion_efficiency_no_disk(self, value):
        self.__triple_mass_transfer_secondary_star_accretion_efficiency_no_disk = value
        self.__set_parameters_in_code()

    @property
    def triple_mass_transfer_primary_star_accretion_efficiency_disk(self):
        return self.__triple_mass_transfer_primary_star_accretion_efficiency_disk
    @triple_mass_transfer_primary_star_accretion_efficiency_disk.setter
    def triple_mass_transfer_primary_star_accretion_efficiency_disk(self, value):
        self.__triple_mass_transfer_primary_star_accretion_efficiency_disk = value
        self.__set_parameters_in_code()

    @property
    def triple_mass_transfer_secondary_star_accretion_efficiency_disk(self):
        return self.__triple_mass_transfer_secondary_star_accretion_efficiency_disk
    @triple_mass_transfer_secondary_star_accretion_efficiency_disk.setter
    def triple_mass_transfer_secondary_star_accretion_efficiency_disk(self, value):
        self.__triple_mass_transfer_secondary_star_accretion_efficiency_disk = value
        self.__set_parameters_in_code()

    @property
    def triple_mass_transfer_inner_binary_alpha_times_lambda(self):
        return self.__triple_mass_transfer_inner_binary_alpha_times_lambda
    @triple_mass_transfer_inner_binary_alpha_times_lambda.setter
    def triple_mass_transfer_inner_binary_alpha_times_lambda(self, value):
        self.__triple_mass_transfer_inner_binary_alpha_times_lambda = value
        self.__set_parameters_in_code()

    @property
    def NS_model(self):
        return self.__NS_model
    @NS_model.setter
    def NS_model(self, value):
        self.__NS_model = value
        self.__set_parameters_in_code()

    @property
    def ECSNe_model(self):
        return self.__ECSNe_model
    @ECSNe_model.setter
    def ECSNe_model(self, value):
        self.__ECSNe_model = value
        self.__set_parameters_in_code()

    @property
    def system_index(self):
        return self.__system_index
    @system_index.setter
    def system_index(self, value):
        self.__system_index = value
        self.__set_parameters_in_code()

class Particle(object):
    def __init__(self, is_binary, mass=None, mass_dot=0.0, radius=1.0e-10, radius_dot=0.0, child1=None, child2=None, a=None, e=None, TA=0.0, INCL=None, AP=None, LAN=None, \
            integration_method = 0, KS_use_perturbing_potential = True, \
            stellar_type=1, object_type=1, sse_initial_mass=None, metallicity=0.02, sse_time_step=1.0, epoch=0.0, age=0.0, core_mass=0.0, core_radius=0.0, \
            include_mass_transfer_terms=True, \
            kick_distribution = 1, include_WD_kicks = False, kick_distribution_sigma_km_s_NS = 265.0, kick_distribution_sigma_km_s_BH=50.0, kick_distribution_sigma_km_s_WD = 1.0, \
            kick_distribution_2_m_NS=1.4, kick_distribution_4_m_NS=1.2, kick_distribution_4_m_ej=9.0, kick_distribution_5_v_km_s_NS=400.0,kick_distribution_5_v_km_s_BH=200.0, kick_distribution_5_sigma=0.3, kick_distribution_sigma_km_s_NS_ECSN = 20.0, \
            spin_vec_x=0.0, spin_vec_y=0.0, spin_vec_z=1.0e-10, \
            include_pairwise_1PN_terms=True, include_pairwise_25PN_terms=True, include_spin_orbit_1PN_terms=True, exclude_1PN_precession_in_case_of_isolated_binary=True, \
            include_tidal_friction_terms=True, tides_method=1, include_tidal_bulges_precession_terms=True, include_rotation_precession_terms=True, exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = True, \
            minimum_eccentricity_for_tidal_precession = 1.0e-3, apsidal_motion_constant=0.19, gyration_radius=0.08, tides_viscous_time_scale=1.0e100, tides_viscous_time_scale_prescription=1, \
            convective_envelope_mass=1.0e-10, convective_envelope_radius=1.0e-10, luminosity=1.0e-10, \
            check_for_secular_breakdown=True,check_for_dynamical_instability=True,dynamical_instability_criterion=0,dynamical_instability_central_particle=0,dynamical_instability_K_parameter=0, \
            check_for_physical_collision_or_orbit_crossing=True,check_for_minimum_periapse_distance=False,check_for_minimum_periapse_distance_value=0.0,check_for_RLOF_at_pericentre=True,check_for_RLOF_at_pericentre_use_sepinsky_fit=False, check_for_GW_condition=False, check_for_entering_LISA_band=True, check_for_entering_LISA_band_critical_GW_frequency=31557.6, \
            secular_breakdown_has_occurred=False, dynamical_instability_has_occurred=False, physical_collision_or_orbit_crossing_has_occurred=False, minimum_periapse_distance_has_occurred=False, RLOF_at_pericentre_has_occurred = False, GW_condition_has_occurred = False, entering_LISA_band_has_occurred=False, \
            is_external=False, external_t_ref=0.0, external_r_p=0.0, \
            sample_orbital_phase_randomly=False, instantaneous_perturbation_delta_mass=0.0, instantaneous_perturbation_delta_X=0.0, instantaneous_perturbation_delta_Y=0.0, instantaneous_perturbation_delta_Z=0.0, \
            instantaneous_perturbation_delta_VX=0.0, instantaneous_perturbation_delta_VY=0.0, instantaneous_perturbation_delta_VZ=0.0, \
            VRR_model=0, VRR_include_mass_precession=0, VRR_mass_precession_rate=0.0, VRR_Omega_vec_x=0.0, VRR_Omega_vec_y=0.0, VRR_Omega_vec_z=0.0, \
            VRR_eta_20_init=0.0, VRR_eta_a_22_init=0.0, VRR_eta_b_22_init=0.0, VRR_eta_a_21_init=0.0, VRR_eta_b_21_init=0.0, \
            VRR_eta_20_final=0.0, VRR_eta_a_22_final=0.0, VRR_eta_b_22_final=0.0, VRR_eta_a_21_final=0.0, VRR_eta_b_21_final=0.0, \
            VRR_initial_time = 0.0, VRR_final_time = 1.0,roche_lobe_radius_pericenter=0.0, \
            dynamical_mass_transfer_low_mass_donor_timescale=1.0e2, dynamical_mass_transfer_WD_donor_timescale=1.0e2, compact_object_disruption_mass_loss_timescale=1.0e2, common_envelope_alpha=1.0, common_envelope_lambda=1.0, common_envelope_timescale=1.0e2, triple_common_envelope_alpha=1.0, WD_He_layer_mass=0.0, m_dot_accretion_SD=0.0):  # [P5.1] mass-loss timescales: paper default=1e2 yr; was 1e3
                
        ### spin_vec: nonzero spin_vec_z: need to specify a finite initial direction 

        if is_binary==None:
            raise RuntimeError('Error when adding particle: particle should have property is_binary')

        self.is_external = is_external
        if is_external==True:
            is_binary = False ### for is_binary to true for external particles
            self.external_t_ref = external_t_ref
            self.external_r_p = external_r_p
            self.mass = mass
            self.e = e
            self.INCL = INCL
            self.AP = AP
            self.LAN = LAN


        self.index = None
        self.is_binary = is_binary

        self.include_tidal_friction_terms=include_tidal_friction_terms
        self.tides_method=tides_method
        self.include_tidal_bulges_precession_terms=include_tidal_bulges_precession_terms
        self.include_rotation_precession_terms=include_rotation_precession_terms
        self.minimum_eccentricity_for_tidal_precession=minimum_eccentricity_for_tidal_precession
        self.tides_viscous_time_scale=tides_viscous_time_scale
        self.tides_viscous_time_scale_prescription=tides_viscous_time_scale_prescription
        self.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = exclude_rotation_and_bulges_precession_in_case_of_isolated_binary

        self.include_pairwise_1PN_terms = include_pairwise_1PN_terms
        self.include_pairwise_25PN_terms = include_pairwise_25PN_terms
        self.include_spin_orbit_1PN_terms = include_spin_orbit_1PN_terms
        self.exclude_1PN_precession_in_case_of_isolated_binary = exclude_1PN_precession_in_case_of_isolated_binary
        
        self.include_mass_transfer_terms = include_mass_transfer_terms

        self.kick_distribution = kick_distribution
        self.include_WD_kicks = include_WD_kicks
        self.kick_distribution_sigma_km_s_NS = kick_distribution_sigma_km_s_NS
        self.kick_distribution_sigma_km_s_BH = kick_distribution_sigma_km_s_BH
        self.kick_distribution_sigma_km_s_WD = kick_distribution_sigma_km_s_WD
        self.kick_distribution_2_m_NS = kick_distribution_2_m_NS
        self.kick_distribution_4_m_NS = kick_distribution_4_m_NS
        self.kick_distribution_4_m_ej = kick_distribution_4_m_ej
        self.kick_distribution_5_v_km_s_NS = kick_distribution_5_v_km_s_NS
        self.kick_distribution_5_v_km_s_BH = kick_distribution_5_v_km_s_BH
        self.kick_distribution_5_sigma = kick_distribution_5_sigma
        self.kick_distribution_sigma_km_s_NS_ECSN = kick_distribution_sigma_km_s_NS_ECSN
          
        self.check_for_secular_breakdown=check_for_secular_breakdown
        self.check_for_dynamical_instability=check_for_dynamical_instability
        self.dynamical_instability_criterion=dynamical_instability_criterion
        self.dynamical_instability_central_particle=dynamical_instability_central_particle
        self.dynamical_instability_K_parameter=dynamical_instability_K_parameter
        self.check_for_physical_collision_or_orbit_crossing=check_for_physical_collision_or_orbit_crossing
        self.check_for_minimum_periapse_distance=check_for_minimum_periapse_distance
        self.check_for_minimum_periapse_distance_value=check_for_minimum_periapse_distance_value
        self.check_for_RLOF_at_pericentre=check_for_RLOF_at_pericentre
        self.check_for_RLOF_at_pericentre_use_sepinsky_fit=check_for_RLOF_at_pericentre_use_sepinsky_fit
        self.check_for_GW_condition=check_for_GW_condition
        self.check_for_entering_LISA_band = check_for_entering_LISA_band
        self.check_for_entering_LISA_band_critical_GW_frequency = check_for_entering_LISA_band_critical_GW_frequency

        self.secular_breakdown_has_occurred=secular_breakdown_has_occurred
        self.dynamical_instability_has_occurred=dynamical_instability_has_occurred
        self.physical_collision_or_orbit_crossing_has_occurred=physical_collision_or_orbit_crossing_has_occurred
        self.minimum_periapse_distance_has_occurred=minimum_periapse_distance_has_occurred
        self.RLOF_at_pericentre_has_occurred=RLOF_at_pericentre_has_occurred
        self.GW_condition_has_occurred=GW_condition_has_occurred
        self.entering_LISA_band_has_occurred = entering_LISA_band_has_occurred

        self.sample_orbital_phase_randomly=sample_orbital_phase_randomly
        self.instantaneous_perturbation_delta_mass=instantaneous_perturbation_delta_mass
        self.instantaneous_perturbation_delta_X=instantaneous_perturbation_delta_X
        self.instantaneous_perturbation_delta_Y=instantaneous_perturbation_delta_Y
        self.instantaneous_perturbation_delta_Z=instantaneous_perturbation_delta_Z
        self.instantaneous_perturbation_delta_VX=instantaneous_perturbation_delta_VX
        self.instantaneous_perturbation_delta_VY=instantaneous_perturbation_delta_VY
        self.instantaneous_perturbation_delta_VZ=instantaneous_perturbation_delta_VZ

        self.VRR_model = VRR_model
        self.VRR_include_mass_precession = VRR_include_mass_precession
        self.VRR_mass_precession_rate = VRR_mass_precession_rate
        self.VRR_Omega_vec_x = VRR_Omega_vec_x
        self.VRR_Omega_vec_y = VRR_Omega_vec_y
        self.VRR_Omega_vec_z = VRR_Omega_vec_z
        self.VRR_eta_20_init = VRR_eta_20_init
        self.VRR_eta_a_22_init = VRR_eta_a_22_init
        self.VRR_eta_b_22_init = VRR_eta_b_22_init
        self.VRR_eta_a_21_init = VRR_eta_a_21_init
        self.VRR_eta_b_21_init = VRR_eta_b_21_init
        self.VRR_eta_20_final = VRR_eta_20_final
        self.VRR_eta_a_22_final = VRR_eta_a_22_final
        self.VRR_eta_b_22_final = VRR_eta_b_22_final
        self.VRR_eta_a_21_final = VRR_eta_a_21_final
        self.VRR_eta_b_21_final = VRR_eta_b_21_final
        self.VRR_initial_time = VRR_initial_time
        self.VRR_final_time = VRR_final_time

        self.dynamical_mass_transfer_low_mass_donor_timescale = dynamical_mass_transfer_low_mass_donor_timescale
        self.dynamical_mass_transfer_WD_donor_timescale = dynamical_mass_transfer_WD_donor_timescale
        self.compact_object_disruption_mass_loss_timescale = compact_object_disruption_mass_loss_timescale
        self.common_envelope_alpha = common_envelope_alpha
        self.common_envelope_lambda = common_envelope_lambda
        self.common_envelope_timescale = common_envelope_timescale
        self.triple_common_envelope_alpha = triple_common_envelope_alpha

        if is_binary==False:
            if mass==None:
                raise RuntimeError('Error when adding particle: body should have mass specified') 
            self.mass = mass
            self.mass_dot = mass_dot
            self.stellar_type = stellar_type
            self.object_type = object_type
            self.sse_initial_mass = sse_initial_mass if sse_initial_mass is not None else mass  # [C46/H12]
            self.metallicity = metallicity
            self.sse_time_step = sse_time_step
            self.epoch = epoch
            self.age = age
            self.core_mass = core_mass
            self.core_radius = core_radius
            self.child1 = None
            self.child2 = None
            self.radius = radius
            self.radius_dot = radius_dot
            self.spin_vec_x = spin_vec_x
            self.spin_vec_y = spin_vec_y
            self.spin_vec_z = spin_vec_z
            self.apsidal_motion_constant=apsidal_motion_constant
            self.gyration_radius=gyration_radius
            self.convective_envelope_mass=convective_envelope_mass
            self.convective_envelope_radius=convective_envelope_radius
            self.luminosity=luminosity
            self.roche_lobe_radius_pericenter = roche_lobe_radius_pericenter
            self.WD_He_layer_mass = WD_He_layer_mass
            self.m_dot_accretion_SD = m_dot_accretion_SD
        
        else:
            if is_external==False:
                #if child1==None or child2==None:
                #    raise RuntimeError('Error when adding particle: a binary should have two children!')
                if a==None or e==None or INCL==None or LAN==None:
                    raise RuntimeError('Error when adding particle: a binary should have its orbital elements specified!')
                else:
                    self.child1 = child1
                    self.child2 = child2
                    #self.mass = child1.mass + child2.mass

                    self.a = a
                    self.e = e
                    self.TA = TA
                    self.INCL = INCL
                    self.AP = AP
                    self.LAN = LAN
                    
                    self.integration_method = integration_method
                    self.KS_use_perturbing_potential = KS_use_perturbing_potential
                    
    def __repr__(self):

        if self.index is None:
            if self.is_binary == False:
                return "Particle(is_binary={0}, mass={1:g}, stellar_type={2:d})".format(self.is_binary,self.mass,self.stellar_type)
            else:
                #return "Particle(is_binary={0}, child1={1:d}, child2={2:d}, a={3:g}, e={4:g}, INCL={5:g}, AP={6:g}, LAN={7:g})".format(self.is_binary,self.child1,self.child2,self.a,self.e,self.INCL,self.AP,self.LAN)
                return "Particle(is_binary={0})".format(self.is_binary)
        else:
            if self.is_binary == False:
                return "Particle(is_binary={0}, index={1:d}, mass={2:g}, stellar_type={3:d})".format(self.is_binary,self.index,self.mass,self.stellar_type)
            else:
                return "Particle(is_binary={0}, index={1:d}, child1={2:d}, child2={3:d}, a={4:g}, e={5:g}, INCL={6:g}, AP={7:g}, LAN={8:g})".format(self.is_binary,self.index,self.child1.index,self.child2.index,self.a,self.e,self.INCL,self.AP,self.LAN)


class Tools(object):

    @staticmethod
    def check_for_default_values(N_bodies,metallicities,stellar_types,object_types,inclinations,longitudes_of_ascending_node,arguments_of_pericentre):
        if stellar_types == []:
            for i in range(N_bodies):
                stellar_types.append(1)
            print("mse.py -- stellar_types not explicitly given -- setting initial stellar types to",stellar_types)

        if object_types == []:
            for i in range(N_bodies):
                object_types.append(1)
            print("mse.py -- object_types not explicitly given -- setting initial object types to",object_types)

        if metallicities == []:
            for i in range(N_bodies):
                metallicities.append(0.02)
            print("mse.py -- metallicities not explicitly given -- setting initial metallicities to",metallicities)
     
        if inclinations == []:
            for i in range(N_bodies-1):
                inclinations.append(np.arccos(np.random.random()))
            print("mse.py -- inclinations not explicitly given -- setting initial inclinations to",inclinations)
            
        if longitudes_of_ascending_node == []:
            for i in range(N_bodies-1):
                longitudes_of_ascending_node.append(2.0*np.pi * np.random.random())
            print("mse.py -- longitudes_of_ascending_node not explicitly given -- setting initial longitudes_of_ascending_node to",longitudes_of_ascending_node)
            
        if arguments_of_pericentre == []:
            for i in range(N_bodies-1):
                arguments_of_pericentre.append(2.0*np.pi * np.random.random())
            print("mse.py -- arguments_of_pericentre not explicitly given -- setting initial arguments_of_pericentre to",arguments_of_pericentre)
     
    @staticmethod       
    def create_fully_nested_multiple(N,masses,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,radii=None,metallicities=None,stellar_types=None,object_types=None):  # [C49] mutable defaults

        """
        N is number of bodies
        masses should be N-sized array
        the other arguments should be (N-1)-sized arrays
        """

        if metallicities is None: metallicities = []
        if stellar_types is None: stellar_types = []
        if object_types is None: object_types = []

        N_bodies = N
        N_binaries = N-1

        Tools.check_for_default_values(N_bodies,metallicities,stellar_types,object_types,inclinations,longitudes_of_ascending_node,arguments_of_pericentre)

        particles = []

        for index in range(N_bodies):
            particle = Particle(is_binary=False,mass=masses[index])
            if radii is not None:
                particle.radius = radii[index]

            particle.metallicity = metallicities[index]
            particle.stellar_type = stellar_types[index]
            particle.object_type = object_types[index]
                
            particles.append(particle)

        for index in range(N_binaries):
            if index==0:
                child1 = particles[0]
                child2 = particles[1]
            else:
                child1 = previous_binary
                child2 = particles[index+1]
            particle = Particle(is_binary=True,child1=child1,child2=child2,a=semimajor_axes[index],e=eccentricities[index],INCL=inclinations[index],AP=arguments_of_pericentre[index],LAN=longitudes_of_ascending_node[index])

            previous_binary = particle
            particles.append(particle)
            
        return particles

    @staticmethod
    def create_2p2_quadruple_system(masses,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,radii=None,metallicities=None,stellar_types=None,object_types=None):  # [C49] mutable defaults

        """
        Create a 2+2 quadruple system.
        Masses should contain the four masses.
        The other arguments should be length 3 arrays; first two entries: the two inner binaries; third entry: outer binary.
        """

        if metallicities is None: metallicities = []
        if stellar_types is None: stellar_types = []
        if object_types is None: object_types = []

        N_bodies = 4
        N_binaries = N_bodies-1

        Tools.check_for_default_values(N_bodies,metallicities,stellar_types,object_types,inclinations,longitudes_of_ascending_node,arguments_of_pericentre)

        particles = []

        ### Add the bodies ###
        for index in range(N_bodies):
            particle = Particle(is_binary=False,mass=masses[index])
            if radii is not None:
                particle.radius = radii[index]

            particle.metallicity = metallicities[index]
            particle.stellar_type = stellar_types[index]
            particle.object_type = object_types[index]
            
            particles.append(particle)

        ### Add the binaries ###
        for index in range(N_binaries):
            if index==0:
                child1 = particles[0]
                child2 = particles[1]
            elif index==1:
                child1 = particles[2]
                child2 = particles[3]
            elif index==2:
                child1 = particles[4]
                child2 = particles[5]

            particle = Particle(is_binary=True,child1=child1,child2=child2,a=semimajor_axes[index],e=eccentricities[index],INCL=inclinations[index],AP=arguments_of_pericentre[index],LAN=longitudes_of_ascending_node[index])
            particles.append(particle)
        
        return particles

   
    @staticmethod
    def parse_config(N_bodies,configuration):
        # convert input string '{num+[others]}' to the basic '[1,...[1,[others]]...]' list string 
        # for first encounter of '+' (and corresponding '{' and '}') only => call multiple times
        def first_plus_to_nested(old_str):
            err = 0
            # first occurances of '+' and '{'
            first_plus = old_str.find('+')
            first_left = old_str.find('{')
            # check ordering
            if first_left < first_plus:                
                chars = old_str[first_left+1:first_plus]
                # check if characters between '{' and '+' are int 
                try:
                    num = int(chars)                
                except:
                    err = 1
                    return '', err
                # variable to identify the '}' corresponding to '{'
                proper_brkt = -1
                for ind, char in enumerate(old_str):  # [C8] iterate over local parameter, not closure variable
                    if ind == first_left:
                        proper_brkt = 0
                        continue
                    # case of further nested '{' and '}'
                    if char == '{':
                        proper_brkt += 1
                    elif char == '}':
                        # error if '}' is found before '{'
                        if proper_brkt < 0:
                            err = 2
                            return '', err
                        # first corresponding '}' found
                        elif proper_brkt == 0:
                            first_right = ind
                            break
                        # case of further nested '{' and '}'
                        elif proper_brkt > 0:
                            proper_brkt -= 1
                # replacing '{num+' and '}' num times and writing in new string
                replace_left = ''
                replace_right = ''            
                for i in range(num):            
                    replace_left += '[1,'
                    replace_right += ']'
                new_str = old_str[:first_left] + replace_left + old_str[first_plus+1:first_right] + replace_right + old_str[first_right+1:] 
                return new_str, err 
            # error if placement of '{' and '+' is wrong                
            else:
                err = 3
                return '', err        

        # check if parsed list contains either list or int elements
        def list_is_int(lst):
            if type(lst) == int:            
                is_int = True
            # children of list need to be tested recursively
            else:
                is_int = True
                for elem in lst:
                    if type(elem) == list:  
                        is_int = list_is_int(elem)
                    elif type(elem) == int:
                        is_int = True
                    else:
                        is_int = False
                        break   
            return is_int

        # check if parsed list is a binary tree - each level has either two children (node) or one (body)
        def list_is_binary(lst):
            if type(lst) == int:            
                is_binary = True
            # children of list need to be tested recursively
            elif len(lst) == 2:
                is_binary = True
                for elem in lst:
                    if type(elem) == list:
                        if len(elem) == 2:
                            is_binary = list_is_binary(elem)
                        else:
                            is_binary = False
                            break
            else:
                is_binary = False 
            return is_binary

        # convert parsed list [num,others] to basic [[1,...[1,1]],others] containg only ones
        def convert_to_ones(lst):
            # fullly nested hierachy if input is int
            if type(lst) == int:
                if lst == 1:  # [C7] base case: single body
                    return 1
                for i in range(lst):
                    if i == 0:
                        continue
                    elif i == 1:
                        sub_lst = [1,1]
                    else:
                        sub_lst = [1]+[sub_lst]
                lst = sub_lst
            else:
                for ind, elem in enumerate(lst):
                    if type(elem) == list:
                        lst[ind] = convert_to_ones(elem)  # [C7] propagate recursive result
                    elif elem == 1:
                        continue
                    elif type(elem) == int:
                        for i in range(elem):
                            if i == 0:
                                continue
                            # smallest child element with 2 bodies
                            elif i == 1:
                                sub_lst = [1,1]
                            # adding extra body to previous hierarchy
                            else:
                                sub_lst = [1]+[sub_lst]
                        lst[ind] = sub_lst
            return lst

        # find total number of bodies in a arbitrary nested list containing only ones
        def nested_list_size(lst):
            if type(lst) == int:            
                count = lst
            # iterate through children lists if they exist    
            else:
                count = 0            
                for elem in lst:
                    if type(elem) == list:  
                        count += nested_list_size(elem)
                    else:
                        count += 1    
            return count  

        # remove all spaces in string
        configuration = configuration.replace(' ','')
        # number of '+' should equal number of '{' and '}'
        num_plus = configuration.count('+')
        num_curl_left = configuration.count('{')
        num_curl_right = configuration.count('}')
        if num_curl_left == num_plus and num_curl_right == num_plus:
            while num_plus != 0:
                # convert sting notation with '+','{','}' to list string, and get error if any                    
                configuration, error = first_plus_to_nested(configuration)
                if configuration == '' and error == 1:
                    print("Value between '{' and '+' should be int. Input a valid configuration.")
                    print("Exiting...")
                    exit()
                elif configuration == '' and error == 2:
                    print("'{' should occur before '}'. Input a valid configuration.")
                    print("Exiting...")
                    exit()
                elif configuration == '' and error == 3:
                    print("'{' should occur before '+'. Input a valid configuration.")
                    print("Exiting...")
                    exit()
                num_plus -= 1
        else:
            print("Wrong number of curly brackets. Input a valid configuration.")
            print("Exiting...")
            exit()
        
        # check if string can be parsed to a meaningful list
        try:
            # function which parses string to Python expression
            lst = ast.literal_eval(configuration)
        except:
            print("String could not be parsed to meaningful list. Input a valid configuration.")
            print("Exiting...")
            exit()
        # required conditions to convert parsed list to contain only ones
        if list_is_binary(lst):
            if list_is_int(lst):
                lst = convert_to_ones(lst)   
                # check if configuration agrees with given number of bodies   
                if nested_list_size(lst) == N_bodies:
                    return lst
                else:
                    print("Number of elements incorrect. Input a valid configuration.")
                    print("Exiting...")
                    exit()
            else:
                print("Data types of elements should be int. Input a valid configuration.")
                print("Exiting...")
                exit()
        else:
            print("Configuration should be int or binary list. Input a valid configuration.")
            print("Exiting...")
            exit()

    @staticmethod
    def create_hierarchy(N_bodies,configuration,masses,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,radii=None,metallicities=None,stellar_types=None,object_types=None):  # [C49] mutable defaults

        if metallicities is None: metallicities = []
        if stellar_types is None: stellar_types = []
        if object_types is None: object_types = []

        Tools.check_for_default_values(N_bodies,metallicities,stellar_types,object_types,inclinations,longitudes_of_ascending_node,arguments_of_pericentre)

        print("="*50)
        print("mse.py -- parsing given configuration into particles")

        config_list = Tools.parse_config(N_bodies, configuration)
        print("Verbose configuration :",config_list)
        print()

        N_binaries = N_bodies-1

        particles = []

        for index in range(N_bodies):
            particle = Particle(is_binary=False,mass=masses[index])
            if radii is not None:
                particle.radius = radii[index]

            particle.metallicity = metallicities[index]
            particle.stellar_type = stellar_types[index]
            particle.object_type = object_types[index]

            print("Particle id :",index)
            particles.append(particle)
        print()

        def nested_iteration(lst): 
            N_subbinary = 0   
            lst_tpl = ()
            N_subbinary_tpl = ()
            # iterate through binary list backward (rightmost first)
            for ind,elem in reversed(list(enumerate(lst))):        
                if type(elem) == list:     
                    ret_N, ret_lst_tpl, ret_N_tpl = nested_iteration(elem)
                    N_subbinary += ret_N 
                    lst_tpl += ret_lst_tpl
                    N_subbinary_tpl += ret_N_tpl
            N_subbinary += 1
            lst_tpl += (lst,)
            N_subbinary_tpl += (N_subbinary,) 
            return N_subbinary, lst_tpl, N_subbinary_tpl

        # lst_tpl : tuple of each of the N_binaries, starting from the rightmost in tree diagram [not used in the program; visual confirmation only]
        # N_subbinaries_tpl : tuple of number of subbinaries in each of the N_binaries (corresponding to lst_tpl) 
        _, lst_tpl, N_subbinary_tpl = nested_iteration(config_list)
        print("Tuple of each of the",N_binaries,"binaries (starting from rightmost, least hierarchy) :",lst_tpl)
        print()
        print("Tuple of number of subbinaries in each of the",N_binaries,"binaries (starting from rightmost, least hierarchy) :",N_subbinary_tpl)
        print()

        for index in range(N_binaries):
            if index == 0:
                child1 = particles[0]
                child2 = particles[1]
                # particle id only for printing; redundant otherwise
                id1 = 0
                id2 = 1

                particle_id = 2
                binary_id = N_bodies

            elif N_subbinary_tpl[index] == 1 and N_subbinary_tpl[index-1] >= 1:
                child1 = particles[particle_id]
                child2 = particles[particle_id+1]
                # particle id only for printing; redundant otherwise
                id1 = particle_id
                id2 = particle_id+1

                old_binary_id = binary_id
                particle_id += 2
                binary_id += 1

            elif N_subbinary_tpl[index] == N_subbinary_tpl[index-1]+1:
                child1 = particles[binary_id]
                child2 = particles[particle_id]
                # particle id only for printing; redundant otherwise
                id1 = binary_id
                id2 = particle_id

                particle_id += 1
                binary_id += 1

            elif N_subbinary_tpl[index] > N_subbinary_tpl[index-1]+1:
                child1 = particles[old_binary_id]
                child2 = particles[binary_id]
                # particle id only for printing; redundant otherwise
                id1 = old_binary_id
                id2 = binary_id

                binary_id += 1

            print("Particle id :",binary_id,"\tChild 1 id :",id1,"\tChild 2 id :",id2)
            particle = Particle(is_binary=True,child1=child1,child2=child2,a=semimajor_axes[index],e=eccentricities[index],INCL=inclinations[index],AP=arguments_of_pericentre[index],LAN=longitudes_of_ascending_node[index])
            particles.append(particle)

        return particles         


    @staticmethod
    def compute_mutual_inclination(INCL_k,INCL_l,LAN_k,LAN_l):
        cos_INCL_rel = np.cos(INCL_k)*np.cos(INCL_l) + np.sin(INCL_k)*np.sin(INCL_l)*np.cos(LAN_k-LAN_l)
        return np.arccos(cos_INCL_rel)

    @staticmethod
    def compute_effective_temperature(luminosity, radius, CONST_L_SUN, CONST_R_SUN):
        """
        Assumes black body radiation.
        Luminosity and radius should be in standard code units.
        Returns T_eff in K
        """
        
        T_Sun = 5770.0 ### (the Sun knows about ATI's old product stack)
        T_eff = T_Sun * pow(luminosity/CONST_L_SUN,0.25) * pow(radius/CONST_R_SUN,-0.5)
        return T_eff
    
    @staticmethod
    def determine_binary_masses(particles):
        ### set binary masses -- to ensure this happens correctly, do this from highest level to lowest level ###

        Tools.determine_binary_levels_in_particles(particles)

        max_level = np.amax([x.level for x in particles])
        level = max_level
        while (level > -1):
            for index,p in enumerate(particles):
                if (p.is_binary == True and p.level == level):
                    p.mass = p.child1.mass + p.child2.mass
            level -= 1

    @staticmethod
    def determine_binary_levels_in_particles(particles):
        for index,p in enumerate(particles):
            p.index_temp = index
            p.parent = None

        ### determine top binary ###
        for index_particle_1,particle_1 in enumerate(particles):
            if particle_1.is_binary == True:
                child1 = particle_1.child1
                child2 = particle_1.child2
                
                for index_particle_2,particle_2 in enumerate(particles):
                    if (index_particle_2 == child1.index_temp or index_particle_2 == child2.index_temp):
                        particle_2.parent = particle_1
                        
        for index_particle_1,particle_1 in enumerate(particles):
            particle_1.level = 0
            
            child = particle_1;
            parent = particle_1.parent

            if (parent != None): ### if parent == -1, P_p is the `top' binary, for which level=0 
                while (parent != None): ### search parents until reaching the top binary 
                    for index_particle_2,particle_2 in enumerate(particles):
                        
                        if parent == None: break
                        if (particle_2.index_temp == parent.index_temp):
                            particle_1.level += 1
                            
                            parent = particle_2.parent
                     
    @staticmethod
    def evolve_system(configuration,N_bodies,masses,metallicities,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,tend,N_steps,stellar_types=None,make_plots=True,fancy_plots=False,plot_filename="test1",show_plots=True,object_types=None,random_seed=0,verbose_flag=0,include_WD_kicks=False,kick_distribution_sigma_km_s_WD=1.0,NS_model=0,ECSNe_model=0,kick_distribution_sigma_km_s_NS=265.0,kick_distribution_sigma_km_s_BH=50.0,flybys_stellar_density_per_cubic_pc=0.1,flybys_encounter_sphere_radius_au=1.0e5,flybys_stellar_relative_velocity_dispersion_km_s=30.0,flybys_include_secular_encounters=False,include_flybys=True,save_data=False,plot_only=False,wall_time_max_s=3.6e4,common_envelope_timescale=1.0e2,binary_evolution_SNe_Ia_single_degenerate_model=0,binary_evolution_SNe_Ia_double_degenerate_model=0,effective_radius_multiplication_factor_for_collisions_compact_objects=100.0,effective_radius_multiplication_factor_for_collisions_stars=1.0,tides_viscous_time_scale_prescription=1,dynamical_instability_criterion=0):  # [C49] mutable defaults

        if stellar_types is None: stellar_types = []
        if object_types is None: object_types = []

        np.random.seed(random_seed)
        
        if plot_only==True: ### load previously-generated data from disk; do not run the system
            import pickle
            try:
                with open(plot_filename + ".pkl",'rb') as file:
                    data = pickle.load(file)
            except IOError:
                print("Could not load pickle file ",plot_filename + ".pkl" + "; please make sure you run the system with the same filename first and with the --save_data argument enabled.")
                exit(0)

            log_copy = data["log"]
            error_code_copy = data.get("error_code", 0)  # [C6] ensure error_code_copy is defined in plot_only path

            N_orbits_status = data['N_orbits_status']
            N_bodies_status = data['N_bodies_status']
            N_status = data['N_status']
            t_print = data['t_print']
            m_print = data['m_print']
            mc_print = data['mc_print']
            k_print = data['k_print']
            R_print = data['R_print']
            Rc_print = data['Rc_print']
            X_print = data['X_print']
            Y_print = data['Y_print']
            Z_print = data['Z_print']
            a_print = data['a_print']
            e_print = data['e_print']
            T_eff_print = data['T_eff_print']
            L_print = data['L_print']
            rel_INCL_print = data['rel_INCL_print']
            spin_frequency_print = data['spin_frequency_print']
            
        else: ### run the system

            ### set up the system ###
            if configuration == "fully_nested":
                particles = Tools.create_fully_nested_multiple(N_bodies, masses,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,metallicities=metallicities,stellar_types=stellar_types,object_types=object_types)
            elif configuration == "2+2_quadruple":
                particles = Tools.create_2p2_quadruple_system(masses,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,metallicities=metallicities,stellar_types=stellar_types,object_types=object_types)
            else:
                particles = Tools.create_hierarchy(N_bodies,configuration,masses,semimajor_axes,eccentricities,inclinations,arguments_of_pericentre,longitudes_of_ascending_node,radii=None,metallicities=metallicities,stellar_types=stellar_types,object_types=object_types)

            print("="*50)
            print("mse.py -- evolve_system() -- running system with parameters:")
            print("Configuration: ",configuration)
            print("N_bodies: ",N_bodies)
            print("Object types:",object_types)
            print("Stellar types:",stellar_types)
            print("Masses/MSun: ",masses)
            print("Metallicities: ",metallicities)
            print("Semimajor axes (au): ",semimajor_axes)
            print("Eccentricities: ",eccentricities)
            print("Inclinations (rad): ",inclinations)
            print("Longitudes of the ascending node (rad): ",longitudes_of_ascending_node)
            print("Arguments of periapsis (rad): ",arguments_of_pericentre)  # [C51] was printing inclinations
            print("Integration time (yr): ",tend)
            print("Number of plot output steps: ",N_steps)

            print("="*50)
            print("Starting evolution")
            
            orbits = [x for x in particles if x.is_binary==True]
            bodies = [x for x in particles if x.is_binary==False]

            for b in bodies:
                b.include_WD_kicks = include_WD_kicks
                b.kick_distribution_sigma_km_s_WD = kick_distribution_sigma_km_s_WD
                b.kick_distribution_sigma_km_s_NS = kick_distribution_sigma_km_s_NS
                b.kick_distribution_sigma_km_s_BH = kick_distribution_sigma_km_s_BH
                b.common_envelope_timescale = common_envelope_timescale
                b.tides_viscous_time_scale_prescription = tides_viscous_time_scale_prescription

            if dynamical_instability_criterion != 0:
                for o in orbits:
                    o.dynamical_instability_criterion = dynamical_instability_criterion

            N_bodies = len(bodies)
            N_orbits = len(orbits)

            ### set up the code ###
            from mse import MSE
            
            code = MSE()
            code.add_particles(particles)

            code.random_seed = random_seed
            code.verbose_flag = verbose_flag
            code.wall_time_max_s = wall_time_max_s

            code.include_flybys = include_flybys
            code.flybys_include_secular_encounters = flybys_include_secular_encounters
            code.flybys_stellar_density = flybys_stellar_density_per_cubic_pc*code.CONST_PER_PC3
            code.flybys_encounter_sphere_radius = flybys_encounter_sphere_radius_au
            code.flybys_stellar_relative_velocity_dispersion = flybys_stellar_relative_velocity_dispersion_km_s * code.CONST_KM_PER_S

            code.binary_evolution_use_eCAML_model = False
            code.NS_model = NS_model
            code.ECSNe_model = ECSNe_model
            code.binary_evolution_SNe_Ia_single_degenerate_model = binary_evolution_SNe_Ia_single_degenerate_model
            code.binary_evolution_SNe_Ia_double_degenerate_model = binary_evolution_SNe_Ia_double_degenerate_model
            code.effective_radius_multiplication_factor_for_collisions_stars = effective_radius_multiplication_factor_for_collisions_stars
            code.effective_radius_multiplication_factor_for_collisions_compact_objects = effective_radius_multiplication_factor_for_collisions_compact_objects

            ### set up custom output printing arrays ###
            t_print = [[]]
            internal_indices_print = [[[] for x in range(N_bodies)]]
            k_print = [[[] for x in range(N_bodies)]]
            m_print = [[[] for x in range(N_bodies)]]
            L_print = [[[] for x in range(N_bodies)]]
            T_eff_print = [[[] for x in range(N_bodies)]]
            R_print = [[[] for x in range(N_bodies)]]
            X_print = [[[] for x in range(N_bodies)]]
            Y_print = [[[] for x in range(N_bodies)]]
            Z_print = [[[] for x in range(N_bodies)]]
            Rc_print = [[[] for x in range(N_bodies)]]
            R_L_print = [[[] for x in range(N_bodies)]]
            t_V_print = [[[] for x in range(N_bodies)]]
            mc_print = [[[] for x in range(N_bodies)]]
            a_print = [[[] for x in range(N_orbits)]]
            e_print = [[[] for x in range(N_orbits)]]
            rel_INCL_print = [[[] for x in range(N_orbits)]]
            spin_frequency_print = [[[] for x in range(N_bodies)]]
            
            N_orbits_status = [N_orbits]
            N_bodies_status = [N_bodies]
            
            ### start time loop within Python ###
            t = 0.0
            integration_flags = [[]]

            i_status = 0

            dt = tend/float(N_steps)
            i = 0
           
            Python_wall_start = time.time()

            error_code = 0
            log_copy_for_save = []
            log_snapshot = []

            error_code_descriptions = {
                -35: "Python-side wall time exceeded",
                -1: "segmentation fault",
                0: "no error",
                1: "tools.cpp -- check_number() (NaN or INF)",
                2: "binary_evolution.cpp -- handle_wind_accretion()",
                3: "binary_evolution.cpp -- handle_mass_transfer_cases()",
                4: "binary_evolution.cpp -- dynamical_mass_transfer_WD_donor()",
                5: "binary_evolution.cpp -- triple_stable_mass_transfer_evolution()",
                6: "collision.cpp -- collision_product()",
                7: "collision.cpp -- handle_collisions()",
                8: "common_envelope_evolution.cpp -- triple_common_envelope_evolution()",
                9: "nbody_evolution.cpp -- handle_collisions_nbody()",
                10: "ODE_mass_changes.cpp -- ODE_handle_RLOF_triple_mass_transfer()",
                11: "ODE_mass_changes.cpp -- ODE_handle_RLOF_triple_mass_transfer()",
                12: "ODE_mass_changes.cpp -- compute_RLOF_emt_model()",
                13: "SNe.cpp -- sample_kick_velocity()",
                14: "flybys.cpp -- sample_next_flyby()",
                15: "flybys.cpp -- sample_flyby_position_and_velocity_at_R_enc()",
                16: "flybys.cpp -- sample_flyby_mass_at_infinity()",
                17: "stellar_evolution.cpp -- initialize_stars()",
                18: "structure.cpp -- check_system_for_dynamical_stability()",
                19: "ODE_newtonian.cpp -- compute_EOM_binary_pairs()",
                20: "ODE_newtonian.cpp -- compute_EOM_binary_pairs_single_averaged()",
                21: "ODE_newtonian.cpp -- compute_EOM_binary_triplets()",
                22: "ODE_tides.cpp -- compute_EOM_equilibrium_tide_BO_full()",
                23: "ODE_root_finding.cpp -- roche_radius_pericenter_sepinsky()",
                24: "ODE_root_finding.cpp -- handle_roots()",
                25: "apsidal_motion_constant.cpp -- compute_apsidal_motion_constant()",
                26: "external.cpp -- compute_EOM_binary_pairs_external_perturbation()",
                27: "mst.c -- die()",
                28: "mst.c -- initialize_mpi_or_serial()",
                29: "mst.c -- check_relative_proximity()",
                30: "mst.c -- compute_U()",
                31: "mst.c -- stopping_condition_function()",
                32: "mst.c -- check_for_initial_stopping_condition()",
                33: "ODE_VRR.cpp -- compute_VRR_perturbations()",
                34: "tools.cpp -- sample_from_Kroupa_93_imf()",
                35: "ODE_system.cpp -- wall time exceeded",
                36: "mst.c -- wall time exceeded",
                37: "stellar_evolution.cpp -- determine_sse_compact_object_radius_RSun()",
                38: "stellar_evolution.cpp -- compute_moment_of_inertia()",
                39: "common_envelope_evolution.cpp -- binary_common_envelope_evolution() -- zero core mass",
                40: "SSE evolv1.f -- radius convergence error",
                41: "SSE evolv1.f -- timestep convergence error",
                42: "binary_evolution.cpp -- white_dwarf_helium_mass_accumulation_efficiency()",
                43: "binary_evolution.cpp -- determine_if_He_accreting_WD_explodes()",
            }

            while t<tend:

                t+=dt
                code.evolve_model(t)

                ### Snapshot log immediately after evolve_model returns ###
                ### (before error checking, since wall-time break skips the end-of-loop snapshot) ###
                try:
                    if len(code.log) > len(log_snapshot):
                        log_snapshot = copy.deepcopy(code.log)
                except Exception as e:
                    print("WARNING -- mse.py -- failed to snapshot log after evolve_model: {}".format(e))

                ### check for errors/wall time ###
                error_code = code.error_code
                if error_code not in [0,35,36]:
                    error_desc = error_code_descriptions.get(error_code, "unknown error")
                    print("="*50)
                    print("WARNING -- mse.py -- Internal error with code ",error_code,"occurred -- stopping the simulation but saving data/making plots if specified in command line arguments.")
                    print("  Error description: ", error_desc)
                    print("  Time of error: t = {:.6g} Myr".format(t*1.0e-6))
                    print("  Initial masses (MSun): ", masses)
                    print("  Initial semimajor axes (AU): ", semimajor_axes)
                    print("  Initial eccentricities: ", eccentricities)
                    print("="*50)
                    break

                wall_time_s = time.time() - Python_wall_start
                if wall_time_s > wall_time_max_s or error_code in [35,36]:
                    if error_code not in [35,36]: ### Wall time was exceeded in Python layer (otherwise, within MSE)
                        error_code = -35
                    print("="*50)
                    print("WARNING -- mse.py -- maximum wall time of ",wall_time_max_s," s exceeded -- stopping the simulation but saving data/making plots if specified in command line arguments.")
                    print("Python wall_time_s",wall_time_s,"error_code",error_code)
                    print("="*50)
                    break

                CVODE_flag = code.CVODE_flag
                state = code.state

                print( 't/Myr',t*1e-6,'masses/MSun',[b.mass for b in bodies],'smas/au',[o.a for o in orbits],'es',[o.e for o in orbits],'integration_flag',code.integration_flag)
                
                ### Custom output printing ###

                particles = code.particles
                orbits = [x for x in particles if x.is_binary==True]
                bodies = [x for x in particles if x.is_binary==False]
                N_orbits = len(orbits)
                N_bodies = len(bodies)
                   
                if code.structure_change == True:
                    #print("Python restruct")#,children1,children1_old,children2,children2_old)
                    t_print.append([])
                    integration_flags.append([])
                    internal_indices_print.append([[] for x in range(N_bodies)])
                    k_print.append([[] for x in range(N_bodies)])
                    m_print.append([[] for x in range(N_bodies)])
                    L_print.append([[] for x in range(N_bodies)])
                    T_eff_print.append([[] for x in range(N_bodies)])
                    R_print.append([[] for x in range(N_bodies)])
                    X_print.append([[] for x in range(N_bodies)])
                    Y_print.append([[] for x in range(N_bodies)])
                    Z_print.append([[] for x in range(N_bodies)])
                    Rc_print.append([[] for x in range(N_bodies)])
                    R_L_print.append([[] for x in range(N_bodies)])
                    t_V_print.append([[] for x in range(N_bodies)])
                    mc_print.append([[] for x in range(N_bodies)])
                    a_print.append([[] for x in range(N_orbits)])
                    e_print.append([[] for x in range(N_orbits)])
                    rel_INCL_print.append([[] for x in range(N_orbits)])
                    spin_frequency_print.append([[] for x in range(N_bodies)])
                    
                    N_orbits_status.append(N_orbits)
                    N_bodies_status.append(N_bodies)
                    
                    i_status += 1
                    
                for index in range(N_orbits):
                    rel_INCL_print[i_status][index].append(orbits[index].INCL_parent)
                    e_print[i_status][index].append(orbits[index].e)
                    a_print[i_status][index].append(orbits[index].a)
                for index in range(N_bodies):
                    internal_indices_print[i_status][index].append(bodies[index].index)
                    m_print[i_status][index].append(bodies[index].mass)
                    L_print[i_status][index].append(bodies[index].luminosity)
                    k_print[i_status][index].append(bodies[index].stellar_type)
                    R_print[i_status][index].append(bodies[index].radius)
                    X_print[i_status][index].append(bodies[index].X)
                    Y_print[i_status][index].append(bodies[index].Y)
                    Z_print[i_status][index].append(bodies[index].Z)
                    t_V_print[i_status][index].append(bodies[index].tides_viscous_time_scale)
                    Rc_print[i_status][index].append(bodies[index].convective_envelope_radius)
                    R_L_print[i_status][index].append(bodies[index].roche_lobe_radius_pericenter)
                    mc_print[i_status][index].append(bodies[index].convective_envelope_mass)
                    T_eff = Tools.compute_effective_temperature(bodies[index].luminosity, bodies[index].radius, code.CONST_L_SUN, code.CONST_R_SUN)
                    T_eff_print[i_status][index].append(T_eff)
                    spin_frequency_print[i_status][index].append( np.sqrt( bodies[index].spin_vec_x**2 + bodies[index].spin_vec_y**2 + bodies[index].spin_vec_z**2) )

                t_print[i_status].append(t)
                integration_flags[i_status].append(code.integration_flag)

                i += 1

            N_status = i_status+1

            for i_status in range(N_status):
                t_print[i_status] = np.array(t_print[i_status])

            try:
                print("Final properties -- ","masses/MSun",[m_print[-1][i][-1] for i in range(N_bodies)],"smas/au",[a_print[-1][i][-1] for i in range(N_orbits)],"es",[e_print[-1][i][-1] for i in range(N_orbits)])
            except (IndexError, TypeError):
                print("Final properties -- unable to print (data arrays may be empty)")

            # Use the incrementally-captured log snapshot (safe Python-side copy)
            # This avoids calling into C++ which may be corrupted after wall-time longjmp
            log_copy = log_snapshot
            log_copy_for_save = log_snapshot
            error_code_copy = error_code

            if save_data==True: ### save code log and custom output data to disk for later analysis/replotting
                ### NOTE: pkl is saved BEFORE write_final_log_entry() so that data is persisted even if
                ### C++ state is corrupted (e.g. after wall-time longjmp). Trade-off: the pkl will not
                ### contain the final log entry.

                wall_time_s = time.time() - Python_wall_start
                data = {"log":log_copy_for_save,'error_code':error_code_copy,"wall_time_s":wall_time_s}

                data['N_orbits_status'] = N_orbits_status
                data['N_bodies_status'] = N_bodies_status
                data['N_status'] = N_status
                data['t_print'] = t_print
                data['m_print'] = m_print
                data['mc_print'] = mc_print
                data['k_print'] = k_print
                data['R_print'] = R_print
                data['Rc_print'] = Rc_print
                data['X_print'] = X_print
                data['Y_print'] = Y_print
                data['Z_print'] = Z_print
                data['a_print'] = a_print
                data['e_print'] = e_print
                data['T_eff_print'] = T_eff_print
                data['L_print'] = L_print
                data['rel_INCL_print'] = rel_INCL_print
                data['spin_frequency_print'] = spin_frequency_print

                import pickle
                print("Saving output data to ",plot_filename + ".pkl")
                try:
                    with open(plot_filename + ".pkl",'wb') as file:
                        pickle.dump(data,file)
                except IOError:
                    print("Error saving output data to ",plot_filename + ".pkl; make sure the path exists and/or enough disk space is available.")

            try:
                code.write_final_log_entry() ### This has to be done within Python, since the C++ code does not know if the desired Python simulation end time has been reached!
            except Exception:
                print("WARNING -- mse.py -- write_final_log_entry() failed (C++ state may be corrupted after wall time exceeded)")

            if verbose_flag > 0:
                try:
                    print("Number of log entries:",len(code.log))
                    print("log",code.log)
                except Exception:
                    print("WARNING -- mse.py -- could not print log (C++ state may be corrupted)")

            code.reset()
            
        if make_plots==True:
            print("Making plots...")

            try:
                from matplotlib import pyplot
                from matplotlib import lines
                import matplotlib
            except ImportError:
                print("evolve_system.py -- ERROR: cannot import Matplotlib")
                exit(-1)

            ### Font settings for mathtext ###
            matplotlib.rcParams['mathtext.fontset'] = 'cm'
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern Roman']

            if fancy_plots == True:
                print("Using LaTeX for plot text")
                pyplot.rc('text', usetex=True)
                pyplot.rc('legend', fancybox=True)

            parsec_in_AU = 206201.0
            CONST_L_SUN = 0.0002710404109745588

            plot_log = []
            previous_event_flag = -1
            for index_log, log in enumerate(log_copy):
                event_flag = log["event_flag"]
                if previous_event_flag == event_flag and (event_flag == 4 or event_flag == 10):

                    continue
                plot_log.append(log)
                previous_event_flag = event_flag

            N_l = len(plot_log)
            N_c = min(4, N_l)
            N_r = int(np.ceil(N_l / max(N_c, 1)))
            panel_w = 3.5
            panel_h = 7.0

            fontsize = 20
            fig = pyplot.figure(figsize=(N_c * panel_w, N_r * panel_h))

            ### Build legend with updated markers and Roche lobe entries ###
            legend_elements = []
            for k in range(16):
                color, s, description, marker = Tools.get_color_and_size_and_description_for_star(k, 1.0, mass=1.0)
                if description == '':
                    continue
                legend_elements.append(lines.Line2D([0], [0], marker='o', markerfacecolor=color,
                                       color='w', markersize=10,
                                       label=r"$\mathrm{%s}$" % description.replace(' ', r'\,')))
            ### Roche filling legend entries ###
            for rl_color, rl_label in [('green', r'$R/R_L < 0.5$'), ('gold', r'$0.5 \leq R/R_L < 0.8$'), ('red', r'$R/R_L \geq 0.8$')]:
                legend_elements.append(lines.Line2D([0], [0], marker='o', markerfacecolor='none',
                                       markeredgecolor=rl_color, markeredgewidth=1.5,
                                       color='w', markersize=10, label=rl_label))

            for index_log, log in enumerate(plot_log):
                plot = fig.add_subplot(N_r, N_c, index_log + 1)
                plot.set_facecolor('white')
                particles = log["particles"]
                event_flag = log["event_flag"]
                index1 = log["index1"]
                index2 = log["index2"]

                Tools.generate_mobile_diagram(particles, plot, fontsize=fontsize, index1=index1, index2=index2, event_flag=event_flag)

                ### Event name at top, timestamp at bottom ###
                event_text = Tools.get_description_for_event_flag(event_flag, log["SNe_type"])
                time_text = r"$t \simeq %s\,\mathrm{Myr}$" % round(log["time"] * 1e-6, 2)
                plot.text(0.5, 1.01, event_text,
                          transform=plot.transAxes, ha='center', va='bottom',
                          fontsize=14, zorder=15)
                plot.text(0.5, 0.01, time_text,
                          transform=plot.transAxes, ha='center', va='bottom',
                          fontsize=14, zorder=15)

                if index_log == 0:
                    plot.legend(handles=legend_elements, bbox_to_anchor=(-0.05, 1.45), loc='upper left', ncol=5, fontsize=12, handletextpad=0.3, columnspacing=1.0)

            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05, hspace=0.4)
            fig.savefig(plot_filename + "_mobile.pdf")
        
            fig=pyplot.figure(figsize=(8,10))
            Np=4
            plot1=fig.add_subplot(Np,1,1)
            plot2=fig.add_subplot(Np,1,2,yscale="log")
            plot3=fig.add_subplot(Np,1,3,yscale="linear")
            plot4=fig.add_subplot(Np,1,4,yscale="log")
            
            fig_pos=pyplot.figure(figsize=(8,8))
            plot_pos=fig_pos.add_subplot(1,1,1)

            fig_HRD=pyplot.figure(figsize=(8,8))
            plot_HRD=fig_HRD.add_subplot(1,1,1)
            
            colors = ['k','tab:red','tab:green','tab:blue','y','k','tab:red','tab:green','tab:blue','y']
            linewidth=1.0
            for i_status in range(N_status):
                N_bodies = N_bodies_status[i_status]
                N_orbits = N_orbits_status[i_status]
                
                #if i_status==0:
                #    plot1.plot(1.0e-6*t_print[i_status],np.array(m_print[i_status][0])**2*np.array(m_print[i_status][1])**2*np.array(a_print[i_status][0]),color='y',linewidth=3)
                #    plot1.plot(1.0e-6*t_print[i_status],(np.array(m_print[i_status][0])+np.array(m_print[i_status][1]))*np.array(a_print[i_status][0]),color='y',linewidth=3)
                for index in range(N_bodies):
                    color=colors[index]
                    plot1.plot(1.0e-6*t_print[i_status],m_print[i_status][index],color=color,linewidth=linewidth)
                    plot1.plot(1.0e-6*t_print[i_status],mc_print[i_status][index],color=color,linestyle='dotted',linewidth=linewidth)
                    plot3.plot(1.0e-6*t_print[i_status],k_print[i_status][index],color=color,linestyle='solid',linewidth=linewidth)
                    plot2.plot(1.0e-6*t_print[i_status],R_print[i_status][index],color=color,linestyle='solid',linewidth=linewidth)
                    plot2.plot(1.0e-6*t_print[i_status],Rc_print[i_status][index],color=color,linestyle='dotted',linewidth=linewidth)
                    
                    plot_pos.plot(np.array(X_print[i_status][index])/parsec_in_AU,np.array(Y_print[i_status][index])/parsec_in_AU,color=color,linestyle='solid',linewidth=linewidth)
                    
                    plot_HRD.scatter(np.log10(np.array(T_eff_print[i_status][index])), np.log10(np.array(L_print[i_status][index])/CONST_L_SUN),color=color,linestyle='solid',linewidth=linewidth,s=10)
                    #plot4.plot(1.0e-6*t_print[i_status],spin_frequency_print[i_status][index],color=color,linewidth=linewidth) 
                    plot4.plot(1.0e-6*t_print[i_status],3.15576e7*2.0*np.pi/np.array(spin_frequency_print[i_status][index]),color=color,linewidth=linewidth) 
                   
                linewidth=1.0
                for index in range(N_orbits):
                    color = colors[index]
                    smas = np.array(a_print[i_status][index])
                    es = np.array(e_print[i_status][index])
                    plot2.plot(1.0e-6*t_print[i_status],smas,color=color,linestyle='dotted',linewidth=linewidth)
                    plot2.plot(1.0e-6*t_print[i_status],smas*(1.0-es),color=color,linestyle='solid',linewidth=linewidth)
                
                linewidth+=0.4

            fontsize=18
            labelsize=12

            plots = [plot1,plot2,plot3,plot4]
            for plot in plots:
                plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

            log_CEs = [x for x in plot_log if x["event_flag"] == 6]
            t_CEs_Myr = np.array([x["time"]*1e-6 for x in log_CEs])
            
            for k,t in enumerate(t_CEs_Myr):
                plot2.axvline(x=t,linestyle='dotted',color='tab:red',linewidth=0.5)
                plot2.annotate(r"$\mathrm{CE}$",xy=(1.02*t,1.0e3),fontsize=0.8*fontsize)

            plot1.set_ylabel(r"$m/\mathrm{M}_\odot$",fontsize=fontsize)
            plot2.set_ylabel(r"$r/\mathrm{au}$",fontsize=fontsize)
            plot3.set_ylabel(r"$\mathrm{Stellar\,Type}$",fontsize=fontsize)
            #plot4.set_ylabel(r"$\Omega_\mathrm{spin}/\mathrm{yr^{-1}}$",fontsize=fontsize)
            plot4.set_ylabel(r"$P_\mathrm{spin}/\mathrm{s}$",fontsize=fontsize)
            plot4.set_xlabel(r"$t/\mathrm{Myr}$",fontsize=fontsize)
            plot2.set_ylim(1.0e-5,1.0e5)
            
            plot_pos.set_xlabel(r"$X/\mathrm{pc}$",fontsize=fontsize)
            plot_pos.set_ylabel(r"$Y/\mathrm{pc}$",fontsize=fontsize)
            plot_pos.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            

            plot_HRD.set_xlim(5.0,3.0)
            plot_HRD.set_ylim(-4.0,6.0)
            plot_HRD.set_xlabel(r"$\mathrm{log}_{10}(T_\mathrm{eff}/\mathrm{K})$",fontsize=fontsize)
            plot_HRD.set_ylabel(r"$\mathrm{log}_{10}(L/L_\odot)$",fontsize=fontsize)
            plot_HRD.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            
            fig.savefig(plot_filename + ".pdf")
            fig_pos.savefig(plot_filename + "_pos.pdf")
            fig_HRD.savefig(plot_filename + "_HRD.pdf")
            
            print("Plots generated and written to disk.") 
            
            if show_plots == True:
                pyplot.show()
    
      
        return error_code_copy, log_copy

       
    @staticmethod
    def generate_mobile_diagram(particles, plot, line_width_horizontal=1.5, line_width_vertical=1.2, line_color='k', line_width=3.0, fontsize=20, use_default_colors=True, index1=-1, index2=-1, event_flag=-1):
        r"""
        Generate a Mobile diagram of a given multiple system.
        """

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            print("mse.py -- generate_mobile_diagram -- unable to import Matplotlib which is needed to generate a Mobile diagram!")
            exit(0)

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        if len(binaries) == 0:
            if len(bodies) == 0:
                print("mse.py -- generate_mobile_diagram -- zero bodies and zero binaries!")
                return
            else:
                body_spacing = 4.0
                Tools.draw_bodies(plot, bodies, fontsize, index1=index1, index2=index2, event_flag=event_flag)
                ### Set axis limits for unbound-only panels ###
                n_b = len(bodies)
                x_pad = max(body_spacing * 0.6, 2.0)
                plot.set_xlim([-x_pad, (n_b - 1) * body_spacing + x_pad])
                plot.set_ylim([-1.5, 2.5])
                plot.set_xticks([])
                plot.set_yticks([])
                for spine in plot.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                    spine.set_edgecolor('grey')
                return

        Tools.determine_binary_levels_in_particles(particles)
        unbound_bodies = [x for x in particles if x.is_binary == False and x.parent is None]
        if len(unbound_bodies) > 0:
            Tools.draw_bodies(plot, unbound_bodies, fontsize, y_ref=1.5 * line_width_vertical, dx=0.8 * line_width_horizontal, dy=0.6 * line_width_vertical, index1=index1, index2=index2, event_flag=event_flag)

        top_level_binaries = [x for x in binaries if x.level == 0]

        ### Assign bone colormap colours to orbits (middle third) ###
        orbit_cmap = cm.get_cmap('bone')
        n_binaries = len(binaries)
        binaries_sorted = sorted(binaries, key=lambda b: b.level)
        for idx, b in enumerate(binaries_sorted):
            if n_binaries == 1:
                t = 0.5
            else:
                t = 0.33 + 0.34 * idx / (n_binaries - 1)
            b.color = orbit_cmap(t)

        ### Collect orbit info for top boxes ###
        orbit_info_list = []

        ### Make mobile diagram ###
        top_x = 0.0
        top_y = 0.0  # Tree starts at origin, grows downward

        if len(top_level_binaries) > 1:
            top_x = -3 * line_width_horizontal
            top_y = 0.0

        for index, top_level_binary in enumerate(top_level_binaries):

            top_level_binary.x = top_x - 5 * index * line_width_horizontal
            top_level_binary.y = top_y

            x_min = x_max = y_min = 0.0
            y_max = line_width_vertical

            ### No top stem — start directly at the binary node ###
            x_min, x_max, y_min, y_max = Tools.draw_binary_node(plot, top_level_binary, line_width_horizontal, line_width_vertical, line_color, line_width, fontsize, x_min, x_max, y_min, y_max, index1=index1, index2=index2, event_flag=event_flag, orbit_info_list=orbit_info_list)

        plot.set_xticks([])
        plot.set_yticks([])

        ### Extend limits to include unbound bodies if any ###
        if len(unbound_bodies) > 0:
            body_spacing = 4.0
            ub_y = 1.5 * line_width_vertical
            for ub_idx, ub in enumerate(unbound_bodies):
                ub_x = ub_idx * body_spacing
                if ub_x < x_min: x_min = ub_x
                if ub_x > x_max: x_max = ub_x
                if ub_y > y_max: y_max = ub_y

        ### Padding: labels extend ~100pt below lowest star via offset points ###
        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)
        x_pad = max(0.25 * x_range, 1.0)
        y_pad_bottom = max(0.7 * y_range, 3.0)
        n_orbits = len(orbit_info_list)
        if n_orbits >= 3:
            y_pad_top = max(0.6 * y_range, 3.5)
        elif n_orbits == 2:
            y_pad_top = max(0.3 * y_range, 2.0)
        elif n_orbits == 1:
            y_pad_top = max(0.2 * y_range, 1.2)
        else:
            y_pad_top = max(0.1 * y_range, 0.3)
        plot.set_xlim([x_min - x_pad, x_max + x_pad])
        plot.set_ylim([y_min - y_pad_bottom, y_max + y_pad_top])

        for spine in plot.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('grey')

        ### Draw orbit parameter boxes ###
        ### Always at the top of the panel ###
        has_unbound = len(unbound_bodies) > 0
        box_y = 0.97
        box_va = 'top'
        n_orbits = len(orbit_info_list)
        if n_orbits > 0:
            for i, info in enumerate(orbit_info_list):
                a_val = info['a']
                e_val = info['e']
                orbit_color = info['color']
                orbit_idx = info['orbit_index']

                N_ra = 1 if a_val > 0.1 else 2
                lines_text = r"$a_{%d} = %s\,\mathrm{au}$" % (orbit_idx, round(a_val, N_ra))
                lines_text += "\n" + r"$e_{%d} = %.2f$" % (orbit_idx, e_val)
                if 'i_rel' in info and info['i_rel'] is not None:
                    lines_text += "\n" + r"$i_{%d} = %.0f^\circ$" % (orbit_idx, info['i_rel'])

                ### All boxes in one row, evenly spaced ###
                box_x_i = (i + 0.5) / n_orbits
                box_ha = 'center'
                if n_orbits >= 2:
                    if i == 0:
                        box_x_i = 0.02
                        box_ha = 'left'
                    elif i == n_orbits - 1:
                        box_x_i = 0.98
                        box_ha = 'right'

                bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=orbit_color, lw=2.0, alpha=0.85)
                plot.text(box_x_i, box_y, lines_text, transform=plot.transAxes,
                          ha=box_ha, va=box_va, fontsize=12,
                          bbox=bbox_props, zorder=15)
        
        #plot.autoscale(enable=True,axis='both')
        
    @staticmethod
    def _total_mass(particle):
        r"""Recursively compute total mass of a particle (body or binary subtree)."""
        if not particle.is_binary:
            return particle.mass
        return Tools._total_mass(particle.child1) + Tools._total_mass(particle.child2)

    @staticmethod
    def draw_binary_node(plot, particle, line_width_horizontal, line_width_vertical, line_color, line_width, fontsize, x_min, x_max, y_min, y_max, index1=-1, index2=-1, event_flag=-1, i_rel=None, orbit_info_list=None, orbit_counter=None):
        CONST_MJUP = 0.000954248
        CONST_R_SUN = 0.004649130343817401
        CONST_KM_PER_AU = 1.4966e8

        if orbit_info_list is None:
            orbit_info_list = []
        if orbit_counter is None:
            orbit_counter = [0]

        x = particle.x
        y = particle.y

        child1 = particle.child1
        child2 = particle.child2

        from matplotlib.patches import Ellipse

        ### Orbit info collected AFTER children (so inner orbits get lower indices) ###
        ### See below, after child handling ###

        ### Proportional arm length based on semi-major axis ###
        ### Minimum alpha ensures sibling labels don't overlap horizontally ###
        both_leaves = (not child1.is_binary) and (not child2.is_binary)
        alpha = max(1.5, 0.8 + 0.3 * np.log10(max(particle.a, 0.01)))
        if child1.is_binary == True and child2.is_binary == True:
            alpha = max(alpha, 5.0)
        elif child1.is_binary == True or child2.is_binary == True:
            alpha = max(alpha, 3.0)  # wider so subtree labels don't overlap sibling body
        elif both_leaves:
            alpha = max(alpha, 2.2)  # room for side-by-side text blocks

        orbit_color = particle.color

        ### Node point at the junction ###
        plot.scatter([x], [y], color=orbit_color, s=60, zorder=12, edgecolors='none')

        ### lines to child1 ###
        plot.plot([x, x - alpha * line_width_horizontal], [y, y], color=orbit_color, linewidth=line_width)
        plot.plot([x - alpha * line_width_horizontal, x - alpha * line_width_horizontal], [y, y - line_width_vertical], color=orbit_color, linewidth=line_width)

        ### lines to child2 ###
        plot.plot([x, x + alpha * line_width_horizontal], [y, y], color=orbit_color, linewidth=line_width)
        plot.plot([x + alpha * line_width_horizontal, x + alpha * line_width_horizontal], [y, y - line_width_vertical], color=orbit_color, linewidth=line_width)

        ### positions of children ###
        child1.x = particle.x - alpha * line_width_horizontal
        child2.x = particle.x + alpha * line_width_horizontal

        child1.y = particle.y - line_width_vertical
        child2.y = particle.y - line_width_vertical

        if (child1.x < x_min): x_min = child1.x
        if (child1.x > x_max): x_max = child1.x
        if (child2.x < x_min): x_min = child2.x
        if (child2.x > x_max): x_max = child2.x

        if (child1.y < y_min): y_min = child1.y
        if (child1.y > y_max): y_max = child1.y
        if (child2.y < y_min): y_min = child2.y
        if (child2.y > y_max): y_max = child2.y

        ### helper to draw a body (star) node ###
        def _draw_body(child, other_child, x_label_offset_sign):
            color, s, description, marker = Tools.get_color_and_size_and_description_for_star(child.stellar_type, child.radius, mass=child.mass)

            ### Main star marker ###
            plot.scatter([child.x], [child.y], c=[color], s=s, zorder=10,
                         edgecolors='black', linewidth=2.0, alpha=0.8, marker=marker)

            ### Roche lobe filling ring ###
            try:
                ### Eggleton (1983) formula for Roche lobe radius ###
                m_child = child.mass
                m_other = other_child.mass if not other_child.is_binary else Tools._total_mass(other_child)
                if m_other > 0 and particle.a > 0:
                    q = m_child / m_other
                    q13 = q ** (1.0 / 3.0)
                    q23 = q13 * q13
                    rl_over_a = 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q13))
                    rp = particle.a * (1.0 - particle.e)
                    rl = rl_over_a * rp
                    r_star_au = child.radius
                    fill_frac = r_star_au / rl if rl > 0 else 0.0

                    if fill_frac < 0.5:
                        ring_color = 'green'
                    elif fill_frac < 0.8:
                        ring_color = 'gold'
                    else:
                        ring_color = 'red'

                    s_ring = min(s * 2.0, 1200)
                    plot.scatter([child.x], [child.y], s=s_ring, facecolors='none',
                                edgecolors=ring_color, linewidth=1.5, zorder=9, marker='o')
            except (AttributeError, ZeroDivisionError):
                pass

            ### Star label: single text block below star ###
            label_x = child.x
            label_y = child.y

            type_text = description
            if getattr(child, 'object_type', 1) == 2:
                mass_text = r"$%.1f\,M_\mathrm{J}$" % (child.mass / CONST_MJUP)
            else:
                mass_text = r"$%.2f\,M_\odot$" % child.mass
            r_rsun = child.radius / CONST_R_SUN
            if r_rsun < 0.001:
                r_km = child.radius * CONST_KM_PER_AU
                radius_text = r"$%.0f$ km" % r_km
            else:
                radius_text = r"$%.2f\,R_\odot$" % r_rsun

            label_block = type_text + "\n" + mass_text + "\n" + radius_text
            plot.annotate(label_block, xy=(label_x, label_y), xytext=(0, -24),
                          textcoords='offset points', ha='center', va='top',
                          color='k', fontsize=12, zorder=10, linespacing=1.4)

        ### handle children ###
        if child1.is_binary == True:

            if event_flag in [14]:  ### Triple CE
                ell = Ellipse(xy=[x, child1.y], width=2.0 * (child2.x - child1.x), height=3.5 * (y - child1.y), angle=0.0, color='tab:red', alpha=0.5)
                plot.add_artist(ell)

            i_rel_child = Tools.compute_mutual_inclination(particle.INCL, child1.INCL, particle.LAN, child1.LAN) * (180.0 / np.pi)
            x_min, x_max, y_min, y_max = Tools.draw_binary_node(plot, child1, line_width_horizontal, line_width_vertical, line_color, line_width, fontsize, x_min, x_max, y_min, y_max, index1=index1, index2=index2, event_flag=event_flag, i_rel=i_rel_child, orbit_info_list=orbit_info_list, orbit_counter=orbit_counter)
        else:
            _draw_body(child1, child2, -1)

            if event_flag in [4, 6, 8, 14, 19, 20]:
                if event_flag in [4, 19, 20] and child1.index == index1:  ### RLOF
                    ell = Ellipse(xy=[child1.x, child1.y], width=0.5 * np.fabs(child2.x - child1.x), height=0.5 * np.fabs(y - child1.y), angle=0.0, color='tab:orange', alpha=0.5)
                    plot.add_artist(ell)

                if event_flag in [6] and child1.index == index1:  ### CE
                    ell = Ellipse(xy=[x, child1.y], width=1.5 * np.fabs(child2.x - child1.x), height=1.5 * np.fabs(y - child1.y), angle=0.0, color='tab:red', alpha=0.5)
                    plot.add_artist(ell)

                if child1.index in [index1, index2] or child2.index in [index1, index2]:
                    plot.plot([child1.x, child2.x], [child1.y, child2.y], color='r', zorder=8)
                    if child1.index == index1:
                        plot.arrow(child1.x, child1.y, 0.5 * (child2.x - child1.x), 0, head_width=0.08, head_length=0.3 * np.fabs(child2.x - child1.x), zorder=9, color='r')
                    else:
                        plot.arrow(child2.x, child2.y, -0.5 * np.fabs(child2.x - child1.x), 0, head_width=0.08, head_length=0.1 * np.fabs(child2.x - child1.x), zorder=9, color='r')
            if event_flag in [2, 12]:
                if child1.index == index1:
                    color, s, description, marker = Tools.get_color_and_size_and_description_for_star(child1.stellar_type, child1.radius, mass=child1.mass)
                    plot.scatter([child1.x], [child1.y], color=color, s=3 * s, zorder=9, marker='*')
                if child2.index == index1:
                    color, s, description, marker = Tools.get_color_and_size_and_description_for_star(child2.stellar_type, child2.radius, mass=child2.mass)
                    plot.scatter([child2.x], [child2.y], color=color, s=3 * s, zorder=9, marker='*')

        if child2.is_binary == True:

            if event_flag in [14]:  ### Triple CE
                ell = Ellipse(xy=[x, child2.y], width=2.0 * (child2.x - child1.x), height=3.5 * (y - child2.y), angle=0.0, color='tab:red', alpha=0.5)
                plot.add_artist(ell)

            i_rel_child = Tools.compute_mutual_inclination(particle.INCL, child2.INCL, particle.LAN, child2.LAN) * (180.0 / np.pi)
            x_min, x_max, y_min, y_max = Tools.draw_binary_node(plot, child2, line_width_horizontal, line_width_vertical, line_color, line_width, fontsize, x_min, x_max, y_min, y_max, index1=index1, index2=index2, event_flag=event_flag, i_rel=i_rel_child, orbit_info_list=orbit_info_list, orbit_counter=orbit_counter)
        else:
            _draw_body(child2, child1, 1)

            if event_flag in [4, 6, 8, 14, 19, 20]:
                if event_flag in [4, 19, 20] and child2.index == index1:  ### RLOF
                    ell = Ellipse(xy=[child2.x, child2.y], width=0.5 * np.fabs(child2.x - child1.x), height=0.5 * np.fabs(y - child1.y), angle=0.0, color='tab:orange', alpha=0.5)
                    plot.add_artist(ell)

                if event_flag in [6] and child2.index == index1:  ### CE
                    ell = Ellipse(xy=[x, child1.y], width=1.5 * np.fabs(child2.x - child1.x), height=1.5 * np.fabs(y - child1.y), angle=0.0, color='tab:red', alpha=0.5)
                    plot.add_artist(ell)

                if child1.index in [index1, index2] or child2.index in [index1, index2]:
                    plot.plot([child1.x, child2.x], [child1.y, child2.y], color='r', zorder=8)
                    if child1.index == index1:
                        plot.arrow(child1.x, child1.y, 0.5 * (child2.x - child1.x), 0, head_width=0.05, head_length=0.3 * np.fabs(child2.x - child1.x), zorder=9, color='r')
                    else:
                        plot.arrow(child2.x, child2.y, -0.5 * np.fabs(child2.x - child1.x), 0, head_width=0.05, head_length=0.1 * np.fabs(child2.x - child1.x), zorder=9, color='r')

            if event_flag in [2, 12]:
                if child1.index == index1:
                    color, s, description, marker = Tools.get_color_and_size_and_description_for_star(child1.stellar_type, child1.radius, mass=child1.mass)
                    plot.scatter([child1.x], [child1.y], color=color, s=3 * s, zorder=9, marker='*')
                if child2.index == index1:
                    color, s, description, marker = Tools.get_color_and_size_and_description_for_star(child2.stellar_type, child2.radius, mass=child2.mass)
                    plot.scatter([child2.x], [child2.y], color=color, s=3 * s, zorder=9, marker='*')

        ### Collect orbit info AFTER children (inner orbits get lower indices) ###
        orbit_counter[0] += 1
        orbit_idx = orbit_counter[0]
        info = {'a': particle.a, 'e': particle.e, 'color': particle.color, 'orbit_index': orbit_idx}
        if i_rel is not None:
            info['i_rel'] = i_rel
        orbit_info_list.append(info)

        return x_min, x_max, y_min, y_max


    @staticmethod
    def draw_bodies(plot, bodies, fontsize, y_ref=1.0, dx=0.5, dy=0.5, index1=-1, index2=-1, event_flag=-1):
        CONST_R_SUN = 0.004649130343817401
        CONST_KM_PER_AU = 1.4966e8
        CONST_MJUP = 0.000954248

        body_spacing = 4.0  # horizontal spacing between unbound bodies
        for index, body in enumerate(bodies):
            bx = index * body_spacing
            color, s, description, marker = Tools.get_color_and_size_and_description_for_star(body.stellar_type, body.radius, mass=body.mass)
            body.plot_x = bx
            body.plot_y = y_ref
            plot.scatter([bx], [y_ref], c=[color], s=s, zorder=10,
                         edgecolors='black', linewidth=2.0, alpha=0.8, marker=marker)

            if getattr(body, 'object_type', 1) == 2:
                mass_text = r"$%.1f\,M_\mathrm{J}$" % (body.mass / CONST_MJUP)
            else:
                mass_text = r"$%.2f\,M_\odot$" % body.mass

            type_text = description

            r_rsun = body.radius / CONST_R_SUN
            if r_rsun < 0.001:
                r_km = body.radius * CONST_KM_PER_AU
                radius_text = r"$%.0f$ km" % r_km
            else:
                radius_text = r"$R=%.2f\,R_\odot$" % r_rsun

            label_block = type_text + "\n" + mass_text + "\n" + radius_text
            plot.annotate(label_block, xy=(bx, y_ref), xytext=(0, -24),
                          textcoords='offset points', ha='center', va='top',
                          color='k', fontsize=12, zorder=10, linespacing=1.4)

            if body.index == index1:
                if event_flag in [2, 12]:
                    plot.scatter([bx], [y_ref], color=color, s=3 * s, zorder=9, marker='*')
            try:
                VX = body.VX
                VY = body.VY
                VZ = body.VZ
                V = np.sqrt(VX ** 2 + VY ** 2 + VZ ** 2)
                by = y_ref
                Adx = 0.5 * dx * VX / V
                Ady = 0.5 * dx * VY / V
                plot.arrow(bx, by, Adx, Ady, color=color, head_width=0.05, head_length=0.05)
            except AttributeError:
                pass

        event_bodies = [x for x in bodies if x.index in [index1, index2]]
        if len(event_bodies) == 2:

            if event_flag in [4, 6, 8]:
                child1 = event_bodies[0]
                child2 = event_bodies[1]

                plot.plot([child1.plot_x, child2.plot_x], [child1.plot_y, child2.plot_y], color='r', zorder=8)
                if child1.index == index1:
                    plot.arrow(child1.plot_x, child1.plot_y, 0.5 * (child2.plot_x - child1.plot_x), 0, head_width=0.05, head_length=0.1 * np.fabs(child2.plot_x - child1.plot_x), zorder=9, color='r')
                else:
                    plot.arrow(child2.plot_x, child2.plot_y, -0.5 * np.fabs(child2.plot_x - child1.plot_x), 0, head_width=0.05, head_length=0.1 * np.fabs(child2.plot_x - child1.plot_x), zorder=9, color='r')

        plot.set_xlim([-2 * dx, len(bodies)])
        plot.set_ylim([y_ref - 2 * dy, y_ref + 2 * dy])

        plot.set_xticks([])
        plot.set_yticks([])


    @staticmethod
    def get_color_and_size_and_description_for_star(stellar_type, radius, mass=None):
        import matplotlib.colors as mcolors

        if (stellar_type == 0):
            color = 'gold'
            description = 'dM'
        elif (stellar_type == 1):
            color = 'gold'
            description = 'MS'
        elif (stellar_type == 2):
            color = 'darkorange'
            description = 'HG'
        elif (stellar_type == 3):
            color = 'firebrick'
            description = 'RGB'
        elif (stellar_type == 4):
            color = 'darkorange'
            description = 'CHeB'
        elif (stellar_type == 5):
            color = 'orangered'
            description = 'EAGB'
        elif (stellar_type == 6):
            color = 'crimson'
            description = 'TPAGB'
        elif (stellar_type == 7):
            color = 'royalblue'
            description = 'HeMS'
        elif (stellar_type == 8):
            color = 'orangered'
            description = 'HeHG'
        elif (stellar_type == 9):
            color = 'crimson'
            description = 'HeGB'
        elif (stellar_type == 10):
            color = 'silver'
            description = 'HeWD'
        elif (stellar_type == 11):
            color = 'silver'
            description = 'COWD'
        elif (stellar_type == 12):
            color = 'silver'
            description = 'ONeWD'
        elif (stellar_type == 13):
            color = 'gainsboro'
            description = 'NS'
        elif (stellar_type == 14):
            color = 'k'
            description = 'BH'
        elif (stellar_type == 15):
            color = 'k'
            description = 'MR'
        else:
            color = 'k'
            description = ''

        marker = 'o'

        if stellar_type == 15:
            s = 40
            marker = 'x'
        elif stellar_type == 10:
            s = 60
            rgb = mcolors.to_rgb(color)
            color = tuple(min(1.0, c + 0.3) for c in rgb)
        elif stellar_type == 11:
            s = 70
        elif stellar_type == 12:
            s = 80
            rgb = mcolors.to_rgb(color)
            color = tuple(max(0.0, c - 0.2) for c in rgb)
        elif stellar_type == 13:
            s = 50
        elif stellar_type == 14:
            s = 60
        else:
            m = mass if mass is not None else 1.0
            s = max(80, 150 * (1 + np.log10(max(m, 0.1))))
            if stellar_type in [2, 3, 4, 5, 6]:
                s *= 1.5

        return color, s, description, marker


    @staticmethod
    def get_description_for_event_flag(event_flag, SNe_type=None):
        if event_flag == 0:
            text = r"$\mathrm{Initial\,system}$"
        elif event_flag == 1:
            text = r"$\mathrm{Stellar\,type\,change}$"
        elif event_flag == 2:
            if SNe_type is not None:
                SNe_type_string = ""
                if SNe_type == 1:
                    SNe_type_string = r"Type\,Ia"
                if SNe_type == 2:
                    SNe_type_string = r"Type\,II"
                if SNe_type == 3:
                    SNe_type_string = r"Electron\,capture"
                if SNe_type == 4:
                    SNe_type_string = r"Type\,Ib"
                text = r"$\mathrm{SNe\,start\,(%s)}$" % SNe_type_string
            else:
                text = r"$\mathrm{SNe\,start}$"
        elif event_flag == 3:
            text = r"$\mathrm{SNe\,end}$"
        elif event_flag == 4:
            text = r"$\mathrm{RLOF\,start}$"
        elif event_flag == 5:
            text = r"$\mathrm{RLOF\,end}$"
        elif event_flag == 6:
            text = r"$\mathrm{CE\,start}$"
        elif event_flag == 7:
            text = r"$\mathrm{CE\,end}$"
        elif event_flag == 8:
            text = r"$\mathrm{Collision\,start}$"
        elif event_flag == 9:
            text = r"$\mathrm{Collision\,end}$"
        elif event_flag == 10:
            text = r"$\mathrm{Dyn.\,inst.}$"
        elif event_flag == 11:
            text = r"$\mathrm{Sec.\,break.}$"
        elif event_flag == 12:
            text = r"$\mathrm{WD\,kick\,start}$"
        elif event_flag == 13:
            text = r"$\mathrm{WD\,kick\,end}$"
        elif event_flag == 14:
            text = r"$\mathrm{Triple\,CE\,start}$"
        elif event_flag == 15:
            text = r"$\mathrm{Triple\,CE\,end}$"
        elif event_flag == 16:
            text = r"$\mathrm{MSP\,formation}$"
        elif event_flag == 17:
            text = r"$\mathrm{Final\,state}$"
        elif event_flag == 18:
            text = r"$\mathrm{sdB\,formation}$"
        elif event_flag == 19:
            text = r"$\mathrm{RLOF\,low\,mass\,donor}$"
        elif event_flag == 20:
            text = r"$\mathrm{RLOF\,WD\,donor}$"
        elif event_flag == 21:
            text = r"$\mathrm{Entering\,LISA\,band}$"
        elif event_flag == 22:
            text = r"$\mathrm{Start\,N\!-\!body}$"
        elif event_flag == 23:
            text = r"$\mathrm{End\,N\!-\!body}$"
        else:
            text = ""
        return text
