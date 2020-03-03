/* SecularMultiple */
/* Adrian Hamers November 2019 */

#include "evolve.h"

extern "C"
{

int evolve(ParticlesMap *particlesMap, double start_time, double end_time, double *output_time, double *hamiltonian, int *state, int *CVODE_flag, int *CVODE_error_code)
{
    //printf("evolve \n");
    
    /* Obtain stellar evolution metallicity-dependent parameters (only has to be done once) */
//    double z = 0.02; /* change in future */
//    double *zpars;
//    zpars = new double[20];
//    zcnsts_(&z,zpars);

    /* WARNING: TEMPORARY CODE TO QUICKLY SET MASS TRANSFER RATES FOR TESTING PURPOSES */
    Particle *s1 = (*particlesMap)[0];
    Particle *s2 = (*particlesMap)[1];
    s1->mass_dot_RLOF = -1.0e-5;
    s2->mass_dot_RLOF = +1.0e-5;

    if (start_time == end_time)
    {
        *output_time = start_time;
        *hamiltonian = 0.0;
        *state = 0.0;
        *CVODE_flag = 0;
        *CVODE_error_code = 0;
        
        return 0;
    }
    
    /* Set up integration variables */
    int flag;
    double t = start_time;
    double t_old = t;
    double t_end = end_time;
    double dt,dt_stev,dt_binary_evolution;
    double t_out;
    //double t_next_encounter = 0.0;

    bool apply_SNe_effects = false;
    bool unbound_orbits = false;

    //int N_enc,N_not_impulsive;
    //if (include_flybys == true)
    //{
        //handle_next_flyby(particlesMap, true, &unbound_orbits);
    //}
    //printf("t_next_encounter %g\n",flybys_t_next_encounter);
    
    //double hamiltonian;
    //int output_flag,error_code;
    //printf("start time %g time step %g\n",start_time,time_step);
        
        
    //initialize_stars(particlesMap);
    
    /* Get initial timestep -- set by stellar evolution or user output time */
    double min_dt = 1.0e0;
    double max_dt = end_time - start_time;
    flag = evolve_stars(particlesMap,t,0.0,&dt,true,&apply_SNe_effects);
    //printf("dt0 %g max_dt %g\n",dt,max_dt);
    dt = min(dt,min_dt);
    dt_binary_evolution = min_dt;
    
    //printf("initial dt %g\n",dt);
    /* Time loop */
    int i=0;
    *state = 0;
    bool last_iteration = false;
    while (t <= t_end)
    //while (q<10)
    {
        t += dt;

        /* Stellar evolution */

//        printf("i %d\n",i);
        //printf("1 t %g t_old %g dt %g t-t_old %g\n",t,t_old,dt,t-t_old);
        //flag = evolve_stars(particlesMap,t_old,t,&dt_stev,false,&apply_SNe_effects);
        flag = evolve_stars(particlesMap,t_old,t,&dt_stev,false,&apply_SNe_effects);
        //printf("dt_stev %g\n",dt_stev);
        if (apply_SNe_effects == true)
        {
            flag = handle_SNe_in_system(particlesMap,&unbound_orbits);
            if (unbound_orbits == true)
            {
                printf("Unbound orbits in system due to supernova!\n");
                *state = 3; /* TO DO: make general state macros */
                break;
            }
        }

        /* Secular dynamical evolution */
        flag = integrate_ODE_system(particlesMap,t_old,t,&t_out,hamiltonian,CVODE_flag,CVODE_error_code);
        
        #ifdef VERBOSE
        printf("ODE dt %g t_old - t %g t - t_out %g\n",dt,t_old-t,t-t_out);
        #endif
        
        /* Handle roots */
        if (*CVODE_flag==2)
        {
            printf("ROOT occurred; setting t = t_out; t %g t_out %g\n",t,t_out);
            t = t_out;
            
            flag = check_if_RLOF_has_occurred(particlesMap);
            if (flag == 0)
            {
                //break;
            }
            else
            {
                
                printf("RLOF occurred, but continuing dt %g t_old - t %g t - t_out %g\n",dt,t_old-t,t-t_out);
                *CVODE_flag = 0;
            }
        }
        
        dt = dt_stev;
        
        
        
        /* Secular mass transfer */
        /* Includes possibility of CE evolution */
        flag = handle_binary_evolution(particlesMap,&dt_binary_evolution);
        
        dt = min(dt,dt_binary_evolution);
        
        /* Flybys */
        //printf("test dt %g\n",dt);
        if (include_flybys == true)
        {

            if (t + dt >= flybys_t_next_encounter and last_iteration == false)
            {
                //printf("NE t+dt %g flybys_t_next_encounter %g \n",t+dt,flybys_t_next_encounter);
                dt = flybys_t_next_encounter - t;
            
                handle_next_flyby(particlesMap, false, &unbound_orbits);

                if (unbound_orbits == true)
                {
                    printf("Unbound orbits in system due to flyby!\n");
                    *state = 4; /* TO DO: make general state macros */
                    break;
                }
                //N_enc,N_not_impulsive,time_next_encounter,M_per,b_vec,V_per_vec = sample_next_impulsive_encounter(cmd_options,simulation_parameters_particle,time_next_encounter,N_enc,N_not_impulsive,output_arrays,store_encounter_data,W_max,total_encounter_rate,initial_final_mass_data)
            
                //stop = apply_impulsive_encounter(cmd_options,secular_code,channel_from_particles_to_secular_code,channel_from_secular_code_to_particles,simulation_parameters_particle.particles,M_per,b_vec,V_per_vec)

                //if stop == True:
                  //  print 'index system ',simulation_parameters_particle.index,' unbound due to flyby'
                    //secular_code.flag = 400
            }
        }
        
        /* Time-step without flybys */

        //dt = max(dt,1.0e0);

        if (t + dt >= end_time)
        {
            dt = end_time - t;
            last_iteration = true;
            //printf("adjust dt to reach end time \n");
        }


        dt = min(dt,max_dt);
        
        dt = max(dt,min_dt);


        
        //printf("test2 dt %g\n",dt);
        //double dt_temp = 1.0e-1;
        //dt = min(dt_stev,dt_temp);
        
        //*output_flag = flag;
      //  printf("t %g dt %g t_out %g t-t_out %g i %d\n",t,dt,t_out,t-t_out,i);

        //printf("test2 dt %g dt_stev %g\n",dt,dt_stev);
        
        //dt = min(dt, t_out - t_old);
        
        t_old = t;
        
        //dt = time_step*1.0e-3;
//        printf("2 t %g t_old %g dt %g\n",t,t_old,dt);

        if (t >= t_end)
        {
            //printf("breakingn\n");
            break;
        }
        i+=1;
//        if (q > 10)
//        {
//            printf("????????????????\n");
            //exit(0);
//        }

    }
    //printf("done %d\n",i);
    
    *output_time = t;
    *hamiltonian = 0.0;
    //*output_flag = 0;
    //*error_code = error_code;
    
    return 0;
}

int integrate_ODE_system(ParticlesMap *particlesMap, double start_time, double end_time, double *output_time, double *hamiltonian, int *output_flag, int *error_code)
{
    int N_particles = particlesMap->size();
    int N_bodies, N_binaries;
    int N_root_finding;
    
    determine_binary_parents_and_levels(particlesMap,&N_bodies,&N_binaries,&N_root_finding);
    set_binary_masses_from_body_masses(particlesMap);

//    printf("N_bodies %d N_binaries %d N_particles %d N_root_finding %d\n",N_bodies,N_binaries,N_particles,N_root_finding);

    /*********************
     * setup of UserData *
     ********************/
     
    double CVODE_start_time = 0.0;
    double CVODE_time_offset = start_time;
    double CVODE_end_time = end_time - CVODE_time_offset;
    //printf("CVODE %g %g\n",CVODE_start_time,CVODE_end_time);

	UserData data;
	data = NULL;
	data = (UserData) malloc(sizeof *data);
	data->particlesMap = particlesMap;
    data->N_root_finding = N_root_finding;
    data->start_time = CVODE_start_time;

    
    /********************************
     * set ODE tolerances   *
     ********************************/
    if (relative_tolerance <= 0.0)
    {
        printf("relative tolerance cannot be zero; setting default value of 1e-16\n");
        relative_tolerance = 1.0e-15;
    }

    /* Warning: hardcoded parameters for ODE solver */
    //double abs_tol_spin_vec = 1.0e-12;
    double abs_tol_spin_vec = 1.0e4;
    double abs_tol_e_vec = absolute_tolerance_eccentricity_vectors;
    //abs_tol_e_vec = 1.0e-10;
    double abs_tol_h_vec = 1.0e-2;
    double initial_ODE_timestep = 1.0e0; /* one year */
    int maximum_number_of_internal_ODE_steps = 5e8;
    int maximum_number_of_convergence_failures = 100;    
    double maximum_ODE_integration_time = 1.0e20;

    /***************************
     * setup of ODE variables  *
     **************************/    
	N_Vector y, y_out, y_abs_tol;
	void *cvode_mem;
	int flag;

	y = y_out = y_abs_tol = NULL;
	cvode_mem = NULL;

    int number_of_ODE_variables = N_bodies*5 + N_binaries*6; // spin vectors + mass + radius for each body + e & h vectors for each binary
//    printf("N_ODE %d\n",number_of_ODE_variables);
    
//    data->number_of_ODE_variables = number_of_ODE_variables;
    y = N_VNew_Serial(number_of_ODE_variables);
	if (check_flag((void *)y, "N_VNew_Serial", 0)) return 1;
    y_out = N_VNew_Serial(number_of_ODE_variables);
	if (check_flag((void *)y_out, "N_VNew_Serial", 0)) return 1;
    y_abs_tol = N_VNew_Serial(number_of_ODE_variables); 
	if (check_flag((void *)y_abs_tol, "N_VNew_Serial", 0)) return 1;         

    set_initial_ODE_variables(particlesMap, y, y_abs_tol,abs_tol_spin_vec,abs_tol_e_vec,abs_tol_h_vec);

    /***************************
     * setup of ODE integrator *
     **************************/    

    /* use Backward Differentiation Formulas (BDF)
        scheme in conjunction with Newton iteration --
        these choices are recommended for stiff ODEs
        in the CVODE manual                          
    */

    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
	if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return 1;    
    
    /* essential initializations */
    flag = CVodeInit(cvode_mem, compute_y_dot, CVODE_start_time, y);
	if (check_flag(&flag, "CVodeInit", 1)) return 1;    

	flag = CVodeSetUserData(cvode_mem, data);
	if (check_flag(&flag, "CVodeSetUsetData", 1)) return 1;

    flag = CVodeSVtolerances(cvode_mem, relative_tolerance, y_abs_tol);
	if (check_flag(&flag, "CVodeSVtolerances", 1)) return 1;

	flag = CVDense(cvode_mem, number_of_ODE_variables);
	if (check_flag(&flag, "CVDense", 1)) return 1;

	flag = CVodeSetInitStep(cvode_mem, initial_ODE_timestep);
	if (check_flag(&flag, "CVodeSetInitStep", 1)) return 1;

    /* optional initializations */
//	flag = CVodeSetErrHandlerFn(cvode_mem, ehfun, eh_data); // error handling function
//	if (check_flag(&flag, "CVodeSetErrHandlerFn", 1)) return;
	  		
	flag = CVodeSetMaxNumSteps(cvode_mem, maximum_number_of_internal_ODE_steps);
	if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return 1;

	flag = CVodeSetMinStep(cvode_mem, 1.0e-15); // minimum step size
	if (check_flag(&flag, "CVodeSetMinStep", 1)) return 1;

	flag = CVodeSetMaxHnilWarns(cvode_mem, 1);
	if (check_flag(&flag, "CVodeSetMaxHnilWarns", 1)) return 1;
			
//	flag = CVodeSetStopTime(cvode_mem, MAXTIME); // maximum time
//	if (check_flag(&flag, "CVodeSetStopTime", 1)) return 1;

	flag = CVodeSetMaxConvFails(cvode_mem, maximum_number_of_convergence_failures);
	if (check_flag(&flag, "CVodeSetMaxConvFails", 1)) return 1;

    /* initialization of root finding */
    int roots_found[N_root_finding];
	flag = CVodeRootInit(cvode_mem, N_root_finding, root_finding_functions);
	if (check_flag(&flag, "CVodeRootInit", 1)) return 1;	


    /***************************
     * ODE integration         *
     **************************/ 
    
	//double user_end_time = end_time;
	double CVODE_reached_end_time;

	flag = CVode(cvode_mem, CVODE_end_time, y_out, &CVODE_reached_end_time, CV_NORMAL);	
    double reached_end_time = CVODE_reached_end_time + CVODE_time_offset;

    if (check_for_initial_roots(particlesMap) > 0)
    {
        flag = CV_ROOT_RETURN;
    }

	if (flag == CV_SUCCESS)
	{
		*output_flag = CV_SUCCESS;
		*error_code = 0;
        *output_time = reached_end_time;
	}
	else if (flag == CV_ROOT_RETURN) // a root was found during the integration
	{
		CVodeGetRootInfo(cvode_mem,roots_found);
        read_root_finding_data(particlesMap,roots_found);
        
        *output_flag = CV_ROOT_RETURN;
        *error_code = 0;
        *output_time = reached_end_time;
    }
    else if (flag == CV_WARNING) // a warning has occurred during the integration
    {
		*output_flag = CV_WARNING;
		*error_code = flag;
    }
	else // an error has occurred during the integration
    {
		*output_flag = flag;
		*error_code = flag;
    }

    /***************************
     * y_out -> particlesMap   *
     * ************************/

    extract_final_ODE_variables(particlesMap,y_out);
    update_position_vectors_external_particles(particlesMap,reached_end_time);

    *hamiltonian = data->hamiltonian;
    
    N_VDestroy_Serial(y);
    N_VDestroy_Serial(y_out);
    N_VDestroy_Serial(y_abs_tol);
    CVodeFree(&cvode_mem);

	return 0;
    
}

/* function to check ODE solver-related function return values */
static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
	      funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  return 0;
}

}