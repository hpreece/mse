/* SecularMultiple */
/* Adrian Hamers November 2019 */

#include "evolve.h"
#include "SNe.h"

extern "C"
{

int handle_SNe_in_system(ParticlesMap *particlesMap, bool *unbound_orbits)
{
    int flag;
    double vx,vy,vz;
    ParticlesMapIterator it_p;
    //std::vector<int>::iterator it_parent_p,it_parent_q;

//    int seed = orbital_phases_random_seed;
    int index=0;
    
    for (it_p = particlesMap->begin(); it_p != particlesMap->end(); it_p++)
    {
        Particle *p = (*it_p).second;
        if (p->is_binary == 0 and p->evolve_as_star == 1)
        {
            /* delta m was calculated in evolve_stars (stellar_evolution.cpp) */
            p->instantaneous_perturbation_delta_x = 0.0;
            p->instantaneous_perturbation_delta_y = 0.0;
            p->instantaneous_perturbation_delta_z = 0.0;

            if (fabs(p->instantaneous_perturbation_delta_mass) > 0.0)
            {
                flag = sample_kick_velocity(p,&vx,&vy,&vz);
                printf("SNe.cpp -- vk %g % g %g\n",vx,vy,vz);
                p->instantaneous_perturbation_delta_vx = vx;
                p->instantaneous_perturbation_delta_vy = vy;
                p->instantaneous_perturbation_delta_vz = vz;
                index+=1;
            }
            
        }
    }


    apply_user_specified_instantaneous_perturbation(particlesMap);

    *unbound_orbits = check_for_unbound_orbits(particlesMap);
    
    reset_instantaneous_perturbation_quantities(particlesMap);
            
    return 0;
}

int sample_kick_velocity(Particle *p, double *vx, double *vy, double *vz)
{
    //srand(seed);
    double x;
    x = ((double) rand() / (RAND_MAX));
    double theta = 2.0*M_PI*x - M_PI;
    x = ((double) rand() / (RAND_MAX));
    double phi = 2.0*M_PI*x;
    *vx = sin(theta)*cos(phi);
    *vy = sin(theta)*sin(phi);
    *vz = cos(theta);

    double vnorm;
    
    if (p->kick_distribution == 0) // no kicks
    {
        vnorm = 0.0;
    }
    else if (p->kick_distribution == 1) // Maxwellian
    {
        double sigma = p->kick_distribution_sigma;
        double v[3];
        sample_from_3d_maxwellian_distribution(sigma, v);
        //vnorm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        vnorm = norm3(v);
    }
    
  //    std::default_random_engine generator (seed);

//  std::normal_distribution<double> distribution (0.0,1.0);

//    vnorm = 0.0;
    printf("SNe.cpp -- distr %d sigma %g vnorm %g\n",p->kick_distribution,p->kick_distribution_sigma,vnorm);
    *vx *= vnorm;
    *vy *= vnorm;
    *vz *= vnorm;
    
    return 0;
}

bool check_for_unbound_orbits(ParticlesMap *particlesMap)
{
    ParticlesMapIterator it_p;
    double h_vec[3],e_vec[3];
    double e;
    
    bool unbound_orbits = false;
    
    for (it_p = particlesMap->begin(); it_p != particlesMap->end(); it_p++)
    {
        Particle *p = (*it_p).second;
        if (p->is_binary == 1)
        {
            get_e_and_h_vectors_from_particle(p,e_vec,h_vec);
            e = norm3(e_vec);
            
            if (e<0 or e > 1.0)
            {
                unbound_orbits = true;
            }
            //printf("test e %g\n",e);
        }
    }
    
    return unbound_orbits;
}

}