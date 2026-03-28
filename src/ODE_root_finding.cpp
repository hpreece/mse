/* MSE */

#include "types.h"
#include "evolve.h"
#include "ODE_root_finding.h"
#include "mlp_forward_pass.h"
#include <stdio.h>

#define MLP_INSTABILITY_ENTRY_THRESHOLD 0.5
#define MLP_INSTABILITY_EXIT_THRESHOLD 0.3

static bool mlp_features_in_range_triple(const double f[6])
{
    if (f[0] < 0.01 || f[0] > 1.0) return false;
    if (f[1] < 0.01 || f[1] > 50.0) return false;
    if (f[2] < 0.001 || f[2] > 0.95) return false;
    if (f[3] < 0.0 || f[3] > 0.9999) return false;
    if (f[4] < 0.0 || f[4] > 0.9999) return false;
    if (f[5] < 0.0 || f[5] > 1.0) return false;
    return true;
}

static bool mlp_features_in_range_quad_3p1(const double f[11])
{
    if (f[0] < 0.01 || f[0] > 1.0) return false;
    if (f[1] < 0.01 || f[1] > 50.0) return false;
    if (f[2] < 0.01 || f[2] > 50.0) return false;
    if (f[3] < 0.001 || f[3] > 0.95) return false;
    if (f[4] < 0.001 || f[4] > 0.95) return false;
    if (f[5] < 0.0 || f[5] > 0.9999) return false;
    if (f[6] < 0.0 || f[6] > 0.9999) return false;
    if (f[7] < 0.0 || f[7] > 0.9999) return false;
    if (f[8] < 0.0 || f[8] > 1.0) return false;
    if (f[9] < 0.0 || f[9] > 1.0) return false;
    if (f[10] < 0.0 || f[10] > 1.0) return false;
    return true;
}

static bool mlp_features_in_range_quad_2p2(const double f[11])
{
    if (f[0] < 0.01 || f[0] > 1.0) return false;   /* qi1 */
    if (f[1] < 0.01 || f[1] > 1.0) return false;   /* qi2 */
    if (f[2] < 0.01 || f[2] > 1.0) return false;   /* qo  */
    if (f[3] < 0.001 || f[3] > 0.95) return false;  /* ali1o */
    if (f[4] < 0.001 || f[4] > 0.95) return false;  /* ali2o */
    if (f[5] < 0.0 || f[5] > 0.9999) return false;
    if (f[6] < 0.0 || f[6] > 0.9999) return false;
    if (f[7] < 0.0 || f[7] > 0.9999) return false;
    if (f[8] < 0.0 || f[8] > 1.0) return false;
    if (f[9] < 0.0 || f[9] > 1.0) return false;
    if (f[10] < 0.0 || f[10] > 1.0) return false;
    return true;
}

extern "C"
{
int root_finding_functions(realtype time, N_Vector y, realtype *root_functions, void *data_)
{
    UserData data;
    data = (UserData) data_;
    ParticlesMap *particlesMap = data->particlesMap;
    int N_root_finding = data->N_root_finding;
    double start_time = data->start_time;
    double delta_time = time - start_time;

    double large_quantity = 1.0e10;    
    for (int i=0; i<N_root_finding; i++)
    {
        root_functions[i] = large_quantity;
    }
    
    extract_ODE_variables(particlesMap, y, delta_time);
       
    check_for_roots(particlesMap, true, root_functions);
    return 0;
}

void check_for_roots(ParticlesMap *particlesMap, bool use_root_functions, realtype *root_functions)
{
    int i_root = 0;

    ParticlesMapIterator it_p;
    std::vector<int>::iterator it_parent;
    
    double f_root = 1.0e100;
    
    for (it_p = particlesMap->begin(); it_p != particlesMap->end(); it_p++)
    {
        Particle *P_p = (*it_p).second;
        if (P_p->is_binary == true)
        {
            if (P_p->check_for_secular_breakdown == true)
            {
                if (P_p->parent != -1)
                {
                    
                    double hamiltonian=0.0;
                    double KS_V=0.0;
                    compute_EOM_Newtonian_for_particle(particlesMap,P_p,&hamiltonian,&KS_V,false);
                    
                    double AM_time_scale = compute_AM_time_scale(P_p);
                    double orbital_period = compute_orbital_period_from_semimajor_axis(P_p->mass,P_p->a);

                    f_root = 1.0 - AM_time_scale/orbital_period;

                    #ifdef VERBOSE
                    if (verbose_flag > 2)
                    {
                        printf("root_finding.cpp -- check_for_roots -- check_for_secular_breakdown f_root %g\n",f_root);
                    }
                    #endif
                    
                    if (use_root_functions == true)
                    {
                        root_functions[i_root] = f_root;
                    }
                    else
                    {
                        if (f_root >= 0.0)
                        {
                            P_p->secular_breakdown_has_occurred = true;
                        }
                        else
                        {
                            P_p->secular_breakdown_has_occurred = false;
                        }
                    }
                }
                
                i_root++;
            }
            
            if (P_p->check_for_dynamical_instability == true)
            {

                if (P_p->parent != -1)
                {
                    Particle *P_parent = (*particlesMap)[P_p->parent];
                    Particle *P_child1 = (*particlesMap)[P_p->child1];
                    Particle *P_child2 = (*particlesMap)[P_p->child2];
                
                    double a_out = P_parent->a;
                    double e_out = P_parent->e;
                    double a_in = P_p->a;
                    double e_in = P_p->e;
                    double M_p = P_p->mass;
                    double ra_in = a_in*(1.0+e_in);
                    double rp_out = a_out*(1.0-e_out);
                    
                    #ifdef VERBOSE
                    if (verbose_flag > 2)
                    {
                        printf("ODE_root_finding.cpp -- check_for_roots --  p id %d parent %d c1 %d c2 %d \n",P_p->index,P_parent->index,P_child1->index,P_child2->index);
                        print_system(particlesMap,0);
                    }
                    #endif

                    Particle *P_sibling = (*particlesMap)[P_p->sibling];

                    if (P_p->dynamical_instability_criterion == 0) /* for mass ratios on the order of unity */
                    {
                        /* Mardling & Aarseth 2001 */
                        double q_out = P_sibling->mass/M_p;
                        double rel_INCL = 0.0;
                        get_inclination_relative_to_parent(particlesMap,P_p->index,&rel_INCL);
                        double rp_out_crit = compute_rp_out_crit_MA01(a_in,q_out,e_out,rel_INCL);
                        
                        f_root = rp_out - rp_out_crit;

                    }
                    else if (P_p->dynamical_instability_criterion == 1) /* for S-type test particles in binaries */
                    {
                        /* Wiegert & Holman 1999 */
                        double m1 = M_p;
                        double m2 = P_sibling->mass;
                        double mu = m2/(m1+m2);
                        double e = e_out;
                        f_root = (0.464 - 0.38*mu - 0.631*e + 0.586*mu*e + 0.15*e*e - 0.198*mu*e*e) - a_in/a_out;
                    }
                    else if (P_p->dynamical_instability_criterion == 2 || P_p->dynamical_instability_criterion == 3)
                    /* in case of a central dominant particle
                     * m1 is the `inner' mass; m2 is the `outer' mass */
                    {
                        int central_particle_index = P_p->dynamical_instability_central_particle;
                        Particle *P_central_particle = (*particlesMap)[central_particle_index];
                        double central_particle_mass = P_central_particle->mass;
                        double m2 = P_sibling->mass;
                        double m1;
                        if (P_p->child1 == central_particle_index)
                        {
                            m1 = P_child2->mass;
                        }
                        else if (P_p->child2 == central_particle_index)
                        {
                            m1 = P_child1->mass;
                        }

                        int central_particle_parent;
                        std::vector<int>::iterator it_C_parent;
                        for (it_C_parent = P_central_particle->parents.begin(); it_C_parent != P_central_particle->parents.end(); it_C_parent++)
                        {
                            central_particle_parent = *it_C_parent;
                            if (P_p->child1 == central_particle_parent)
                            {
                                m1 = P_child2->mass;
                            }
                            else if (P_p->child2 == central_particle_parent)
                            {
                                m1 = P_child1->mass;
                            }
                        }

                        if (P_p->dynamical_instability_criterion == 2)
                        {
                            double mu1 = m1/central_particle_mass;
                            double mu2 = m2/central_particle_mass;
                            double R_H = c_1div2*(a_in+a_out)*pow( c_1div3*(mu1+mu2), c_1div3 );

                            double K = P_p->dynamical_instability_K_parameter;
                            f_root = (rp_out - ra_in) - K*R_H;
                        }

                        else if (P_p->dynamical_instability_criterion == 3)
                        /* Petrovich 2015 */
                        {
                            double mu1 = m1/central_particle_mass;
                            double mu2 = m2/central_particle_mass;
                            double R_H = c_1div2*(a_in+a_out)*pow( c_1div3*(mu1+mu2), c_1div3 );

                            f_root = rp_out/ra_in - ( 2.4*pow( CV_max(mu1,mu2), c_1div3)*sqrt(a_out/a_in) + 1.15 );
                        }

                    }
                    else if (P_p->dynamical_instability_criterion == 4)
                    /* Vynatheya MLP stability classifier */
                    {
                        double P_unstable = 1.0; /* safe fallback: assume unstable */
                        bool mlp_out_of_range = false;

                        if (P_sibling->is_binary == false)
                        {
                            /* Triple or 3+1 quad (middle level) */
                            bool child1_is_binary = P_child1->is_binary;
                            bool child2_is_binary = P_child2->is_binary;

                            if (!child1_is_binary && !child2_is_binary)
                            {
                                /* Triple: P_p is inner binary with two body children, sibling is a body */
                                double m1 = P_child1->mass;
                                double m2 = P_child2->mass;
                                double m_comp = P_sibling->mass;

                                if (m1 > 0.0 && m2 > 0.0 && M_p > 0.0 && a_out > 0.0)
                                {
                                    double qi = (m1 < m2) ? m1/m2 : m2/m1;
                                    double qo = m_comp / M_p;
                                    double a_ratio = a_in / a_out;
                                    double rel_INCL = 0.0;
                                    get_inclination_relative_to_parent(particlesMap, P_p->index, &rel_INCL);

                                    double features[6] = {qi, qo, a_ratio, e_in, e_out, rel_INCL / M_PI};
                                    if (mlp_features_in_range_triple(features))
                                    {
                                        P_unstable = mlp_predict_triple(features);
                                    }
                                    else
                                    {
                                        mlp_out_of_range = true;
                                        #ifdef VERBOSE
                                        if (verbose_flag > 0)
                                        {
                                            printf("ODE_root_finding.cpp -- MLP triple features out of range, using MA01 fallback\n");
                                        }
                                        #endif
                                    }
                                }
                            }
                            else
                            {
                                /* 3+1 quad: P_p is the middle binary, one child is an inner binary */
                                Particle *P_inner = child1_is_binary ? P_child1 : P_child2;
                                Particle *P_mid_body = child1_is_binary ? P_child2 : P_child1;

                                Particle *P_inner_c1 = (*particlesMap)[P_inner->child1];
                                Particle *P_inner_c2 = (*particlesMap)[P_inner->child2];

                                double mi1 = P_inner_c1->mass;
                                double mi2 = P_inner_c2->mass;
                                double m_mid_comp = P_mid_body->mass;
                                double m_out_comp = P_sibling->mass;
                                double a_inner = P_inner->a;
                                double e_inner = P_inner->e;
                                double a_mid = a_in; /* P_p->a */
                                double e_mid = e_in; /* P_p->e */

                                if (mi1 > 0.0 && mi2 > 0.0 && P_inner->mass > 0.0 && M_p > 0.0 && a_inner > 0.0 && a_mid > 0.0 && a_out > 0.0)
                                {
                                    double qi = (mi1 < mi2) ? mi1/mi2 : mi2/mi1;
                                    double qm = m_mid_comp / P_inner->mass;
                                    double qo = m_out_comp / M_p;
                                    double alim = a_inner / a_mid;
                                    double almo = a_mid / a_out;

                                    /* Mutual inclinations from h_vec dot products */
                                    double h_inner_norm = norm3(P_inner->h_vec);
                                    double h_mid_norm = norm3(P_p->h_vec);
                                    double h_out_norm = norm3(P_parent->h_vec);

                                    double cos_iim = (h_inner_norm > 0.0 && h_mid_norm > 0.0) ?
                                        dot3(P_inner->h_vec, P_p->h_vec) / (h_inner_norm * h_mid_norm) : 1.0;
                                    double cos_iio = (h_inner_norm > 0.0 && h_out_norm > 0.0) ?
                                        dot3(P_inner->h_vec, P_parent->h_vec) / (h_inner_norm * h_out_norm) : 1.0;
                                    double cos_imo = (h_mid_norm > 0.0 && h_out_norm > 0.0) ?
                                        dot3(P_p->h_vec, P_parent->h_vec) / (h_mid_norm * h_out_norm) : 1.0;

                                    /* Clamp to [-1, 1] */
                                    if (cos_iim < -1.0) cos_iim = -1.0; if (cos_iim > 1.0) cos_iim = 1.0;
                                    if (cos_iio < -1.0) cos_iio = -1.0; if (cos_iio > 1.0) cos_iio = 1.0;
                                    if (cos_imo < -1.0) cos_imo = -1.0; if (cos_imo > 1.0) cos_imo = 1.0;

                                    double features[11] = {qi, qm, qo, alim, almo, e_inner, e_mid, e_out,
                                                           acos(cos_iim) / M_PI, acos(cos_iio) / M_PI, acos(cos_imo) / M_PI};
                                    if (mlp_features_in_range_quad_3p1(features))
                                    {
                                        P_unstable = mlp_predict_quad_3p1(features);
                                    }
                                    else
                                    {
                                        mlp_out_of_range = true;
                                        #ifdef VERBOSE
                                        if (verbose_flag > 0)
                                        {
                                            printf("ODE_root_finding.cpp -- MLP 3+1 quad features out of range, using MA01 fallback\n");
                                        }
                                        #endif
                                    }
                                }
                            }
                        }
                        else
                        {
                            /* 2+2 quad: both P_p and P_sibling are inner binaries under P_parent */
                            Particle *P_sib_c1 = (*particlesMap)[P_sibling->child1];
                            Particle *P_sib_c2 = (*particlesMap)[P_sibling->child2];

                            double mi1_1 = P_child1->mass;
                            double mi1_2 = P_child2->mass;
                            double mi2_1 = P_sib_c1->mass;
                            double mi2_2 = P_sib_c2->mass;
                            double a_sib = P_sibling->a;
                            double e_sib = P_sibling->e;

                            if (mi1_1 > 0.0 && mi1_2 > 0.0 && mi2_1 > 0.0 && mi2_2 > 0.0 && M_p > 0.0 && P_sibling->mass > 0.0 && a_in > 0.0 && a_sib > 0.0 && a_out > 0.0)
                            {
                                double qi1 = (mi1_1 < mi1_2) ? mi1_1/mi1_2 : mi1_2/mi1_1;
                                double qi2 = (mi2_1 < mi2_2) ? mi2_1/mi2_2 : mi2_2/mi2_1;
                                double qo = (M_p < P_sibling->mass) ? M_p/P_sibling->mass : P_sibling->mass/M_p;
                                double ali1o = a_in / a_out;
                                double ali2o = a_sib / a_out;

                                /* Mutual inclinations from h_vec dot products */
                                double h1_norm = norm3(P_p->h_vec);
                                double h2_norm = norm3(P_sibling->h_vec);
                                double ho_norm = norm3(P_parent->h_vec);

                                double cos_i12 = (h1_norm > 0.0 && h2_norm > 0.0) ?
                                    dot3(P_p->h_vec, P_sibling->h_vec) / (h1_norm * h2_norm) : 1.0;
                                double cos_i1o = (h1_norm > 0.0 && ho_norm > 0.0) ?
                                    dot3(P_p->h_vec, P_parent->h_vec) / (h1_norm * ho_norm) : 1.0;
                                double cos_i2o = (h2_norm > 0.0 && ho_norm > 0.0) ?
                                    dot3(P_sibling->h_vec, P_parent->h_vec) / (h2_norm * ho_norm) : 1.0;

                                /* Clamp to [-1, 1] */
                                if (cos_i12 < -1.0) cos_i12 = -1.0; if (cos_i12 > 1.0) cos_i12 = 1.0;
                                if (cos_i1o < -1.0) cos_i1o = -1.0; if (cos_i1o > 1.0) cos_i1o = 1.0;
                                if (cos_i2o < -1.0) cos_i2o = -1.0; if (cos_i2o > 1.0) cos_i2o = 1.0;

                                double features[11] = {qi1, qi2, qo, ali1o, ali2o, e_in, e_sib, e_out,
                                                       acos(cos_i12) / M_PI, acos(cos_i1o) / M_PI, acos(cos_i2o) / M_PI};
                                if (mlp_features_in_range_quad_2p2(features))
                                {
                                    P_unstable = mlp_predict_quad_2p2(features);
                                }
                                else
                                {
                                    mlp_out_of_range = true;
                                    #ifdef VERBOSE
                                    if (verbose_flag > 0)
                                    {
                                        printf("ODE_root_finding.cpp -- MLP 2+2 quad features out of range, using MA01 fallback\n");
                                    }
                                    #endif
                                }
                            }
                        }

                        if (mlp_out_of_range)
                        {
                            /* Fall back to MA01 criterion */
                            double q_out_ma01 = P_sibling->mass / M_p;
                            double rel_INCL_ma01 = 0.0;
                            get_inclination_relative_to_parent(particlesMap, P_p->index, &rel_INCL_ma01);
                            double rp_out_crit_ma01 = compute_rp_out_crit_MA01(a_in, q_out_ma01, e_out, rel_INCL_ma01);
                            f_root = rp_out - rp_out_crit_ma01;
                        }
                        else
                        {
                            /* f_root = 0.5 - P_unstable: positive when stable, negative when unstable */
                            f_root = MLP_INSTABILITY_ENTRY_THRESHOLD - P_unstable;
                        }
                    }
                    
                    #ifdef VERBOSE
                    if (verbose_flag > 2)
                    {
                        printf("root_finding.cpp -- check_for_roots -- check_for_dynamical_instability f_root %g\n",f_root);
                    }
                    #endif

                    if (use_root_functions == true)
                    {
                        root_functions[i_root] = f_root;
                    }
                    else
                    {
                        if (f_root <= 0.0)
                        {
                            P_p->dynamical_instability_has_occurred = true;
                        }
                        else
                        {
                            P_p->dynamical_instability_has_occurred = false;
                        }
                    }
                }
                i_root++;                
            }
            if (P_p->check_for_physical_collision_or_orbit_crossing == true)
            {

                Particle *P_child1 = (*particlesMap)[P_p->child1];
                Particle *P_child2 = (*particlesMap)[P_p->child2];
                
                double cross_section = 0.0;
                if (P_child1->is_binary == false and P_child2->is_binary == false)
                {
                    
                    cross_section_function(P_child1,&cross_section);
                    cross_section_function(P_child2,&cross_section);
                }
                else
                {
                    cross_section = epsilon;
                }

                double periapse_distance = P_p->a*(1.0 - P_p->e);
                f_root = 1.0 - periapse_distance/cross_section;
                
                #ifdef VERBOSE
                if (verbose_flag > 2)
                {
                    printf("root_finding.cpp -- check_for_roots -- check_for_physical_collision_or_orbit_crossing f_root %g P_p index %d\n",f_root,P_p->index);
                }
                #endif
                
                if (use_root_functions == true)
                {
                    root_functions[i_root] = f_root;
                }
                else
                {
                    if (f_root >= 0.0)
                    {
                        P_p->physical_collision_or_orbit_crossing_has_occurred = true;

                        #ifdef VERBOSE
                        if (verbose_flag > 0)
                        {
                            printf("root_finding.cpp -- check_for_roots -- check_for_physical_collision_or_orbit_crossing index %d a %g cross-section %g root function %Lg\n",P_p->index,P_p->a,cross_section, root_functions[i_root]);
                        }
                        #endif

                    }  
                    else
                    {
                        P_p->physical_collision_or_orbit_crossing_has_occurred = false;
                    }
                }
                
                i_root++;
            }
            if (P_p->check_for_minimum_periapse_distance == true)
            {
                #ifdef VERBOSE
                if (verbose_flag > 2)
                {
                    printf("root_finding.cpp -- check_for_roots -- check_for_minimum_periapse_distance\n");
                }
                #endif

                Particle *P_child1 = (*particlesMap)[P_p->child1];
                Particle *P_child2 = (*particlesMap)[P_p->child2];
                
                double cross_section = P_p->check_for_minimum_periapse_distance_value;
                double periapse_distance = P_p->a*(1.0 - P_p->e);
                f_root = 1.0 - periapse_distance/cross_section;
                
                if (use_root_functions == true)
                {
                    root_functions[i_root] = f_root;
                }
                else
                {
                    if (f_root >= 0.0)
                    {
                        P_p->minimum_periapse_distance_has_occurred = true;

                        #ifdef VERBOSE
                        if (verbose_flag > 0)
                        {
                            printf("root_finding.cpp -- check_for_roots -- check_for_physical_collision_or_orbit_crossing root finding check_for_minimum_periapse_distance index %d a %g cross-section %g root function %Lg\n",P_p->index,P_p->a,cross_section, root_functions[i_root]);
                        }
                        #endif
                    }                
                    else
                    {
                        P_p->minimum_periapse_distance_has_occurred = false;
                    }
                }

                i_root++;
            }
            if (P_p->check_for_GW_condition == true)
            {
                #ifdef VERBOSE
                if (verbose_flag > 2)
                {
                    printf("root_finding.cpp -- check_for_roots -- check_for_GW_condition\n");
                }
                #endif

                if (P_p->parent != -1)
                {
                    Particle *P_parent = (*particlesMap)[P_p->parent];
                    Particle *P_child1 = (*particlesMap)[P_p->child1];
                    Particle *P_child2 = (*particlesMap)[P_p->child2];
                    Particle *P_sibling = (*particlesMap)[P_p->sibling];
                
                    double m_1 = P_child1->mass;
                    double m_2 = P_child2->mass;
                    double m_3 = P_sibling->mass;
                    double M_p = P_p->mass;
                    double a_out = P_parent->a;
                    double a_out_p2 = a_out*a_out;
                    double a_out_p3 = a_out*a_out_p2;
                    double e_out = P_parent->e;
                    double e_out_p2 = e_out*e_out;
                    double a_in = P_p->a;
                    double a_in_p2 = a_in*a_in;
                    double a_in_p3 = a_in*a_in_p2;
                    double a_in_p4 = a_in_p2*a_in_p2;
                    double e_in = P_p->e;
                    double e_in_p2 = e_in*e_in;
                    double e_in_p4 = e_in_p2*e_in_p2;
                    double j_in = sqrt(1.0-e_in_p2);
                    double j_in_p2 = j_in*j_in;
                    double j_in_p4 = j_in_p2*j_in_p2;
                    double j_in_p5 = j_in*j_in_p4;
                    double j_in_p6 = j_in_p2*j_in_p4;
                    double j_in_p7 = j_in*j_in_p6;

                    double t_GW_a = 1.0/( c_64div5*m_1*m_2*M_p*CONST_G_P3*(1.0 + c_73div24*e_in_p2 + c_37div96*e_in_p4)/(CONST_C_LIGHT_P5*a_in_p4*j_in_p7) );
                    double t_LK_rp = 1.0 / ( (75.0/64.0)*sqrt(5.0/3.0)*(e_in/j_in)*sqrt(CONST_G*M_p/a_in_p3)*(m_3/M_p)*(a_in_p3/a_out_p3)*pow(1.0-e_out_p2,-3.0/2.0) );
                    
                    f_root = 1.0 - t_LK_rp/(10.0*t_GW_a);

                    if (use_root_functions == true)
                    {
                        root_functions[i_root] = f_root;
                    }
                    else
                    {
                        if (f_root >= 0.0)
                        {
                            P_p->GW_condition_has_occurred = true;
                        }
                        else
                        {
                            P_p->GW_condition_has_occurred = false;
                        }
                    }   
                }

                i_root++;
            }
            if (P_p->check_for_entering_LISA_band == true)
            {
                #ifdef VERBOSE
                if (verbose_flag > 2)
                {
                    printf("root_finding.cpp -- check_for_roots -- check_for_entering_LISA_band\n");
                }
                #endif

                
                {
                
                    double M = P_p->mass;
                    double a = P_p->a;
                    double e = P_p->e;
                    double e_p2 = e*e;
                    double e_p3 = e*e_p2;
                    double e_p4 = e_p2*e_p2;
                    
                    double f_orb = 1.0/compute_orbital_period_from_semimajor_axis(M,a);

                    double n_peak = 2.0*( 1.0 - 1.01678*e + 5.57372*e_p2 - 4.9271*e_p3 + 1.68506*e_p4 ) * pow(1.0 - e_p2,-1.5);
                    double f_peak = f_orb * n_peak;

                    f_root = 1.0 - f_peak/P_p->check_for_entering_LISA_band_critical_GW_frequency;

                    if (use_root_functions == true)
                    {
                        root_functions[i_root] = f_root;
                    }
                    else
                    {
                        if (f_root <= 0.0 and P_p->check_for_entering_LISA_band_occurrences == 0)
                        {
                            P_p->entering_LISA_band_has_occurred = true;
                        }
                        else
                        {
                            P_p->entering_LISA_band_has_occurred = false;
                        }
                    }   
                }

                i_root++;
            }
        }
        else /* P_p not a binary */
        {

            if (P_p->check_for_RLOF_at_pericentre == true)
            {
                #ifdef VERBOSE
                if (verbose_flag > 2)
                {
                    printf("root_finding.cpp -- check_for_roots -- check_for_RLOF_at_pericentre P_p %d\n",P_p->index);
                }
                #endif
                
                if (P_p->parent != -1)
                {
                    Particle *P_parent = (*particlesMap)[P_p->parent];
                    Particle *P_sibling = (*particlesMap)[P_p->sibling];
                    
                    double a = P_parent->a;
                    double e = P_parent->e;
                    double rp = a*(1.0 - e);
                    double subject_mass = P_p->mass;
                    double companion_mass = P_sibling->mass;
    
                    double spin_angular_frequency = P_p->spin_vec_norm;
                    double orbital_angular_frequency_periapse = sqrt( CONST_G*(subject_mass + companion_mass)*(1.0 + e)/(rp*rp*rp) );
                    double f = spin_angular_frequency/orbital_angular_frequency_periapse;
            
                    double roche_radius_pericenter;
                    if (P_p->check_for_RLOF_at_pericentre_use_sepinsky_fit == false)
                    {
                        roche_radius_pericenter = roche_radius_pericenter_eggleton(rp, subject_mass/companion_mass);
                    }
                    else
                    {
                        roche_radius_pericenter = roche_radius_pericenter_sepinsky(rp, subject_mass/companion_mass, e, f);
                    }
                    
                    f_root = 1.0 - P_p->radius/roche_radius_pericenter;

                    #ifdef VERBOSE
                    if (verbose_flag > 2)
                    {
                        printf("root_finding.cpp -- check_for_roots -- check_for_RLOF_at_pericentre id %d rp %g roche_radius_pericenter %g R %g\n", P_p->index,rp, roche_radius_pericenter, P_p->radius);
                    }
                    #endif

                    if (use_root_functions == true)
                    {
                        root_functions[i_root] = f_root;
                    }
                    else
                    {
                        if (f_root <= 0.0)
                        {
                            P_p->RLOF_at_pericentre_has_occurred = true;
                        }
                        else
                        {
                            P_p->RLOF_at_pericentre_has_occurred = false;
                            P_p->RLOF_flag = 0;
                        }
                    }
                }
                
                i_root++;
                
            }
        }
    }
}

int investigate_roots_in_system(ParticlesMap *particlesMap, double t, int integration_flag)
{
    int return_flag = 0;
    bool RLOF_occurred = false;
    bool collision_occurred = false;
    ParticlesMapIterator it_p;
    for (it_p = particlesMap->begin(); it_p != particlesMap->end(); it_p++)
    {
        Particle *p = (*it_p).second;
        if (p->is_binary == false)
        {
            if (p->RLOF_at_pericentre_has_occurred == true)
            {
                p->RLOF_at_pericentre_has_occurred = false; /* Reset ODE root found flag */
                
                if (p->RLOF_at_pericentre_has_occurred_entering_RLOF == true) /* Going into RLOF */
                {
                    p->RLOF_flag = 1;
                }
                else if (p->RLOF_at_pericentre_has_occurred_entering_RLOF == false) /* Going out of RLOF */
                {
                    p->RLOF_flag = 0;
                }
                
                #ifdef LOGGING
                Log_info_type log_info;
                log_info.binary_index = p->parent;
                log_info.index1 = p->index;
                log_info.index2 = p->sibling;
                int event_flag;
                if (p->RLOF_flag == 1)
                {
                    event_flag = LOG_MT_START;
                }
                else
                {
                    event_flag = LOG_MT_END;
                }
                
                Log_type &last_entry = logData.back();
                Log_info_type &last_log_info = last_entry.log_info;
                
                if ((last_entry.event_flag == LOG_MT_START or last_entry.event_flag == LOG_MT_END) and (last_log_info.index1 == p->index) and (last_log_info.index2 == p->sibling) and (last_log_info.binary_index == p->parent))
                {
                    //printf("Skipping log entry\n");
                }
                else
                {
                    update_log_data(particlesMap, t, integration_flag, event_flag, log_info);
                }
                #endif

                #ifdef VERBOSE
                if (verbose_flag > 0)
                {
                    printf("root_finding.cpp -- RLOF true for body %d; p->check_for_RLOF_at_pericentre %d; p->RLOF_flag %d\n",p->index,p->check_for_RLOF_at_pericentre,p->RLOF_flag);
                }
                #endif

                RLOF_occurred = true;
                return_flag = 1;
            }
            else
            {
                p->RLOF_flag = 0;
            }
                
        }
        else // binary
        {
            if (p->dynamical_instability_has_occurred == true)
            {
                p->dynamical_instability_has_occurred = false;
                return_flag = 2;

                #ifdef LOGGING
                Log_info_type log_info;
                log_info.binary_index = p->index;
                update_log_data(particlesMap, t, integration_flag, LOG_DYN_INST, log_info);
                #endif
            }
            if (p->secular_breakdown_has_occurred == true)
            {
                p->secular_breakdown_has_occurred = false;
                return_flag = 3;

                #ifdef LOGGING
                Log_info_type log_info;
                log_info.binary_index = p->index;
                update_log_data(particlesMap, t, integration_flag, LOG_SEC_BREAK, log_info);
                #endif

            }
            if (p->physical_collision_or_orbit_crossing_has_occurred == true)
            {
                p->physical_collision_or_orbit_crossing_has_occurred = false;
                Particle *child1 = (*particlesMap)[p->child1];
                Particle *child2 = (*particlesMap)[p->child2];

                if (child1->is_binary == false and child2->is_binary == false) // only consider collision/merger between two bodies; TO DO: what to do in other cases (can they occur)?
                {
                    p->merged = true;
                    return_flag = 4;
                    collision_occurred = true;
                }
            }
            if (p->entering_LISA_band_has_occurred == true)
            /*if (p->entering_LISA_band_has_occurred == true and return_flag == 0) */
            {
                
                p->entering_LISA_band_has_occurred = false;
                p->check_for_entering_LISA_band_occurrences++;

                if (return_flag ==0)
                {
                return_flag = 5;
                
                #ifdef LOGGING
                Log_info_type log_info;
                log_info.binary_index = p->index;
                update_log_data(particlesMap, t, integration_flag, LOG_ENTER_LISA_BAND, log_info);
                
                #endif
                }


            }
            
        }
        
    }
    
    if (RLOF_occurred == true and collision_occurred == true) // In case of multiple roots, give preference to collision over RLOF
    {
        return_flag = 4;
    }
    
    return return_flag;
}

void cross_section_function(Particle *p, double *cross_section)
{ 
    if (p->is_binary==false)
    {
        *cross_section += determine_effective_radius_for_collision(p->radius, p->stellar_type, 0);
    }
    //else
    //{
    //    *cross_section += p->a*(1.0 + p->e);
    //}
}
double compute_AM_time_scale(Particle *P_p)
{
    double e = P_p->e;
    double e_p2 = P_p->e_p2;
    double de_dt = dot3(P_p->e_vec_unit,P_p->de_vec_dt);    
    double AM_time_scale = ((1.0-e_p2)/(e*fabs(de_dt)));
    
    return AM_time_scale;
}
         

double roche_radius_pericenter_eggleton(double rp, double q)
{
    /* 2007ApJ...660.1624S Eqs. (45) */    
    /* q is defined as m_primary/m_secondary */
    double q_pow_one_third = pow(q,c_1div3);
    double q_pow_two_third = q_pow_one_third*q_pow_one_third;
    return rp*0.49*q_pow_two_third/(0.6*q_pow_two_third + log(1.0 + q_pow_one_third));
}
double roche_radius_pericenter_sepinsky(double rp, double q, double e, double f)
{
    /* 2007ApJ...660.1624S Eqs. (47)-(52) */
    double log_q = log10(q);
    double A = f*f*(1.0 + e); // assumes pericenter
    double log_A = log10(A);

    double R_L_pericenter_eggleton = roche_radius_pericenter_eggleton(rp,q);
    double ratio = 0.0; // this is R_L divided by R_L_pericenter_eggleton

    if (log_q < 0.0)
    {
        if (log_A <= -0.1)
        {
            double c = 0.5*(1.0+A) + log_q;
            ratio = 1.0 + 0.11*(1.0-A) - 0.05*(1.0-A)*exp(-c*c);
        }
        if ((log_A > -0.1) && (log_A < 0.2))
        {
            double g_0 = 0.9978 - 0.1229*log_A - 0.1273*log_A*log_A;
            double g_1 = 0.001 + 0.02556*log_A;
            double g_2 = 0.0004 + 0.0021*log_A;
            ratio = g_0 + g_1*log_q + g_2*log_q*log_q;
        }
        if (log_A >= 0.2)
        {
            double num_0 = 6.3014*pow(log_A,1.3643);
            double den_0 = exp(2.3644*pow(log_A,0.70748)) - 1.4413*exp(-0.0000184*pow(log_A,-4.5693));
            double i_0 = num_0/den_0;

            double den_1 = 0.0015*exp(8.84*pow(log_A,0.282)) + 15.78;
            double i_1 = log_A/den_1;

            double num_2 = 1.0 + 0.036*exp(8.01*pow(log_A,0.879));
            double den_2 = 0.105*exp(7.91*pow(log_A,0.879));
            double i_2 = num_2/den_2;

            double den_3 = 1.38*exp(-0.035*pow(log_A,0.76)) + 23.0*exp(-2.89*pow(log_A,0.76));
            double i_3 = 0.991/den_3;

            double c = log_q + i_3;
            ratio = i_0 + i_1*exp(-i_2*c*c);
        }
    }
    if (log_q >= 0.0)
    {
        if (log_A <= -0.1)
        {
            ratio = 1.226 - 0.21*A - 0.15*(1.0-A)*exp( (0.25*A - 0.3)*pow(log_q,1.55) );
        }
        if ((log_A > -0.1) && (log_A < 0.2))
        {
            double log_A_p2 = log_A*log_A;
            double h_0 = 1.0071 - 0.0907*log_A - 0.0495*log_A_p2;
            double h_1 = -0.004 - 0.163*log_A - 0.214*log_A_p2;
            double h_2 = 0.00022 - 0.0108*log_A - 0.02718*log_A_p2;
            ratio = h_0 + h_1*log_q + h_2*log_q*log_q;
        }
        if (log_A >= 0.2)
        {
            double num_0 = 1.895*pow(log_A,0.837);
            double den_0 = exp(1.636*pow(log_A,0.789)) - 1.0;
            double j_0 = num_0/den_0;

            double num_1 = 4.3*pow(log_A,0.98);
            double den_1 = exp(2.5*pow(log_A,0.66)) + 4.7;
            double j_1 = num_1/den_1;

            double den_2 = 8.8*exp(-2.95*pow(log_A,0.76)) + 1.64*exp(-0.03*pow(log_A,0.76));
            double j_2 = 1.0/den_2;

//            double j_3 = 0.256*exp(-1.33*pow(log_A,2.9))*( 5.5*exp(1.33*pow(log_A,2.9)) + 1.0 );
            double j_3 = 0.256*(5.5 + exp(-1.33*pow(log_A,2.9)));

            ratio = j_0 + j_1*exp(-j_2*pow(log_q,j_3));

            #ifdef VERBOSE
            if (verbose_flag > 0)
            {
                printf("ODE_root_finding.cpp -- roche_radius_pericenter_sepinsky\n");
                printf("log_A %g\n",log_A);
                printf("1 %g %g %g \n",num_0,den_0,j_0);
                printf("2 %g %g %g \n",num_1,den_1,j_1);            
                printf("2 %g %g %g \n",den_2,j_2,j_3);            
                printf("ratio %g\n",ratio);            
            }
            #endif
        }
    }

    if (ratio == 0.0)
    {
        printf("ODE_root_finding.cpp -- unrecoverable error occurred in function roche_radius_pericenter_sepinsky\n");
        printf("log_q %g log_A %g ratio %g\n",log_q,log_A,ratio);
        printf("rp %g q %g e %g f %g\n",rp,q,e,f);
        //exit(-1);
        error_code = 23;
        longjmp(jump_buf,1);
    }
    
    return ratio*R_L_pericenter_eggleton;
}

int read_root_finding_data(ParticlesMap *particlesMap, int *roots_found)
{
    ParticlesMapIterator it_p;
    
    int i_root = 0;
    
    for (it_p = particlesMap->begin(); it_p != particlesMap->end(); it_p++)
    {
        Particle *P_p = (*it_p).second;
        if (P_p->is_binary == true)
        {
            if (P_p->check_for_secular_breakdown == true)
            {
                if FOUND_ROOT
                {
                    P_p->secular_breakdown_has_occurred = true;
                    
                }
                i_root++;
            }
            if (P_p->check_for_dynamical_instability == true)
            {
                if FOUND_ROOT
                {
                    P_p->dynamical_instability_has_occurred = true;
                }
                i_root++;                
            }
            if (P_p->check_for_physical_collision_or_orbit_crossing == true)
            {
                if FOUND_ROOT
                {
                    P_p->physical_collision_or_orbit_crossing_has_occurred = true;
                }
                i_root++;     
            }
            if (P_p->check_for_minimum_periapse_distance == true)
            {
                if FOUND_ROOT
                {
                    P_p->minimum_periapse_distance_has_occurred = true;
                }
                i_root++;                
            }
            if (P_p->check_for_GW_condition == true)
            {
                if FOUND_ROOT
                {
                    P_p->GW_condition_has_occurred = true;
                }
                i_root++;                
            }
            if (P_p->check_for_entering_LISA_band == true)
            {
                if FOUND_ROOT
                {
                    P_p->entering_LISA_band_has_occurred = true;
                }
                i_root++;                
            }
        }
        else /* P_p not a binary */
        {
            if (P_p->check_for_RLOF_at_pericentre == true)
            {
                if FOUND_ROOT
                {
                    P_p->RLOF_at_pericentre_has_occurred = true;
                }
                if (roots_found[i_root] == 1) /* Going out of RLOF (froot passed 0 and increased from negative (RLOF) to positive (no RLOF)) */
                {
                    P_p->RLOF_at_pericentre_has_occurred_entering_RLOF = false;
                }
                else if (roots_found[i_root] == -1) /* Going into RLOF (froot passed 0 and decreased from positive (no RLOF) to negative (RLOF)) */
                {
                    P_p->RLOF_at_pericentre_has_occurred_entering_RLOF = true;
                }
                i_root++;
            }
        }
    }
    return 0;
}

int check_for_initial_roots(ParticlesMap *particlesMap)
{
    int N_bodies, N_binaries,N_root_finding,N_ODE_equations;
    determine_binary_parents_and_levels(particlesMap,&N_bodies,&N_binaries,&N_root_finding,&N_ODE_equations);
    
    #ifdef VERBOSE
    if (verbose_flag > 2)
    {
        printf("ODE_root_finding.cpp -- check_for_initial_roots -- start\n");
        print_system(particlesMap,0);
    }
    #endif
    
    realtype root_functions[N_root_finding];
    check_for_roots(particlesMap, false, root_functions);
    
    int N_root_found = 0;

    ParticlesMapIterator it_p;
    
    for (it_p = particlesMap->begin(); it_p != particlesMap->end(); it_p++)
    {
        Particle *P_p = (*it_p).second;

        if (P_p->is_binary == true)
        {
            if (P_p->check_for_secular_breakdown == true)
            {
                if (P_p->secular_breakdown_has_occurred == true)
                {
                    N_root_found++;
                    
                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d secular breakdown \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }

            }
            if (P_p->check_for_dynamical_instability == true)
            {
                if (P_p->dynamical_instability_has_occurred == true)
                {
                    N_root_found++;
                    
                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d dynamical instability \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }
            }
            if (P_p->check_for_physical_collision_or_orbit_crossing == true)
            {
                if (P_p->physical_collision_or_orbit_crossing_has_occurred == true)
                {
                    N_root_found++;

                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d collision \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }
            }
            if (P_p->check_for_minimum_periapse_distance == true)
            {
                if (P_p->minimum_periapse_distance_has_occurred == true)
                {
                    N_root_found++;

                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d minimum peri \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }
            }            
            if (P_p->check_for_GW_condition == true)
            {
                if (P_p->GW_condition_has_occurred == true)
                {
                    N_root_found++;

                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d GW condition \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }
            }   
            if (P_p->check_for_entering_LISA_band == true)
            {
                if (P_p->entering_LISA_band_has_occurred == true)
                {
                    N_root_found++; 

                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d LISA band \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }
            } 
        }
        else /* P_p not a binary */
        {
            if (P_p->check_for_RLOF_at_pericentre == true)
            {
                if (P_p->RLOF_at_pericentre_has_occurred == true)
                {
                    #ifdef VERBOSE
                    if (verbose_flag > 0)
                    {
                        printf("ODE_root_finding.cpp -- check_for_initial_roots -- p %d RLOF but continuing \n",P_p->index);
                        print_system(particlesMap,0);
                    }
                    #endif
                }
            }
        }
    }
    
    #ifdef VERBOSE
    if (verbose_flag > 2)
    {
        printf("ODE_root_finding.cpp -- check_for_initial_roots -- done\n");
        print_system(particlesMap,0);
    }
    #endif
    return N_root_found;
}

void handle_roots(ParticlesMap *particlesMap, int root_flag, int *integration_flag, int *CVODE_flag, double t, double *dt_stev, double *dt_binary_evolution)
{
    if (root_flag == 1) // RLOF -- continue secular but with mass transfer terms
    {
        #ifdef VERBOSE
        if (verbose_flag > 0)
        {
            printf("root_finding.cpp -- handle_roots -- RLOF occurred\n");
        }
        #endif
        
        *CVODE_flag = 0;
    }
    else if (root_flag == 2) // Dynamical instability -- switch to direct N-body
    {
        #ifdef VERBOSE
        if (verbose_flag > 0)
        {
            printf("root_finding.cpp -- handle_roots -- Dynamical instability\n");
        }
        #endif

        *integration_flag = 1;
        *CVODE_flag = 0;
    }
    else if (root_flag == 3) // Semisecular -- switch to direct N-body
    {
        #ifdef VERBOSE
        if (verbose_flag > 0)
        {
            printf("root_finding.cpp -- handle_roots -- Semisecular\n");
        }
        #endif
       
        *integration_flag = 2;
        *CVODE_flag = 0;
    }
    else if (root_flag == 4) // Collision/merger
    {
        #ifdef VERBOSE
        if (verbose_flag > 0)
        {
            printf("root_finding.cpp -- handle_roots -- Collision/merger\n");
        }
        #endif

        *CVODE_flag = 0;
        handle_collisions(particlesMap, t, integration_flag);
    }
    else if (root_flag == 5) // Entered LISA band
    {
        #ifdef VERBOSE
        if (verbose_flag > 0)
        {
            printf("root_finding.cpp -- handle_roots -- Entered LISA band\n");
        }
        #endif

        *CVODE_flag = 0;
    }
    else
    {
        printf("root_flag %d\n",root_flag);
        #ifdef VERBOSE
        if (verbose_flag > 0)
        {
            printf("root_finding.cpp -- handle_roots -- Other\n");
            printf("ODE_root_finding.cpp -- handle_roots -- other root condition -- ERROR \n");
            print_system(particlesMap,0);
        }
        #endif

        error_code = 24;
        longjmp(jump_buf,1);
    }
}

}
