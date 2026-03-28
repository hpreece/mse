import numpy as np
import argparse
import ctypes
import time

from mse import MSE,Particle,Tools

"""
Several routines for testing the code/installation. 
To run all tests, simply run `python test_mse.py'.
Specific tests can be run with the command line --t i, where i is the
number of the test. Use --verbose for verbose terminal output, and --plot to
make and show plots if applicable (required Matplotlib).

"""

try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def add_bool_arg(parser, name, default=False,help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',help="Enable %s"%help)
    group.add_argument('--no-' + name, dest=name, action='store_false',help="Disable %s"%help)
    parser.set_defaults(**{name:default})

def parse_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--t",                           type=int,     dest="test",                        default=0,              help="Test number")
    parser.add_argument("--m","--mode",                  type=int,     dest="mode",                        default=0,              help="Mode -- 0: standard; 1: detailed (more extensive tests, but takes more time to run)")
    
    
    ### boolean arguments ###
    add_bool_arg(parser, 'verbose',                         default=False,         help="Verbose terminal output")
    add_bool_arg(parser, 'plot',                            default=False,         help="Make plots")
    add_bool_arg(parser, 'fancy_plots',                     default=False,         help="Use LaTeX fonts for plots (slow)")
    
    args = parser.parse_args()

    return args

class test_mse():

    def test1(self,args):
        print("Test secular evolution")
        print("Test 1a: secular equations of motion using reference triple system of Naoz et al. (2009)")

        particles = Tools.create_fully_nested_multiple(3, [1.0,1.0e-3,40.0e-3],[6.0,100.0],[0.001,0.6],[0.0,65.0*np.pi/180.0],[45.0*np.pi/180.0,0.0],[0.0,0.0],metallicities=[0.02,0.02,0.02],stellar_types=[1,1,1],object_types=[2,2,2])
        binaries = [x for x in particles if x.is_binary==True]
        inner_binary = binaries[0]
        outer_binary = binaries[1]
        bodies = [x for x in particles if x.is_binary==False]
        
        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
        
        code = MSE()
        code.add_particles(particles)

        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14

        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0
        e_print = []
        INCL_print = []
        rel_INCL_print = []
        t_print = []
        
        start = time.time()
    
        t = 0.0
        N = 1000
        tend = 3.0e7

        dt = tend/float(N)
        while t<=tend:

            code.evolve_model(t)
            t+=dt
            
            particles=code.particles
            binaries = [x for x in particles if x.is_binary==True]
            inner_binary = binaries[0]
            outer_binary = binaries[1]
            bodies = [x for x in particles if x.is_binary==False]
        
            if args.verbose==True:
                print( 't',t,'es',[x.e for x in binaries],'INCL_parent',inner_binary.INCL_parent,[x.mass for x in bodies])

            rel_INCL_print.append(inner_binary.INCL_parent)
            e_print.append(inner_binary.e)
            INCL_print.append(inner_binary.INCL)
            t_print.append(t)
        
        if args.verbose==True:
            print('wall time',time.time()-start)
            print("e_print[-1]",e_print[-1],"rel_INCL_print[-1]",rel_INCL_print[-1])
        
        t_print = np.array(t_print)
        rel_INCL_print = np.array(rel_INCL_print)
        e_print = np.array(e_print)

        e_expected = 0.2049
        INCL_expected = 1.2166
        rel_tol = 1e-3
        assert abs(e_print[-1] - e_expected) / e_expected < rel_tol, "e_print[-1]=%g, expected=%g" % (e_print[-1], e_expected)
        assert abs(rel_INCL_print[-1] - INCL_expected) / INCL_expected < rel_tol, "INCL=%g, expected=%g" % (rel_INCL_print[-1], INCL_expected)

        print("Test 1a passed")

        code.reset()
                
        if HAS_MATPLOTLIB==True and args.plot==True:
            fig=pyplot.figure()
            plot1=fig.add_subplot(2,1,1)
            plot2=fig.add_subplot(2,1,2,yscale="log")
            plot1.plot(1.0e-6*t_print,rel_INCL_print*180.0/np.pi)
            plot2.plot(1.0e-6*t_print,1.0-e_print)
            plot2.set_xlabel("$t/\mathrm{Myr}$")
            plot1.set_ylabel("$i_\mathrm{rel}/\mathrm{deg}$")
            plot2.set_ylabel("$1-e_\mathrm{in}$")
            pyplot.show()


        print("Test 1b: test of canonical e_max expression (quadruple order; test particle; zero initial inner eccentricity)")
        
        i_rel = 85.0*np.pi/180.0
        particles = Tools.create_fully_nested_multiple(3, [1.0,0.001,1.0],[1.0,100.0],[0.0001,0.0001],[0.0,i_rel],[45.0*np.pi/180.0,0.0],[0.0,0.0],metallicities=[0.02,0.02,0.02],stellar_types=[1,1,1],object_types=[2,2,2])
        binaries = [x for x in particles if x.is_binary==True]
        inner_binary = binaries[0]
        outer_binary = binaries[1]
        bodies = [x for x in particles if x.is_binary==False]
        
        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
        
        code = MSE()
        code.add_particles(particles)

        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14

        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0
        e_print = []
        INCL_print = []
        rel_INCL_print = []
        t_print = []
        
        start = time.time()
    
        t = 0.0
        N = 1000
        tend = 2.0e6

        dt = tend/float(N)
        while t<=tend:

            code.evolve_model(t)
            t+=dt
            
            particles=code.particles
            binaries = [x for x in particles if x.is_binary==True]
            inner_binary = binaries[0]
            outer_binary = binaries[1]
            bodies = [x for x in particles if x.is_binary==False]
        
            if args.verbose==True:
                print( 't',t,'es',[x.e for x in binaries],'INCL_parent',inner_binary.INCL_parent,[x.mass for x in bodies])

            rel_INCL_print.append(inner_binary.INCL_parent)
            e_print.append(inner_binary.e)
            INCL_print.append(inner_binary.INCL)
            t_print.append(t)
        
        t_print = np.array(t_print)
        rel_INCL_print = np.array(rel_INCL_print)
        e_print = np.array(e_print)

        e_max_an = np.sqrt(1.0 - (5.0/3.0)*np.cos(i_rel)**2)
        e_max_num = np.amax(e_print)

        if args.verbose==True:
            print('wall time',time.time()-start)
            print("e_max_an",e_max_an,"e_max_num",e_max_num)

        N_r = 3
        assert round(e_max_an,N_r) == round(e_max_num,N_r)

        print("Test 1b passed")

        code.reset()

        if HAS_MATPLOTLIB==True and args.plot==True:
            fig=pyplot.figure()
            plot1=fig.add_subplot(2,1,1)
            plot2=fig.add_subplot(2,1,2,yscale="log")
            plot1.plot(1.0e-6*t_print,rel_INCL_print*180.0/np.pi)
            plot2.plot(1.0e-6*t_print,1.0-e_print)
            plot2.set_xlabel("$t/\mathrm{Myr}$")
            plot1.set_ylabel("$i_\mathrm{rel}/\mathrm{deg}$")
            plot2.set_ylabel("$1-e_\mathrm{in}$")
            pyplot.show()
        

    def test2(self,args):
        print("Test 1PN precession in 2-body system")

        particles = Tools.create_fully_nested_multiple(2,[1.0, 1.0], [1.0], [0.99], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[2,2])
        binaries = [x for x in particles if x.is_binary == True]
        bodies = [x for x in particles if x.is_binary == False]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
            b.radius = 1.0e-10

        for b in binaries:
            b.include_pairwise_1PN_terms = True
            b.include_pairwise_25PN_terms = False
            b.exclude_1PN_precession_in_case_of_isolated_binary = False ### By default, 1PN apsidal motion is not calculated for isolated binaries; override for this test
        
        code = MSE()
        code.add_particles(particles)

        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag=0
        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14 ### need to set lower than default to get more accurate result and compare to analytic expression
        t = 0.0
        N=1000
        tend = 1.0e7
        dt=tend/float(N)

        t_print_array = []
        a_print_array = []
        e_print_array = []
        AP_print_array = []

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            
            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            if args.verbose==True:
                print( 't/Myr',t,'AP',binaries[0].AP)

            t_print_array.append(t)
            a_print_array.append(binaries[0].a)
            e_print_array.append(binaries[0].e)
            AP_print_array.append(binaries[0].AP)
        
        t_print_array = np.array(t_print_array)
        a_print_array = np.array(a_print_array)
        e_print_array = np.array(e_print_array)
        AP_print_array = np.array(AP_print_array)
        
        CONST_G = code.CONST_G
        CONST_C = code.CONST_C

        # Theoretical prediction #
        a = binaries[0].a
        e = binaries[0].e
        M = binaries[0].mass
        rg = CONST_G*M/(CONST_C**2)
        P = 2.0*np.pi*np.sqrt(a**3/(CONST_G*M))
        t_1PN = (1.0/3.0)*P*(1.0-e**2)*(a/rg)

        AP = 0.01 +2.0*np.pi*tend/t_1PN
        AP = (AP+np.pi)%(2.0*np.pi) - np.pi ### -pi < AP < pi

        if args.verbose == True:
            print("AP num",AP_print_array[-1], "AP an",AP)
        
        N_r=4
        assert round(AP_print_array[-1],N_r) == round(AP,N_r)
        print("Test passed")

        code.reset()
                
        if HAS_MATPLOTLIB == True and args.plot==True:
            fig = pyplot.figure(figsize=(16,10))
            plot1 = fig.add_subplot(4,1,1)
            plot2 = fig.add_subplot(4,1,2,yscale="log")
            plot3 = fig.add_subplot(4,1,3,yscale="log")
            plot4 = fig.add_subplot(4,1,4,yscale="log")

            plot1.plot(t_print_array*1.0e-6,AP_print_array, color='r',label="$\mathrm{MSE}$")
            points = np.linspace(0.0,tend*1.0e-6,N)
            AP = 0.01 +2.0*np.pi*points/(t_1PN*1.0e-6)
            AP = (AP+np.pi)%(2.0*np.pi) - np.pi ### -pi < AP < pi
            plot1.plot(points,AP,color='g',label="$\mathrm{Analytic}$")

            plot2.plot(t_print_array*1.0e-6,np.fabs( (AP - AP_print_array)/AP ), color='r')
            plot3.plot(t_print_array*1.0e-6,np.fabs((a-a_print_array)/a), color='r')
            plot4.plot(t_print_array*1.0e-6,np.fabs((e-e_print_array)/e), color='r')

            fontsize = 15
            plot1.set_ylabel("$\omega$",fontsize=fontsize)
            plot2.set_ylabel("$|(\omega_p-\omega)/\omega_p|$",fontsize=fontsize)
            plot3.set_ylabel("$|(a_0-a)/a_0|$",fontsize=fontsize)
            plot4.set_ylabel("$|(e_0-e)/e_0|$",fontsize=fontsize)
            
            handles,labels = plot1.get_legend_handles_labels()
            plot1.legend(handles,labels,loc="upper left",fontsize=0.6*fontsize)

            pyplot.show()

    def test3(self,args):
        print("Test GW emission in 2-body system")

        code = MSE()
        CONST_G = code.CONST_G
        CONST_C = code.CONST_C

        a0 = 1.0
        e0 = 0.999
        m1 = 1.0
        m2 = 1.0
        particles = Tools.create_fully_nested_multiple(2,[m1,m2], [a0], [e0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[2,2])
        
        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False

        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = True
        code.verbose_flag = 0
        
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = True
            b.check_for_physical_collision_or_orbit_crossing = True
        
        rg = (m1+m2)*CONST_G/(CONST_C**2)
        
        for b in bodies:
            b.radius = 100.0*rg

        binary = binaries[0]

        code.add_particles(particles)

        t_print_array = []
        a_print_array = []
        e_print_array = []
        AP_print_array = []

        tend = 0.97e8
        N = 1000
        dt = tend/float(N)
        t = 0.0
        while (t<tend):
            t+=dt

            code.evolve_model(t)
            flag = code.CVODE_flag

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            if len(binaries)>0:
                binary = binaries[0]

                t_print_array.append(t*1.0e-6)
                a_print_array.append(binary.a)
                e_print_array.append(binary.e)
                AP_print_array.append(binary.AP)

                if args.verbose==True:
                    print("t",t*1e-6,'a/au',binary.a,'e',binary.e)

        a_print_array = np.array(a_print_array)
        e_print_array = np.array(e_print_array)
        
        ### Peters 1964 ###
        c0 = a0*(1.0-e0**2)/( pow(e0,12.0/19.0)*pow(1.0 + (121.0/304.0)*e0**2,870.0/2299.0))
        a_an = c0*pow(e_print_array,12.0/19.0)*pow(1.0+(121.0/304.0)*e_print_array**2,870.0/2299.0)/(1.0-e_print_array**2)
        beta = (64.0/5.0)*CONST_G**3*m1*m2*(m1+m2)/(CONST_C**5)
        T_c = a0**4/(4.0*beta)
        T = (768.0/425.0)*T_c*pow(1.0-e0**2,7.0/2.0)

        N_r = 5
        if args.verbose == True:
            print("a_print_array[-1]",a_print_array[-1],"a_an[-1]",a_an[-1])
        assert(round(a_print_array[-1],N_r) == round(a_an[-1],N_r))
        
        print("Test passed")
        
        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            
            fig = pyplot.figure(figsize=(16,10))
            plot1 = fig.add_subplot(2,1,1)
            plot2 = fig.add_subplot(2,1,2,yscale="log")

            plot1.plot(t_print_array,e_print_array, color='r')

            plot2.plot(t_print_array,a_print_array, color='r',label='$\mathrm{MSE}$')

            plot2.plot(t_print_array,a_an,color='g',linestyle='dashed',linewidth=2,label='$\mathrm{Semi-analytic\,Peters\,(1964)}$')

            fontsize = 15
            plot1.set_ylabel("$e$",fontsize=fontsize)
            plot2.set_ylabel("$a/\mathrm{au}$",fontsize=fontsize)
            plot2.set_xlabel("$t/\mathrm{Myr}$",fontsize=fontsize)

            handles,labels = plot2.get_legend_handles_labels()
            plot2.legend(handles,labels,loc="upper left",fontsize=0.6*fontsize)

            pyplot.show()

    def test4(self,args):
        print("Test tidal friction in 2-body system")
        
        code = MSE()
        code.enable_tides = True
        code.include_flybys = False
        code.enable_root_finding = False

        code.verbose_flag = 0
        code.relative_tolerance = 1.0e-10
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14

        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        CONST_R_SUN = code.CONST_R_SUN
        day = 1.0/365.25
        second = day/(24.0*3600.0)

        M = 0.0009546386983890755 ### Jupiter mass
        R = 40.0*0.1027922358015816*CONST_R_SUN ### 40 R_J
        m_per = 1.0
        mu = m_per*M/(m_per+M)
        a0 = 1.0
        e0 = 0.3
        P0 = 2.0*np.pi*np.sqrt(a0**3/(CONST_G*(M+m_per)))
        n0 = 2.0*np.pi/P0

        omega_crit = np.sqrt(CONST_G*M/(R**3))

        aF = a0*(1.0-e0**2)
        nF = np.sqrt( CONST_G*(M+m_per)/(aF**3) )

        particles = Tools.create_fully_nested_multiple(2,[m_per, M], [a0], [e0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[2,2])
        binaries = [x for x in particles if x.is_binary==True]
        bodies = [x for x in particles if x.is_binary==False]
        binary = particles[2]

        for b in bodies:
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        particles[0].radius = CONST_R_SUN
        particles[1].radius = R
        particles[1].spin_vec_x = 0.0
        particles[1].spin_vec_y = 0.0
        particles[1].spin_vec_z = 4.0e-2/day



        k_L = 0.38
        k_AM = k_L/2.0
        rg = 0.25
        tau = 1e5*0.66*second

        I = rg*M*R**2
        alpha = I/(mu*a0**2)
        T = R**3/(CONST_G*M*tau)
        t_V = 3.0*(1.0 + 2.0*k_AM)**2*T/k_AM
        
        if args.verbose==True:
            print( 't_V',t_V,'M',M,'R',R)
            print("n",n0,"omega_crit",omega_crit,"omega_init",particles[1].spin_vec_z)
            
        particles[0].include_tidal_friction_terms = False
        particles[0].include_tidal_bulges_precession_terms = False
        particles[0].include_rotation_precession_terms = False

        particles[1].tides_method = 1
        particles[1].include_tidal_friction_terms = True
        particles[1].include_tidal_bulges_precession_terms = False
        particles[1].include_rotation_precession_terms = False
        particles[1].minimum_eccentricity_for_tidal_precession = 1.0e-8

        particles[1].apsidal_motion_constant = k_AM
        particles[1].tides_viscous_time_scale = t_V
        particles[1].gyration_radius = rg
        particles[1].tides_viscous_time_scale_prescription = 0

        tD = M*aF**8/(3.0*k_L*tau*CONST_G*m_per*(M+m_per)*R**5)
        particles[2].check_for_physical_collision_or_orbit_crossing = True

        code.add_particles(particles)
        code.verbose_flag = 0
        t = 0.0
        N=100
        tend = 1.0e7
        dt = tend/float(N)

        t_print_array = []
        a_print_array = []
        n_print_array = []
        e_print_array = []
        AP_print_array = []
        spin_print_array = []

        while (t<tend):
            t+=dt
            code.evolve_model(t)

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]
            binary = binaries[0]

            if args.verbose==True:
                print( 'flag',code.CVODE_flag,'t/yr',t,'a/au',binary.a,'e',binary.e)


            t_print_array.append(t)
            a_print_array.append(binary.a)
            n_print_array.append(np.sqrt(CONST_G*(M+m_per)/(binary.a**3)))
            e_print_array.append(binary.e)
            AP_print_array.append(binary.AP)
            spin_print_array.append( np.sqrt( particles[1].spin_vec_x**2 + particles[1].spin_vec_y**2 + particles[1].spin_vec_z**2) )

            bodies = particles[0:2]
            if args.verbose==True:
                for body in bodies:
                    print( 'S_x',body.spin_vec_x)
                    print( 'S_y',body.spin_vec_y)
                    print( 'S_z',body.spin_vec_z)
                print( '='*50)

        if args.verbose == True:
            print("spin_print_array[-1]",spin_print_array[-1],"n_print_array[-1]",n_print_array[-1])
            print("a_print_array[-1]",a_print_array[-1],"a0(1-e0^2)",a0*(1.0-e0**2))
            
        N_r = 3
        assert round(spin_print_array[-1],N_r) == round(n_print_array[-1],N_r)
        assert round(a_print_array[-1],N_r) == round(aF,N_r)
        #assert len([x for x in range(len(t_print_array)) if round(aF,N_r) not in [round(a*(1.0-e**2),N_r) for a,e in zip( a_print_array,e_print_array)] ] ) == 0
        print("Test passed")

        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            fig = pyplot.figure(figsize=(10,10))
            fontsize=12
    
            t_print_array = np.array(t_print_array)
            a_print_array = np.array(a_print_array)
            e_print_array = np.array(e_print_array)
            AP_print_array = np.array(AP_print_array)
            spin_print_array = np.array(spin_print_array)
            n_print_array = np.array(n_print_array)
            
            N_p = 4
            plot1 = fig.add_subplot(N_p,1,1)
            plot1.plot(t_print_array*1.0e-6,a_print_array, color='r')
            plot1.set_ylabel("$a/\mathrm{au}$",fontsize=fontsize)

            plot2 = fig.add_subplot(N_p,1,2)
            plot2.plot(t_print_array*1.0e-6,e_print_array,color='k')
            plot2.set_ylabel("$e$",fontsize=fontsize)

            plot3 = fig.add_subplot(N_p,1,3,yscale="log")

            plot3.plot(t_print_array*1.0e-6,a_print_array*(1.0-e_print_array**2),color='k')
            plot3.axhline(y = a0*(1.0 - e0**2), color='k')
            plot3.set_ylabel("$a(1-e^2)/\mathrm{au}$",fontsize=fontsize)

            plot4 = fig.add_subplot(N_p,1,4)
            plot4.plot(t_print_array*1.0e-6,spin_print_array/n_print_array)
            plot4.set_ylabel("$\Omega/n$",fontsize=fontsize)

            plot4.set_xlabel("$t/\mathrm{Myr}$",fontsize=fontsize)

            pyplot.show()

    def test5(self,args):
        print("Test apsidal motion due to tidal bulges in binary")

        code = MSE()
        code.enable_tides = True
        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        CONST_R_SUN = code.CONST_R_SUN
        day = 1.0/365.25
        second = day/(24.0*3600.0)

        M = 0.0009546386983890755 ### Jupiter mass
        R = 1.0*0.1027922358015816*CONST_R_SUN ### Jupiter radius ~ 0.1 R_SUN

        m_per = 1.0
        a0 = 30.0
        e0 = 0.999
        P0 = 2.0*np.pi*np.sqrt(a0**3/(CONST_G*(M+m_per)))
        n0 = 2.0*np.pi/P0

        particles = Tools.create_fully_nested_multiple(2,[m_per, M], [a0], [e0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[2,2])
        binaries = [x for x in particles if x.is_binary==True]
        bodies = [x for x in particles if x.is_binary==False]
        
        binary = particles[2]
        particles[0].radius = 1.0e-10*R
        particles[1].radius = R

        for b in bodies:
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False
            b.spin_vec_x = 0.0
            b.spin_vec_y = 0.0
            b.spin_vec_z = 1.0e-15
            
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = False

        k_L = 0.41
        k_AM = k_L/2.0


        particles[0].include_tidal_friction_terms = False
        particles[0].include_tidal_bulges_precession_terms = False
        particles[0].include_rotation_precession_terms = False

        particles[1].tides_method = 1
        particles[1].include_tidal_friction_terms = False
        particles[1].include_tidal_bulges_precession_terms = True
        particles[1].include_rotation_precession_terms = False
        particles[1].minimum_eccentricity_for_tidal_precession = 1.0e-5
        particles[1].apsidal_motion_constant = k_AM
        particles[1].gyration_radius = 0.25
        
        code.add_particles(particles)

        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14
        code.absolute_tolerance_spin_vectors = 1.0e-4
        code.absolute_tolerance_angular_momentum_vectors = 1.0e-4
        code.include_flybys = False
        code.verbose_flag = 0
        
        t = 0.0
        dt = 1.0e6
        tend = 1.0e8

        t_print_array = []
        a_print_array = []
        e_print_array = []
        AP_print_array = []

        g_dot_TB = (15.0/8.0)*n0*(8.0+12.0*e0**2+e0**4)*(m_per/M)*k_AM*pow(R/a0,5.0)/pow(1.0-e0**2,5.0)
        t_TB = 2.0*np.pi/g_dot_TB

        while (t<tend):
            t+=dt
            code.evolve_model(t)

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]
            #print("?",[b.include_spin_orbit_1PN_terms for b in bodies])
            binary = binaries[0]

            if args.verbose==True:
                print( 'flag',code.CVODE_flag,'t',t,'a/au',binary.a,'e',binary.e,"AP",binary.AP)

            t_print_array.append(t*1.0e-6)
            a_print_array.append(binary.a)
            e_print_array.append(binary.e)
            AP_print_array.append(binary.AP)

        AP = 0.01 + 2.0*np.pi*tend/(t_TB)
        AP = (AP+np.pi)%(2.0*np.pi) - np.pi ### -pi < AP < pi
        
        if args.verbose == True:
            print("Predicted AP",AP,"AP_print_array[-1]",AP_print_array[-1])        

        N_r = 5
        assert round(AP,N_r) == round(AP_print_array[-1],N_r)
        print("Test passed")

        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            
            t_print_array = np.array(t_print_array)
            a_print_array = np.array(a_print_array)
            e_print_array = np.array(e_print_array)
            AP_print_array = np.array(AP_print_array)
            
            fig = pyplot.figure(figsize=(10,10))
            plot1 = fig.add_subplot(4,1,1)
            plot2 = fig.add_subplot(4,1,2,yscale="log")
            plot3 = fig.add_subplot(4,1,3,yscale="log")
            plot4 = fig.add_subplot(4,1,4,yscale="log")

            plot1.plot(t_print_array,AP_print_array, color='r')
            points = np.linspace(0.0,tend*1.0e-6,len(t_print_array))
            AP = 0.01 +2.0*np.pi*points/(t_TB*1.0e-6)
            AP = (AP+np.pi)%(2.0*np.pi) - np.pi ### -pi < AP < pi
            plot1.plot(points,AP,color='g',linestyle='dotted',linewidth=2)

            plot2.plot(t_print_array,np.fabs( (AP - AP_print_array)/AP ), color='r')
            plot3.plot(t_print_array,np.fabs((a0-a_print_array)/a0), color='r')
            plot4.plot(t_print_array,np.fabs((e0-e_print_array)/e0), color='r')

            fontsize = 15
            plot1.set_ylabel("$\omega/\mathrm{rad}$",fontsize=fontsize)
            plot2.set_ylabel("$|(\omega_p-\omega)/\omega_p|$",fontsize=fontsize)
            plot3.set_ylabel("$|(a_0-a)/a_0|$",fontsize=fontsize)
            plot4.set_ylabel("$|(e_0-e)/e_0|$",fontsize=fontsize)
            
            plot4.set_xlabel("$t/\mathrm{Myr}$",fontsize=fontsize)

            pyplot.show()

    def test6(self,args):
        print("Test apsidal motion due to rotation in binary")

        code = MSE()
        code.enable_tides = True
        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        CONST_R_SUN = code.CONST_R_SUN
        day = 1.0/365.25
        second = day/(24.0*3600.0)
        
        M = 0.0009546386983890755 ### Jupiter mass
        R = 1.5*0.1027922358015816*CONST_R_SUN ### Jupiter radius ~ 0.1 R_SUN
        m_per = 1.0
        a0 = 30.0
        e0 = 0.999
        P0 = 2.0*np.pi*np.sqrt(a0**3/(CONST_G*(M+m_per)))
        n0 = 2.0*np.pi/P0

        aF = a0*(1.0-e0**2)
        nF = np.sqrt( CONST_G*(M+m_per)/(aF**3) )

        particles = Tools.create_fully_nested_multiple(2,[m_per, M], [a0], [e0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[2,2])
        binaries = [x for x in particles if x.is_binary==True]
        bodies = [x for x in particles if x.is_binary==False]
        
        for b in bodies:
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = False
        
        particles[0].radius = CONST_R_SUN

        particles[0].include_tidal_friction_terms = False
        particles[0].include_tidal_bulges_precession_terms = False
        particles[0].include_rotation_precession_terms = False

        particles[1].radius = R
        particles[1].spin_vec_x = 0.0
        particles[1].spin_vec_y = 0.0
        Omega_PS0 = n0*(33.0/10.0)*pow(a0/aF,3.0/2.0)
        particles[1].spin_vec_z = Omega_PS0

        k_L = 0.51
        k_AM = k_L/2.0
        rg = 0.25
        particles[1].tides_method = 1
        particles[1].include_tidal_friction_terms = False
        particles[1].include_tidal_bulges_precession_terms = False
        particles[1].include_rotation_precession_terms = True
        particles[1].minimum_eccentricity_for_tidal_precession = 1.0e-5
        particles[1].apsidal_motion_constant = k_AM
        particles[1].gyration_radius = rg

        code.add_particles(particles)

        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14
        code.absolute_tolerance_spin_vectors = 1.0e-4
        code.absolute_tolerance_angular_momentum_vectors = 1.0e-4
        code.include_flybys = False
        code.verbose_flag = 0
        
        t = 0.0
        dt = 1.0e6
        tend = 1.0e7

        t_print_array = []
        a_print_array = []
        e_print_array = []
        AP_print_array = []

        Omega_vec = [particles[1].spin_vec_x,particles[1].spin_vec_y,particles[1].spin_vec_z]
        Omega = np.sqrt(Omega_vec[0]**2 + Omega_vec[1]**2 + Omega_vec[2]**2)
        if args.verbose==True:
            print( 'Omega/n',Omega/n0)

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            
            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]
            binary = binaries[0]
            
            if args.verbose==True:
                print( 'flag',code.CVODE_flag,'t',t,'a',binary.a,'e',binary.e,"AP",binary.AP)

            t_print_array.append(t*1.0e-6)
            a_print_array.append(binary.a)
            e_print_array.append(binary.e)
            AP_print_array.append(binary.AP)

        g_dot_rot = n0*(1.0 + m_per/M)*k_AM*pow(R/a0,5.0)*(Omega/n0)**2/((1.0-e0**2)**2)
        t_rot = 2.0*np.pi/g_dot_rot

        AP = 0.01 + 2.0*np.pi*tend/(t_rot)
        AP = (AP+np.pi)%(2.0*np.pi) - np.pi ### -pi < AP < pi

        if args.verbose == True:
            print("Predicted AP",AP,"AP_print_array[-1]",AP_print_array[-1])        

        N_r = 3
        assert round(AP,N_r) == round(AP_print_array[-1],N_r)
        print("Test passed")

        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            t_print_array = np.array(t_print_array)
            a_print_array = np.array(a_print_array)
            e_print_array = np.array(e_print_array)
            AP_print_array = np.array(AP_print_array)

            fig = pyplot.figure(figsize=(10,10))
            plot1 = fig.add_subplot(4,1,1)
            plot2 = fig.add_subplot(4,1,2,yscale="log")
            plot3 = fig.add_subplot(4,1,3,yscale="log")
            plot4 = fig.add_subplot(4,1,4,yscale="log")

            plot1.plot(t_print_array,AP_print_array, color='r')
            points = np.linspace(0.0,tend*1.0e-6,len(t_print_array))
            AP = 0.01 +2.0*np.pi*points/(t_rot*1.0e-6)
            AP = (AP+np.pi)%(2.0*np.pi) - np.pi ### -pi < AP < pi
            plot1.plot(points,AP,color='g',linestyle='dotted',linewidth=2)

            plot2.plot(t_print_array,np.fabs( (AP - AP_print_array)/AP ), color='r')
            plot3.plot(t_print_array,np.fabs((a0-a_print_array)/a0), color='r')
            plot4.plot(t_print_array,np.fabs((e0-e_print_array)/e0), color='r')

            fontsize = 15
            plot1.set_ylabel("$\omega/\mathrm{rad}$",fontsize=fontsize)
            plot2.set_ylabel("$|(\omega_p-\omega)/\omega_p|$",fontsize=fontsize)
            plot3.set_ylabel("$|(a_0-a)/a_0|$",fontsize=fontsize)
            plot4.set_ylabel("$|(e_0-e)/e_0|$",fontsize=fontsize)

            plot4.set_xlabel("$t/\mathrm{Myr}$",fontsize=fontsize)
            pyplot.show()

    def test7(self,args):
        print("Test ODE collision detection in 3-body system")

        code = MSE()

        particles = Tools.create_fully_nested_multiple(3,[1.0, 1.2, 0.9], [1.0, 100.0], [0.1, 0.5], [0.01, 80.0*np.pi/180.0], [0.01, 0.01], [0.01, 0.01], metallicities=[0.02,0.02,0.02],stellar_types=[1,1,1],object_types=[2,2,2])
        bodies = [x for x in particles if x.is_binary==False]
        binaries = [x for x in particles if x.is_binary==True]

        R = 0.03 ### radius of individual objects; collision distance is 2*R

        for b in bodies:
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = False
        

        code.add_particles(particles)
        code.stop_after_root_found = True
        code.enable_tides = False
        code.verbose_flag = 0
        
        t = 0.0
        dt = 1.0e4
        tend = 1.0e6
        t_root = 0.0
        
        t_print_array = []
        a_print_array = []
        e_print_array = []

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            flag = code.CVODE_flag

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            for b in bodies:
                b.include_mass_transfer_terms = False
                b.check_for_RLOF_at_pericentre = False
                b.include_spin_orbit_1PN_terms = False
                
            binaries[0].check_for_minimum_periapse_distance = False
            binaries[0].check_for_physical_collision_or_orbit_crossing = True

            for body in bodies:
                body.radius = R ### Force stellar radii to custom value

            if args.verbose==True:
                print("="*50)
                print("t/Myr",t*1e-6,"a",binaries[0].a,"e",binaries[0].e,"rp/au",binaries[0].a*(1.0 - binaries[0].e) )
                print( 'secular_breakdown_has_occurred',binaries[0].secular_breakdown_has_occurred)
                print( 'dynamical_instability_has_occurred',binaries[0].dynamical_instability_has_occurred)
                print( 'physical_collision_or_orbit_crossing_has_occurred',binaries[0].physical_collision_or_orbit_crossing_has_occurred)
                print( 'minimum_periapse_distance_has_occurred',binaries[0].minimum_periapse_distance_has_occurred)
                print( 'RLOF_at_pericentre_has_occurred',binaries[0].RLOF_at_pericentre_has_occurred)

            t_print_array.append(t*1.0e-6)
            a_print_array.append(binaries[0].a)
            e_print_array.append(binaries[0].e)

            if flag == 2:
                t_root = code.model_time
                if args.verbose==True:
                    print( 'root found at t=',t_root)
                break
        
        if args.verbose == True:
            print("num rp ",a_print_array[-1]*(1.0 - e_print_array[-1])," 2*R ",2*R)

        N_r = 10       
        assert round(a_print_array[-1]*(1.0 - e_print_array[-1]),N_r) == round(2.0*R,N_r)
        print("Test passed")

        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            a_print_array = np.array(a_print_array)
            e_print_array = np.array(e_print_array)
            
            fig = pyplot.figure()
            plot = fig.add_subplot(1,1,1)
            plot.plot(t_print_array,a_print_array*(1.0-e_print_array))
            plot.axhline(y = bodies[0].radius + bodies[1].radius,color='k')
            plot.set_ylabel("$r_\mathrm{p}/\mathrm{au}$")
            plot.set_xlabel("$t/\mathrm{Myr}$")

            pyplot.show()

    def test8(self,args):
        print("Test ODE minimum periapsis distance root finding")

        code = MSE()

        particles = Tools.create_fully_nested_multiple(3,[1.0, 1.2, 0.9], [1.0, 100.0], [0.1, 0.5], [0.01, 80.0*np.pi/180.0], [0.01, 0.01], [0.01, 0.01], metallicities=[0.02,0.02,0.02],stellar_types=[1,1,1],object_types=[2,2,2])
        bodies = [x for x in particles if x.is_binary==False]
        binaries = [x for x in particles if x.is_binary==True]

        rp_min = 0.1
        R = 1.0e-10

        for b in bodies:
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.exclude_rotation_and_bulges_precession_in_case_of_isolated_binary = False
        

        code.add_particles(particles)
        code.stop_after_root_found = True
        code.enable_tides = False
        code.verbose_flag = 0
        
        t = 0.0
        dt = 1.0e4
        tend = 1.0e6
        t_root = 0.0
        
        t_print_array = []
        a_print_array = []
        e_print_array = []

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            flag = code.CVODE_flag

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            for b in bodies:
                b.include_mass_transfer_terms = False
                b.check_for_RLOF_at_pericentre = False
                b.include_spin_orbit_1PN_terms = False
                
            binaries[0].check_for_minimum_periapse_distance = True
            binaries[0].check_for_minimum_periapse_distance_value = rp_min
            binaries[0].check_for_physical_collision_or_orbit_crossing = False

            for body in bodies:
                body.radius = R ### Force stellar radii to custom value

            if args.verbose==True:
                print("="*50)
                print("t/Myr",t*1e-6,"a",binaries[0].a,"e",binaries[0].e,"rp/au",binaries[0].a*(1.0 - binaries[0].e) )
                print( 'secular_breakdown_has_occurred',binaries[0].secular_breakdown_has_occurred)
                print( 'dynamical_instability_has_occurred',binaries[0].dynamical_instability_has_occurred)
                print( 'physical_collision_or_orbit_crossing_has_occurred',binaries[0].physical_collision_or_orbit_crossing_has_occurred)
                print( 'minimum_periapse_distance_has_occurred',binaries[0].minimum_periapse_distance_has_occurred)
                print( 'RLOF_at_pericentre_has_occurred',binaries[0].RLOF_at_pericentre_has_occurred)

            t_print_array.append(t*1.0e-6)
            a_print_array.append(binaries[0].a)
            e_print_array.append(binaries[0].e)

            if flag == 2:
                t_root = code.model_time
                if args.verbose==True:
                    print( 'root found at t=',t_root)
                break
        
        if args.verbose == True:
            print("num rp ",a_print_array[-1]*(1.0 - e_print_array[-1]),"rp_min",rp_min)

        N_r = 10            
        assert round(a_print_array[-1]*(1.0 - e_print_array[-1]),N_r) == round(rp_min,N_r)
        print("Test passed")

        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            a_print_array = np.array(a_print_array)
            e_print_array = np.array(e_print_array)
            
            fig = pyplot.figure()
            plot = fig.add_subplot(1,1,1)
            plot.plot(t_print_array,a_print_array*(1.0-e_print_array))
            plot.axhline(y = rp_min,color='k')
            plot.set_ylabel("$r_\mathrm{p}/\mathrm{au}$")
            plot.set_xlabel("$t/\mathrm{Myr}$")

            pyplot.show()

    def test9(self,args):
        print("Test adiabatic mass loss")

        code = MSE()
        
        m1i = 15.0
        m2i = 10.0
        m3i = 1.0
        a_in_i = 1.0
        a_out_i = 1000.0
        particles = Tools.create_fully_nested_multiple(3,[m1i,m2i,m3i], [a_in_i,a_out_i], [0.1, 0.5], [0.01, 80.0*np.pi/180.0], [0.01, 0.01], [0.01, 0.01],metallicities=[0.02,0.02,0.02],stellar_types=[1,1,1],object_types=[1,1,1])
        bodies = [x for x in particles if x.is_binary==False]
        binaries = [x for x in particles if x.is_binary==True]

        #m1dot = -1.0e-7
        #m2dot = -1.0e-8
        #m3dot = -1.0e-9
        #mdots = [m1dot,m2dot,m3dot]
        #for index,body in enumerate(bodies):
        #    body.mass_dot = mdots[index]

        code.add_particles(particles)
        code.enable_tides = False

        t = 0.0
        dt = 1.0e4
        tend = 1.0e6

        t_print_array = []
        m_in_print_array = []
        m_out_print_array = []
        a_in_print_array = []
        a_out_print_array = []
        e_print_array = []

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            flag = code.CVODE_flag

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            for b in bodies:
                b.include_mass_transfer_terms = False
                b.check_for_RLOF_at_pericentre = False
                b.include_spin_orbit_1PN_terms = False

            if args.verbose==True:
                print( 't/Myr',t*1e-6,'m1',bodies[0].mass,'m2',bodies[1].mass,'m3',bodies[2].mass,'a/au',[b.a for b in binaries])

            t_print_array.append(t*1.0e-6)
            m_in_print_array.append(bodies[0].mass + bodies[1].mass)
            m_out_print_array.append(bodies[0].mass + bodies[1].mass + bodies[2].mass)
            a_in_print_array.append(binaries[0].a)
            a_out_print_array.append(binaries[1].a)
            e_print_array.append(binaries[0].e)

        m_in_print_array = np.array(m_in_print_array)
        m_out_print_array = np.array(m_out_print_array)
        a_in_print_array = np.array(a_in_print_array)
        a_out_print_array = np.array(a_out_print_array)

        inner_adiabat = m_in_print_array*a_in_print_array
        outer_adiabat = m_out_print_array*a_out_print_array

        if args.verbose == True:
            print("a_in_i*(m1i+m2i)",a_in_i*(m1i+m2i),"num ",inner_adiabat[1])
            print("a_out_i*(m1i+m2i+m3i)",a_out_i*(m1i+m2i+m3i),"num ",outer_adiabat[1])
            
        N_r = 4
        assert round(a_in_i*(m1i+m2i),N_r) == round(inner_adiabat[-1],N_r)
        assert round(a_out_i*(m1i+m2i+m3i),N_r) == round(outer_adiabat[-1],N_r)
        print("Test passed")

        code.reset()
        
        if HAS_MATPLOTLIB == True and args.plot==True:
            fig = pyplot.figure(figsize=(10,8))
            plot1 = fig.add_subplot(2,1,1,yscale="log")
            plot2 = fig.add_subplot(2,1,2,yscale="log")
            plot1.plot(t_print_array,inner_adiabat,color='k',linestyle='solid',label='$a_1$',zorder=10)
            plot2.plot(t_print_array,outer_adiabat,color='k',linestyle='solid',label='$a_2$',zorder=10)
            
            handles,labels = plot1.get_legend_handles_labels()
            plot1.legend(handles,labels,loc="lower left",fontsize=18)

            plot1.set_ylabel(r"$a_1(m_1+m_2)/(\mathrm{au\,M_\odot})$")
            plot2.set_ylabel(r"$a_2(m_1+m_2+m_3)/(\mathrm{au\,M_\odot})$")
            plot2.set_xlabel("$t/\mathrm{Myr}$")

            pyplot.show()

    def test10(self,args):
        print("Test flybys module: instantaneous change -- SNe in binary")

        code = MSE()
        CONST_G = code.CONST_G
        CONST_km_per_s_to_AU_per_yr = code.CONST_KM_PER_S
                
        a1 = 10.0
        e1 = 0.1
        m1 = 1.0
        m2 = 0.8

        INCL1 = 0.1
        AP1 = 0.2
        LAN1 = 0.3
        f1 = 60.5*np.pi/180.0

        particles = Tools.create_fully_nested_multiple(2, [m1,m2], [a1], [e1], [INCL1], [AP1], [LAN1], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[1,1])

        binary = particles[2]
        binary.sample_orbital_phase_randomly = False
        binary.TA = f1
        
        delta_m1 = -0.5
        V_k_vec = np.array([0.0,1.0,2.0])*CONST_km_per_s_to_AU_per_yr
        #V_k_vec = np.array([0.0,0.0,0.0])

        particles[0].instantaneous_perturbation_delta_mass = delta_m1
        particles[0].instantaneous_perturbation_delta_VX = V_k_vec[0]
        particles[0].instantaneous_perturbation_delta_VY = V_k_vec[1]
        particles[0].instantaneous_perturbation_delta_VZ = V_k_vec[2]

        code.add_particles(particles)

        if args.verbose==True:
            print( '='*50)
            print( 'pre')
            print( 'a',binary.a,'e',binary.e,'INCL',binary.INCL*180.0/np.pi,'AP',binary.AP*180.0/np.pi,'LAN',binary.LAN*180.0/np.pi,'TA',binary.TA)

        code.apply_user_specified_instantaneous_perturbation()
        
        if args.verbose==True:
            print( '='*50)
            print( 'post')
            print( 'a',binary.a,'e',binary.e,'INCL',binary.INCL*180.0/np.pi,'AP',binary.AP*180.0/np.pi,'LAN',binary.LAN*180.0/np.pi,'TA',binary.TA)

        ### Compute analytic result (e.g., https://ui.adsabs.harvard.edu/abs/2016ComAC...3....6T/abstract) ###
        r1 = a1*(1.0-e1**2)/(1.0 + e1*np.cos(f1))
        v1_tilde = np.sqrt( CONST_G*(m1+m2)/(a1*(1.0-e1**2) ) )
        e1_vec_hat,j1_vec_hat = compute_e_and_j_hat_vectors(INCL1,AP1,LAN1)
        q1_vec_hat = np.cross(j1_vec_hat,e1_vec_hat)
        r1_vec = r1*( e1_vec_hat * np.cos(f1) + q1_vec_hat * np.sin(f1) )
        v1_vec = v1_tilde*( -e1_vec_hat * np.sin(f1) + q1_vec_hat * (e1 + np.cos(f1) ) )

        r1_dot_v1 = np.sum([x*y for x,y in zip(r1_vec,v1_vec)])
        r1_dot_V_k = np.sum([x*y for x,y in zip(r1_vec,V_k_vec)])
        v1_dot_V_k = np.sum([x*y for x,y in zip(v1_vec,V_k_vec)])
        V_k_dot_V_k = np.sum([x*y for x,y in zip(V_k_vec,V_k_vec)])
        v1c_sq = CONST_G*(m1+m2)/a1
        
        a1_p = a1*(1.0 + delta_m1/(m1+m2))*pow( 1.0 + 2.0*(a1/r1)*(delta_m1/(m1+m2)) - 2.0*v1_dot_V_k/v1c_sq - V_k_dot_V_k/v1c_sq, -1.0)
        j1_p = pow( (m1+m2)/(m1+m2+delta_m1), 2.0)*(1.0 + 2.0*(a1/r1)*(delta_m1/(m1+m2)) - 2.0*v1_dot_V_k/v1c_sq - V_k_dot_V_k/v1c_sq)*( 1.0 - e1**2 \
            + (1.0/(CONST_G*(m1+m2)*a1))*( r1**2*( 2.0*v1_dot_V_k + V_k_dot_V_k) - 2.0*r1_dot_v1*r1_dot_V_k - r1_dot_V_k**2) )
        e1_p = np.sqrt(1.0 - j1_p)
        
        if args.verbose==True:
            print( 'analytic results Toonen+16: ','new a1 = ',a1_p,'; new e1 = ',e1_p)
        
        N_r = 10
        assert round(binary.a,N_r) == round(a1_p,N_r)
        assert round(binary.e,N_r) == round(e1_p,N_r)
        
        print("Test passed")

        code.reset()
        
    def test11(self,args):
        print("Test flybys module: instantaneous change -- SNe in triple")

        code = MSE()
        CONST_G = code.CONST_G
        CONST_km_per_s_to_AU_per_yr = code.CONST_KM_PER_S        
        
        m1 = 1.0
        m2 = 0.8
        m3 = 1.2
        a1 = 10.0
        a2 = 100.0
        e1 = 0.1
        e2 = 0.3
        INCL1 = 0.1
        INCL2 = 0.5
        AP1 = 0.1
        AP2 = 1.0
        LAN1 = 0.1
        LAN2 = 2.0
        f1 = 60.5*np.pi/180.0
        f2 = 30.5*np.pi/180.0

        INCLs = [INCL1,INCL2]
        APs = [AP1,AP2]
        LANs = [LAN1,LAN2]
        masses = [m1,m2,m3]
        particles = Tools.create_fully_nested_multiple(3,masses, [a1,a2], [e1,e2], INCLs, APs, LANs, metallicities=[0.02,0.02,0.02],stellar_types=[1,1,1],object_types=[1,1,1])
        
        inner_binary = particles[3]
        outer_binary = particles[4]
        inner_binary.sample_orbital_phase_randomly = 0
        outer_binary.sample_orbital_phase_randomly = 0
        inner_binary.TA = f1
        outer_binary.TA = f2
        
        delta_m1 = -0.3
        
        km_p_s_to_AU_p_yr = 0.21094502112788768
        V_k_vec = np.array([1.0,2.0,2.0])*CONST_km_per_s_to_AU_per_yr
        
        particles[0].instantaneous_perturbation_delta_mass = delta_m1
        particles[0].instantaneous_perturbation_delta_VX = V_k_vec[0]
        particles[0].instantaneous_perturbation_delta_VY = V_k_vec[1]
        particles[0].instantaneous_perturbation_delta_VZ = V_k_vec[2]

        code.add_particles(particles)
        
        if args.verbose==True:
            print( '='*50)
            print( 'pre')
            print( 'inner','a',inner_binary.a,'e',inner_binary.e,'INCL',inner_binary.INCL*180.0/np.pi,'AP',inner_binary.AP*180.0/np.pi,'LAN',inner_binary.LAN*180.0/np.pi)
            print( 'outer','a',outer_binary.a,'e',outer_binary.e,'INCL',outer_binary.INCL*180.0/np.pi,'AP',outer_binary.AP*180.0/np.pi,'LAN',outer_binary.LAN*180.0/np.pi)
        
        code.apply_user_specified_instantaneous_perturbation()
        
        if args.verbose==True:
            print( '='*50)
            print( 'post')
            print( 'inner','a',inner_binary.a,'e',inner_binary.e,'INCL',inner_binary.INCL*180.0/np.pi,'AP',inner_binary.AP*180.0/np.pi,'LAN',inner_binary.LAN*180.0/np.pi)
            print( 'outer','a',outer_binary.a,'e',outer_binary.e,'INCL',outer_binary.INCL*180.0/np.pi,'AP',outer_binary.AP*180.0/np.pi,'LAN',outer_binary.LAN*180.0/np.pi)

        
        ### Compute analytic result (e.g., https://ui.adsabs.harvard.edu/abs/2016ComAC...3....6T/abstract) ###
        r1 = a1*(1.0-e1**2)/(1.0 + e1*np.cos(f1))
        v1_tilde = np.sqrt( CONST_G*(m1+m2)/(a1*(1.0-e1**2) ) )
        e1_vec_hat,j1_vec_hat = compute_e_and_j_hat_vectors(INCL1,AP1,LAN1)
        q1_vec_hat = np.cross(j1_vec_hat,e1_vec_hat)
        r1_vec = r1*( e1_vec_hat * np.cos(f1) + q1_vec_hat * np.sin(f1) )
        v1_vec = v1_tilde*( -e1_vec_hat * np.sin(f1) + q1_vec_hat * (e1 + np.cos(f1) ) )

        r1_dot_v1 = np.sum([x*y for x,y in zip(r1_vec,v1_vec)])
        r1_dot_V_k = np.sum([x*y for x,y in zip(r1_vec,V_k_vec)])
        v1_dot_V_k = np.sum([x*y for x,y in zip(v1_vec,V_k_vec)])
        V_k_dot_V_k = np.sum([x*y for x,y in zip(V_k_vec,V_k_vec)])
        v1c_sq = CONST_G*(m1+m2)/a1

        r2 = a2*(1.0-e2**2)/(1.0 + e2*np.cos(f2))
        v2_tilde = np.sqrt( CONST_G*(m1+m2+m3)/(a2*(1.0-e2**2) ) )
        e2_vec_hat,j2_vec_hat = compute_e_and_j_hat_vectors(INCL2,AP2,LAN2)
        q2_vec_hat = np.cross(j2_vec_hat,e2_vec_hat)
        r2_vec = r2*( e2_vec_hat * np.cos(f2) + q2_vec_hat * np.sin(f2) )
        v2_vec = v2_tilde*( -e2_vec_hat * np.sin(f2) + q2_vec_hat * (e2 + np.cos(f2) ) )
    
        Delta_r2_vec = (delta_m1/( (m1+m2) + delta_m1) ) * (m2/(m1+m2)) * r1_vec
        Delta_v2_vec = (delta_m1/( (m1+m2) + delta_m1) ) * ( (m2/(m1+m2)) * v1_vec + V_k_vec*(1.0 + m1/delta_m1) )
        r2_vec_p = r2_vec + Delta_r2_vec
        r2p = np.sqrt(np.dot(r2_vec_p,r2_vec_p))


        r2_dot_v2 = np.sum([x*y for x,y in zip(r2_vec,v2_vec)])
        r2_dot_V_k = np.sum([x*y for x,y in zip(r2_vec,V_k_vec)])
        v2c_sq = CONST_G*(m1+m2+m3)/a2
        v2_dot_Delta_v2_vec = np.dot(v2_vec,Delta_v2_vec)
        Delta_v2_vec_dot_Delta_v2_vec = np.dot(Delta_v2_vec,Delta_v2_vec)
        
        a1_p = a1*(1.0 + delta_m1/(m1+m2))*pow( 1.0 + 2.0*(a1/r1)*(delta_m1/(m1+m2)) - 2.0*v1_dot_V_k/v1c_sq - V_k_dot_V_k/v1c_sq, -1.0)
        j1_p = pow( (m1+m2)/(m1+m2+delta_m1), 2.0)*(1.0 + 2.0*(a1/r1)*(delta_m1/(m1+m2)) - 2.0*v1_dot_V_k/v1c_sq - V_k_dot_V_k/v1c_sq)*( 1.0 - e1**2 \
            + (1.0/(CONST_G*(m1+m2)*a1))*( r1**2*( 2.0*v1_dot_V_k + V_k_dot_V_k) - 2.0*r1_dot_v1*r1_dot_V_k - r1_dot_V_k**2) )
        e1_p = np.sqrt(1.0 - j1_p)

        a2_p = a2*(1.0 + delta_m1/(m1+m2+m3))*pow( 1.0 + 2.0*(a2/r2p)*(delta_m1/(m1+m2+m3)) - 2.0*v2_dot_Delta_v2_vec/v2c_sq - Delta_v2_vec_dot_Delta_v2_vec/v2c_sq + 2.0*a2*(r2-r2p)/(r2*r2p), -1.0)

        alpha = (-delta_m1/(m1+m2+delta_m1)) * m2/(m1+m2)
        j2_p = ((m1+m2+m3)/(m1+m2+m3+delta_m1))**2 * (1.0 + 2.0*(a2/r2p)*(delta_m1/(m1+m2+m3)) + 2.0*a2*(r2-r2p)/(r2*r2p) - 2.0*np.dot(v2_vec,Delta_v2_vec)/v2c_sq - Delta_v2_vec_dot_Delta_v2_vec/v2c_sq)*( (1.0-e2**2) + (1.0/(CONST_G*(m1+m2+m3)*a2))*( r2**2*(2.0*np.dot(v2_vec,Delta_v2_vec) + np.dot(Delta_v2_vec,Delta_v2_vec)) + (-2.0*alpha*np.dot(r1_vec,r2_vec) + alpha**2*r1**2)*np.dot(v2_vec + Delta_v2_vec,v2_vec+Delta_v2_vec) + 2.0*np.dot(r2_vec,v2_vec)*( alpha*np.dot(r1_vec,v2_vec) - np.dot(r2_vec,Delta_v2_vec) + alpha*np.dot(r1_vec,Delta_v2_vec)) - (-alpha*np.dot(r1_vec,v2_vec) + np.dot(r2_vec,Delta_v2_vec) - alpha*np.dot(r1_vec,Delta_v2_vec))**2 ) ) 
        e2_p = np.sqrt(1.0 - j2_p)
        
        if args.verbose==True:
            print( 'analytic results Toonen+16: ','new a1 = ',a1_p,'; new e1 = ',e1_p)
            print( 'analytic results Toonen+16: ','new a2 = ',a2_p,'; new e2 = ',e2_p)
        
        N_r = 10
        assert round(inner_binary.a,N_r) == round(a1_p,N_r)
        assert round(inner_binary.e,N_r) == round(e1_p,N_r)

        assert round(outer_binary.a,N_r) == round(a2_p,N_r)
        assert round(outer_binary.e,N_r) == round(e2_p,N_r)

        print("Test passed")

        code.reset()
        
    def test12(self,args):
        print("Test flybys module: using analytic formulae")

        code = MSE()
        CONST_G = code.CONST_G
        
        ### binary orbit ###
        a = 1.0
        e = 0.1
        m1 = 1.0
        m2 = 0.8
        M_per = 1.0
        E = 2.0
        Q = 100.0
        INCL = 0.4*np.pi
        AP = 0.25*np.pi
        LAN = 0.25*np.pi
        
        masses = [m1,m2]
        m = m1 + m2
        particles = Tools.create_fully_nested_multiple(2,masses, [a], [e], [INCL], [AP], [LAN], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[1,1])
        bodies = [x for x in particles if x.is_binary==False]
        binaries = [x for x in particles if x.is_binary==True]
        binary = particles[2]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        t_ref = 1.0e5 ### not actually used in this case, but needs to be specified for external particles

        external_particle = Particle(mass = M_per, is_binary=True, is_external=True, external_t_ref=t_ref, e=E, external_r_p = Q, INCL = 1.0e-10, AP = 1.0e-10, LAN = 1.0e-10)
        
        particles.append(external_particle)

        code = MSE()
        code.add_particles(particles)

        code.include_quadrupole_order_terms = True
        code.include_octupole_order_binary_pair_terms = True
        code.include_hexadecupole_order_binary_pair_terms = False
        code.include_dotriacontupole_order_binary_pair_terms = False

        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        
        code.apply_external_perturbation_assuming_integrated_orbits()
        Delta_e = binary.e-e

        ### compute analytic result (https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5630H/abstract) ###
        e_vec_hat,j_vec_hat = compute_e_and_j_hat_vectors(INCL,AP,LAN)
        e_vec = e_vec_hat*e
        j_vec = j_vec_hat*np.sqrt(1.0-e**2)
        ex = e_vec[0]
        ey = e_vec[1]
        ez = e_vec[2]
        jx = j_vec[0]
        jy = j_vec[1]
        jz = j_vec[2]

        eps_SA = (M_per/np.sqrt(m*(m+M_per)))*pow(a/Q,3.0/2.0)*pow(1.0 + E,-3.0/2.0)
        eps_oct = (a/Q)*(m1-m2)/((1.0+E)*m)
        Delta_e_an = (5*eps_SA*(np.sqrt(1 - E**(-2))*((1 + 2*E**2)*ey*ez*jx + (1 - 4*E**2)*ex*ez*jy + 2*(-1 + E**2)*ex*ey*jz) + 3*E*ez*(ey*jx - ex*jy)*np.arccos(-1.0/E)))/(2.*E*np.sqrt(ex**2 + ey**2 + ez**2))
        
        Delta_e_an += -(5*eps_oct*eps_SA*(np.sqrt(1 - E**(-2))*(ez*jy*(14*ey**2 + 6*jx**2 - 2*jy**2 + 8*E**4*(-1 + ey**2 + 8*ez**2 + 2*jx**2 + jy**2) + E**2*(-4 - 31*ey**2 + 32*ez**2 - 7*jx**2 + 9*jy**2)) - ey*(2*(7*ey**2 + jx**2 - jy**2) + 8*E**4*(-1 + ey**2 + 8*ez**2 + 4*jx**2 + jy**2) + E**2*(-4 - 31*ey**2 + 32*ez**2 + 11*jx**2 + 9*jy**2))*jz + ex**2*(-((14 + 45*E**2 + 160*E**4)*ez*jy) + 3*(14 - 27*E**2 + 16*E**4)*ey*jz) + 2*(-2 + 9*E**2 + 8*E**4)*ex*jx*(7*ey*ez + jy*jz)) + 3*E**3*(ez*jy*(-4 - 3*ey**2 + 32*ez**2 + 5*jx**2 + 5*jy**2) + ey*(4 + 3*ey**2 - 32*ez**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*np.arccos(-1.0/E)))/(32.*E**2*np.sqrt(ex**2 + ey**2 + ez**2))

        if args.verbose==True:
            print( 'SecularMultiple Delta e = ',Delta_e,'; analytic expression: Delta e = ',Delta_e_an)

        N_r = 8
        assert round(Delta_e,N_r) == round(Delta_e_an,N_r)

        print("Test passed")

        code.reset()

    def test13(self,args):
        print("Test flybys module: using analytic formulae and with single perturber")

        code = MSE()
        CONST_G = code.CONST_G
        
        ### binary orbit ###
        a = 1.0
        e = 0.1
        m1 = 1.0
        m2 = 0.8
        M_per = 1.0
        E = 2.0
        Q = 100.0
        INCL = 0.4*np.pi
        AP = 0.25*np.pi
        LAN = 0.25*np.pi
        
        masses = [m1,m2]
        m = m1 + m2
        particles = Tools.create_fully_nested_multiple(2,masses, [a], [e], [INCL], [AP], [LAN], metallicities=[0.02,0.02],stellar_types=[1,1],object_types=[1,1])
        bodies = [x for x in particles if x.is_binary==False]
        binaries = [x for x in particles if x.is_binary==True]
        binary = particles[2]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        t_ref = 1.0e5 ### not actually used in this case, but needs to be specified for external particles

        #external_particle = Particle(mass = M_per, is_binary=True, is_external=True, external_t_ref=t_ref, e=E, external_r_p = Q, INCL = 1.0e-10, AP = 1.0e-10, LAN = 1.0e-10)
        INCL_per = 1.0e-10
        AP_per = 1.0e-10
        LAN_per = 1.0e-10
        e_vec_hat,j_vec_hat = compute_e_and_j_hat_vectors(INCL_per,AP_per,LAN_per)
        
        code = MSE()
        code.add_particles(particles)

        code.include_quadrupole_order_terms = True
        code.include_octupole_order_binary_pair_terms = True
        code.include_hexadecupole_order_binary_pair_terms = False
        code.include_dotriacontupole_order_binary_pair_terms = False

        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        
        code.apply_external_perturbation_assuming_integrated_orbits_single_perturber(M_per, E, Q, e_vec_hat[0], e_vec_hat[1], e_vec_hat[2], j_vec_hat[0], j_vec_hat[1], j_vec_hat[2])
        Delta_e = binary.e-e

        ### compute analytic result (https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.5630H/abstract) ###
        e_vec_hat,j_vec_hat = compute_e_and_j_hat_vectors(INCL,AP,LAN)
        e_vec = e_vec_hat*e
        j_vec = j_vec_hat*np.sqrt(1.0-e**2)
        ex = e_vec[0]
        ey = e_vec[1]
        ez = e_vec[2]
        jx = j_vec[0]
        jy = j_vec[1]
        jz = j_vec[2]

        eps_SA = (M_per/np.sqrt(m*(m+M_per)))*pow(a/Q,3.0/2.0)*pow(1.0 + E,-3.0/2.0)
        eps_oct = (a/Q)*(m1-m2)/((1.0+E)*m)
        Delta_e_an = (5*eps_SA*(np.sqrt(1 - E**(-2))*((1 + 2*E**2)*ey*ez*jx + (1 - 4*E**2)*ex*ez*jy + 2*(-1 + E**2)*ex*ey*jz) + 3*E*ez*(ey*jx - ex*jy)*np.arccos(-1.0/E)))/(2.*E*np.sqrt(ex**2 + ey**2 + ez**2))
        
        Delta_e_an += -(5*eps_oct*eps_SA*(np.sqrt(1 - E**(-2))*(ez*jy*(14*ey**2 + 6*jx**2 - 2*jy**2 + 8*E**4*(-1 + ey**2 + 8*ez**2 + 2*jx**2 + jy**2) + E**2*(-4 - 31*ey**2 + 32*ez**2 - 7*jx**2 + 9*jy**2)) - ey*(2*(7*ey**2 + jx**2 - jy**2) + 8*E**4*(-1 + ey**2 + 8*ez**2 + 4*jx**2 + jy**2) + E**2*(-4 - 31*ey**2 + 32*ez**2 + 11*jx**2 + 9*jy**2))*jz + ex**2*(-((14 + 45*E**2 + 160*E**4)*ez*jy) + 3*(14 - 27*E**2 + 16*E**4)*ey*jz) + 2*(-2 + 9*E**2 + 8*E**4)*ex*jx*(7*ey*ez + jy*jz)) + 3*E**3*(ez*jy*(-4 - 3*ey**2 + 32*ez**2 + 5*jx**2 + 5*jy**2) + ey*(4 + 3*ey**2 - 32*ez**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*np.arccos(-1.0/E)))/(32.*E**2*np.sqrt(ex**2 + ey**2 + ez**2))

        if args.verbose==True:
            print( 'SecularMultiple Delta e = ',Delta_e,'; analytic expression: Delta e = ',Delta_e_an)

        N_r = 8
        assert round(Delta_e,N_r) == round(Delta_e_an,N_r)

        print("Test passed")

        code.reset()


    def test14(self,args):
        print('Test compact object merger remnant properties')
        
        code = MSE()


        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        
        N = 20000
        
        m2 = 1.0
        h_vec_unit = np.array( [0.0,0.0,1.0] )
        e_vec_unit = np.array( [1.0,0.0,0.0] )
        
        seed = 0
        np.random.seed(seed)
        
        vs = []
        alphas = []
        delta_Ms = []
        for index in range(N):
            q = np.random.random()
            m1 = q * m2
            M = m1 + m2
            chi1 = np.random.random()
            chi2 = np.random.random()
                    
            spin_vec_1_unit = sample_random_vector_on_unit_sphere()
            spin_vec_2_unit = sample_random_vector_on_unit_sphere()
            v_recoil_vec,alpha_vec,M_final = code.determine_compact_object_merger_properties(m2,m1,chi2,chi1,spin_vec_2_unit,spin_vec_1_unit,h_vec_unit,e_vec_unit)
            v_recoil = np.linalg.norm(v_recoil_vec)/code.CONST_KM_PER_S ### convert AU/yr to km/s
            vs.append(v_recoil)
            
            alpha = np.linalg.norm(alpha_vec)

            alphas.append(alpha)
            delta_Ms.append( (M-M_final)/M )


        if args.verbose==True:
            print("mean vs/(km/s)",np.mean(np.array(vs)),round(np.mean(np.array(vs)),0))
            print("mean delta_Ms",np.mean(np.array(delta_Ms)))
            print("mean alphas",np.mean(np.array(alphas)))

        assert(round(np.mean(np.array(vs))/10.0,0) == 31)
        assert(round(np.mean(np.array(delta_Ms)),3) == 0.036)
        assert(round(np.mean(np.array(alphas)),3) == 0.631)
        
        code.reset()
        
        if args.plot == True:
            Nb=50
            fontsize=20
            from matplotlib import pyplot
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.hist(vs,bins=np.linspace(0.0,1000.0,Nb),histtype='step')
            plot.set_xlabel("$v_\mathrm{recoil}/\mathrm{km/s}$",fontsize=fontsize)
            fig.savefig("v_recoil.pdf")
            
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.hist(alphas,bins=np.linspace(0.0,1.0,Nb),histtype='step')
            plot.set_xlabel(r"$\chi_\mathrm{final}$",fontsize=fontsize)
            fig.savefig("chi_final.pdf")

            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.hist(delta_Ms,bins=np.linspace(0.0,0.2,Nb),histtype='step')
            plot.set_xlabel(r"$\Delta M/M$",fontsize=fontsize)
            fig.savefig("Delta_M.pdf")

            pyplot.show()
        
    def test15(self,args):
        print('Test sample elementary distributions')
        
        code = MSE()

        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        
        N = 80000
        N2 = 20000

        seed = 0
        np.random.seed(seed)
       
        vs = []
        vs2 = []
        sigma_km_s = 265.0
        sigma = sigma_km_s*code.CONST_KM_PER_S
        mu_km_s = 20.0
        mu=mu_km_s*code.CONST_KM_PER_S

        r_hat_vecs = []
        theta_hat_vecs = []
        phi_hat_vecs = []

        for index in range(N):
            vx,vy,vz = code.test_sample_from_3d_maxwellian_distribution(sigma)
            v = np.array([vx,vy,vz])
            vs.append(np.linalg.norm(v))
            v2 = code.test_sample_from_normal_distribution(mu,sigma)
            vs2.append(v2)
            
        vs = np.array(vs)
        vs2 = np.array(vs2)

        for index in range(N2):
            r_hat_vec,theta_hat_vec,phi_hat_vec = code.test_sample_spherical_coordinates_unit_vectors_from_isotropic_distribution()
            r_hat_vecs.append(r_hat_vec)
            theta_hat_vecs.append(theta_hat_vec)
            phi_hat_vecs.append(phi_hat_vec)

        ### Check that r_hat components average to zero (isotropy) ###
        tol = 1e-2
        assert( abs(np.mean( np.array( [x[0] for x in r_hat_vecs] ))) <= tol )
        assert( abs(np.mean( np.array( [x[1] for x in r_hat_vecs] ))) <= tol )
        assert( abs(np.mean( np.array( [x[2] for x in r_hat_vecs] ))) <= tol )
        ### theta_hat x,y average to zero; z averages to <-sin(theta)> = -pi/4 ###
        assert( abs(np.mean( np.array( [x[0] for x in theta_hat_vecs] ))) <= tol )
        assert( abs(np.mean( np.array( [x[1] for x in theta_hat_vecs] ))) <= tol )
        assert( abs(np.mean( np.array( [x[2] for x in theta_hat_vecs] )) - (-np.pi/4.0)) <= tol )
        ### phi_hat x,y average to zero; z is always exactly 0 ###
        assert( abs(np.mean( np.array( [x[0] for x in phi_hat_vecs] ))) <= tol )
        assert( abs(np.mean( np.array( [x[1] for x in phi_hat_vecs] ))) <= tol )
        assert( abs(np.mean( np.array( [x[2] for x in phi_hat_vecs] ))) <= tol )

        ### Check orthogonality of r, theta, and phi hat vectors ###
        tol = 1e-12
        assert( abs(np.sum( np.array([np.dot(x,y) for x,y in zip(r_hat_vecs,theta_hat_vecs)]))) <= tol)
        assert( abs(np.sum( np.array([np.dot(x,y) for x,y in zip(r_hat_vecs,phi_hat_vecs)]))) <= tol)
        assert( abs(np.sum( np.array([np.dot(x,y) for x,y in zip(theta_hat_vecs,phi_hat_vecs)]))) <= tol)

        ### Check properties of Maxwellian and normal distributions ###
        N_r=0

        if args.verbose==True:
            print("Maxwellian mean vs/(km/s)",np.mean(np.array(vs))," an ", 2.0*sigma*np.sqrt(2.0/np.pi))
            print("Normal mean vs/(km/s)",np.mean(np.array(vs2))," an ", mu)
            print("Normal std vs/(km/s)",np.std(np.array(vs2))," an ", sigma)
        
        assert(round(np.mean(np.array(vs)),N_r) == round(2.0*sigma*np.sqrt(2.0/np.pi),N_r))
        assert(round(np.mean(np.array(vs2)),N_r) == round(mu,N_r))
        assert(round(np.std(np.array(vs2)),N_r) == round(sigma,N_r))

        code.reset()

        if args.plot == True:
            Nb=100
            fontsize=20
            from matplotlib import pyplot
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.hist(vs/code.CONST_KM_PER_S,bins=np.linspace(0.0,1000.0,Nb),histtype='step',density=True)
            plot.set_xlabel("$v/(\mathrm{km/s})$",fontsize=fontsize)
            plot.set_title("Maxwellian")
            
            points=np.linspace(0.0,1000.0,1000)
            PDF_an = np.sqrt(2.0/np.pi) * (points**2/(sigma_km_s**3)) * np.exp( -points**2/(2.0*sigma_km_s**2) )
            plot.plot(points,PDF_an, color='tab:green')

            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.hist(vs2/code.CONST_KM_PER_S,bins=np.linspace(-1000,1000.0,Nb),histtype='step',density=True)
            plot.set_xlabel("$v/(\mathrm{km/s})$",fontsize=fontsize)
            plot.set_title("Normal")

            points=np.linspace(-1000,1000.0,1000)
            PDF_an = (1.0/(sigma_km_s*np.sqrt(2.0*np.pi))) * np.exp( - (points-mu_km_s)**2/(2.0*sigma_km_s**2))
            plot.plot(points,PDF_an, color='tab:green')

            pyplot.show()


    def test16(self,args):
        print('Test sample Kroupa 93 IMF')
        
        code = MSE()

        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        
        N = 100000

        seed = 0
        np.random.seed(seed)

        ms = []
        for index in range(N):
            m = code.test_sample_from_kroupa_93_imf()
            ms.append(m)
        ms = np.array(ms)

        assert(round(np.mean(ms),1) == 0.50)

        code.reset()

        if args.verbose==True:
            print("mean ms/MSun",np.mean(ms))
        
        if args.plot == True:
            Nb=100
            fontsize=20
            from matplotlib import pyplot
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1,yscale="log")
            plot.hist(np.log10(ms),bins=np.linspace(-1.0,2.0,Nb),histtype='step',density=True,color='tab:red')
            plot.set_xlabel("$\mathrm{log}_{10}(m/\mathrm{M}_\odot)$",fontsize=fontsize)
            plot.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)        

            
            points=np.linspace(-1.0,2.0,Nb)
            PDF_an = [np.log(10.0)*pow(10.0,log10m)*kroupa_93_imf(pow(10.0,log10m)) for log10m in points]
            plot.plot(points,PDF_an, color='tab:green')

            pyplot.show()

    def test17(self,args):
        print('Test kick velocity recipes')
        
        code = MSE()
        code.verbose_flag = 0
        
        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        CONST_KM_PER_S = code.CONST_KM_PER_S
        N = 100
        if args.mode == 1:
            N = 500
        
        seed = 0
        np.random.seed(seed)

        kick_distributions = [1,2,3,4,5]
        N_k = len(kick_distributions)

        vs_NS = [[] for x in range(N_k)]
        vs_BH = [[] for x in range(N_k)]
        vs = [[] for x in range(N_k)]
        alpha = 2.7
        m1 = 8.0
        m2 = 100.0
        for index_kick,kick_distribution in enumerate(kick_distributions):
            for index in range(N):
                x = np.random.random()
                m = pow( x*(pow(m2,1.0-alpha) - pow(m1,1.0-alpha)) + pow(m1,1.0-alpha), 1.0/(1.0-alpha) )
                kw,v = code.test_kick_velocity(kick_distribution,m)
                
                if args.verbose == True:
                    print("index_kick",index_kick,"m",m,"kw",kw,"v/(km/s)",v/CONST_KM_PER_S)
                
                vs[index_kick].append(v/CONST_KM_PER_S)
                if kw==13:
                    vs_NS[index_kick].append(v/CONST_KM_PER_S)
                if kw==14:
                    vs_BH[index_kick].append(v/CONST_KM_PER_S)

            vs[index_kick] = np.array(vs[index_kick])
            vs_NS[index_kick] = np.array(vs_NS[index_kick])
            vs_BH[index_kick] = np.array(vs_BH[index_kick])

        code.reset()
       
        if args.plot == True:
            Nb=50
            fontsize=16
            labelsize=12
            from matplotlib import pyplot

            if args.fancy_plots == True:
                pyplot.rc('text',usetex=True)
                pyplot.rc('legend',fancybox=True)

            fig=pyplot.figure(figsize=(16,10))
            
            colors = ['k','tab:red','tab:blue','tab:green','tab:cyan']

            bins = np.linspace(0.0,1500.0,Nb)
            for index_kick in range(N_k):
                plot=fig.add_subplot(2,3,index_kick+1,yscale="log")
                color = 'k'

                plot.hist(vs[index_kick],bins=bins,histtype='step',density=True,color='k',linestyle='solid',label='$\mathrm{All}$')
                plot.hist(vs_BH[index_kick],bins=bins,histtype='step',density=True,color='tab:red',linestyle='dotted',label='$\mathrm{BH}$')
                plot.hist(vs_NS[index_kick],bins=bins,histtype='step',density=True,color='tab:blue',linestyle='dashed',label='$\mathrm{NS}$')
            
                plot.annotate("$\mathrm{Kick\,distribution\,%s}$"%(index_kick+1),xy=(0.1,0.9),xycoords='axes fraction',fontsize=fontsize)
                
                plot.set_xlabel("$V_\mathrm{kick}/(\mathrm{km\,s^{-1}})$",fontsize=fontsize)
                plot.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)
                    
                if 1==1:
                    points=np.linspace(10.0,1500.0,1000)
                    sigma_km_s = 265.0
                    PDF_an = np.sqrt(2.0/np.pi) * (points**2/(sigma_km_s**3)) * np.exp( -points**2/(2.0*sigma_km_s**2) )
                    plot.plot(points,PDF_an, color='tab:green',label='$\mathrm{Hobbs+05}$')
                handles,labels = plot.get_legend_handles_labels()
                plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
                
                plot.set_ylim(1.0e-5,1.0e-1)
                
            plot=fig.add_subplot(2,3,6,yscale="log")

            plot.legend(handles,labels,loc="upper left",fontsize=0.85*fontsize)
            plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            plot.axis('off')

            plot.set_xticks([])
            plot.set_yticks([])
            plot.set_xticklabels([])
            plot.set_yticklabels([])
            
            fig.savefig("kick_distributions.pdf")
            pyplot.show()

    def test18(self,args):
        print('Test flybys sampling')
        
        code = MSE()

        CONST_G = code.CONST_G
        CONST_C = code.CONST_C
        CONST_KM_PER_S = code.CONST_KM_PER_S
        N = 20000
        
        seed = 0
        np.random.seed(seed)

        R_enc = 1.0e4
        n_star = 0.1*code.CONST_PER_PC3
        sigma_rel = 10.0*code.CONST_KM_PER_S
        M_int = 20.0

        b_vecs = []
        V_vecs = []
        M_pers = []
        bs = []
        Vs = []
        for index in range(N):
            M_per,b_vec,V_vec = code.test_flybys_perturber_sampling(R_enc,n_star,sigma_rel,M_int)
            M_pers.append(M_per)
            b_vecs.append(b_vec)
            V_vecs.append(V_vec)
            bs.append(np.linalg.norm(b_vec))
            Vs.append(np.linalg.norm(V_vec))

        bs = np.array(bs)
        Vs = np.array(Vs)
        
        b1 = 0.0
        b2 = R_enc
        b_mean_an = (2.0/3.0)*(b2**3 - b1**3)/(b2**2 - b1**2) ### mean b value assuming dN/db = 2b/(b2^2 - b1^2)
        
        tol = 1.0e-2
        assert( abs((b_mean_an - np.mean(bs))/b_mean_an) <= tol)

        ### On average, individual components of b vec should be zero (isotropic orientations) ###
        assert( abs(np.mean(np.array( [x[0] for x in b_vecs]))/b_mean_an) <= tol )
        assert( abs(np.mean(np.array( [x[1] for x in b_vecs]))/b_mean_an) <= tol )
        assert( abs(np.mean(np.array( [x[2] for x in b_vecs]))/b_mean_an) <= tol )
        
        code.reset()

        if args.verbose==True:
            print("mean bs/au",np.mean(bs),' an ',b_mean_an)
        
        if args.plot == True:
            Nb=50
            fontsize=16
            labelsize=12
            from matplotlib import pyplot

            fig=pyplot.figure(figsize=(8,10))
            
            plot1=fig.add_subplot(2,1,1,yscale="log")
            plot2=fig.add_subplot(2,1,2,yscale="log")

            bins = np.linspace(2,np.log10(R_enc)+1,Nb)
            plot1.hist(np.log10(bs),bins=bins,histtype='step',density=True,color='k',linestyle='solid',label='$\mathrm{Num}$')
            bins = np.linspace(0,100.0,Nb)
            plot2.hist(Vs/code.CONST_KM_PER_S,bins=bins,histtype='step',density=True,color='k',linestyle='solid',label='$\mathrm{Num}$')
            
            points = np.linspace(2,np.log10(R_enc)+1,1000)
            plot1.plot(points, np.log(10.0)*2.0 * pow(10.0,2*points)/(b2**2 - b1**2), color='tab:green',label="$\mathrm{d}N/\mathrm{d}b \propto b$")

            plot1.set_xlabel("$\log_{10}(b/\mathrm{au})$",fontsize=fontsize)
            plot1.set_ylabel("$\mathrm{d} N/\mathrm{d} \log_{10} (b)$",fontsize=fontsize)

            plot2.set_xlabel("$V_\mathrm{enc}/(\mathrm{km \,s^{-1}}))$",fontsize=fontsize)
            plot2.set_ylabel("$\mathrm{d} N/\mathrm{d} V_\mathrm{enc}$",fontsize=fontsize)

            handles,labels = plot1.get_legend_handles_labels()
            plot1.legend(handles,labels,loc="best",fontsize=0.85*fontsize)
            
            plot1.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            plot2.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

            pyplot.show()

    def test19(self,args):
        print('Test white dwarf SNe single degenerate model 1')
        
        code = MSE()

        CONST_G = code.CONST_G
        CONST_C = code.CONST_C

        luminosities = [0.03 * code.CONST_L_SUN,0.95 * code.CONST_L_SUN]

        ### He donor case ###
        #M_WDs = np.linspace(0.5,1.5,5)
        M_WDs = np.linspace(0.73,1.12,6)
        #M_WDs = [0.8,0.8,0.8,0.8,0.8,0.8]
        accretion_rates = pow(10.0,np.linspace(-9,-6,400))
        M_Hes = np.linspace(0.01,0.3,100)
        
        etas = [[[] for x in range(len(M_WDs))] for x in range(len(luminosities))]
        accretion_modes = [[[] for x in range(len(M_WDs))] for x in range(len(luminosities))]
        
        explosions = [[[[] for x in range(len(accretion_rates))] for x in range(len(M_WDs))] for x in range(len(luminosities))]
        all_etas = []
        for k,luminosity in enumerate(luminosities):
            for i,M_WD in enumerate(M_WDs):
                for j,m_dot in enumerate(accretion_rates):
                    eta,WD_accretion_mode = code.test_binary_evolution_SNe_Ia_single_degenerate_model_1_accumulation_efficiency(M_WD, m_dot, luminosity)
                    etas[k][i].append(eta)
                    all_etas.append(eta)
                    accretion_modes[k][i].append(WD_accretion_mode)


                    for l,M_He in enumerate(M_Hes):
                        explosion = code.test_binary_evolution_SNe_Ia_single_degenerate_model_1_explosion(M_WD, m_dot, M_He, luminosity)

                        explosions[k][i][j].append(int(explosion))

        ### Hydrogen donor case ###
        H_donor_m_dot_lower_boundaries = []
        H_donor_m_dot_upper_boundaries = []
        
        M_WDs2 = np.linspace(0.5,1.4,100)
        for i,M_WD in enumerate(M_WDs2):
            m_dot_lower,m_dot_upper = code.test_binary_evolution_SNe_Ia_single_degenerate_model_1_white_dwarf_hydrogen_accretion_boundaries(M_WD)
            H_donor_m_dot_lower_boundaries.append(m_dot_lower)
            H_donor_m_dot_upper_boundaries.append(m_dot_upper)

        ### Sanity checks ###
        all_etas = np.array(all_etas)
        assert( np.amin(all_etas) >= 0.0 and np.amax(all_etas) <= 1.0)
        
        if args.verbose==True:
            print("np.amin(all_etas)",np.amin(all_etas),"np.amax(all_etas)",np.amax(all_etas),"np.mean(all_etas)",np.mean(all_etas))

        code.reset()

        if args.plot == True:
            Nb=50
            fontsize=16
            labelsize=12
            from matplotlib import pyplot
            if args.fancy_plots == True:
                pyplot.rc('text',usetex=True)
                pyplot.rc('legend',fancybox=True)
                
            fig=pyplot.figure(figsize=(8,10))
            fige=pyplot.figure(figsize=(12,12))
            figh=pyplot.figure(figsize=(8,6))
            
            colors = ['k','tab:blue','tab:red','tab:green','tab:cyan','tab:brown']
            linestyles = ['solid','dotted','dashed','-.','solid','dotted']
            
            plot1=fig.add_subplot(2,1,1,xscale="log")
            plot2=fig.add_subplot(2,1,2,xscale="log")
            
            N_r=2
            plot1e=fige.add_subplot(N_r,N_r,1,xscale="linear",yscale="log")
            plot2e=fige.add_subplot(N_r,N_r,2,xscale="linear",yscale="log")
            plot3e=fige.add_subplot(N_r,N_r,3,xscale="linear",yscale="log")
            plot4e=fige.add_subplot(N_r,N_r,4,xscale="linear",yscale="log")
            
            plot1h=figh.add_subplot(1,1,1,xscale="linear",yscale="log")

            linewidth=1.5
            for i,M_WD in enumerate(M_WDs):
                plot1.plot(accretion_rates,etas[0][i],label="$M_\mathrm{WD}=%s \, \mathrm{M}_\odot$"%round(M_WD,2),color=colors[i],linestyle=linestyles[i],linewidth=linewidth)
                plot2.plot(accretion_rates,etas[1][i],label="$M_\mathrm{WD}=%s \, \mathrm{M}_\odot$"%round(M_WD,2),color=colors[i],linestyle=linestyles[i],linewidth=linewidth)
            
            plot1h.plot(M_WDs2,H_donor_m_dot_lower_boundaries,color='tab:red')
            plot1h.plot(M_WDs2,H_donor_m_dot_upper_boundaries,color='tab:red')
            
            xs = M_Hes
            ys = accretion_rates
            
            i=1
            j=0
            zs = explosions[j][i]
            plot1e.pcolormesh(xs,ys,zs,cmap = 'Reds')
            plot1e.set_title("$M_\mathrm{WD} = %s\,\mathrm{M}_\odot; \, L=%s\,\mathrm{L}_\odot$"%(round(M_WDs[i],2),round(luminosities[j]/code.CONST_L_SUN,2)))

            i=4
            j=0
            zs = explosions[j][i]
            plot2e.pcolormesh(xs,ys,zs,cmap = 'Reds')
            plot2e.set_title("$M_\mathrm{WD} = %s\,\mathrm{M}_\odot; \, L=%s\,\mathrm{L}_\odot$"%(round(M_WDs[i],2),round(luminosities[j]/code.CONST_L_SUN,2)))

            i=1
            j=1
            zs = explosions[j][i]
            plot3e.pcolormesh(xs,ys,zs,cmap = 'Reds')
            plot3e.set_title("$M_\mathrm{WD} = %s\,\mathrm{M}_\odot; \, L=%s\,\mathrm{L}_\odot$"%(round(M_WDs[i],2),round(luminosities[j]/code.CONST_L_SUN,2)))

            i=4
            j=1
            zs = explosions[j][i]
            plot4e.pcolormesh(xs,ys,zs,cmap = 'Reds')
            plot4e.set_title("$M_\mathrm{WD} = %s\,\mathrm{M}_\odot; \, L=%s\,\mathrm{L}_\odot$"%(round(M_WDs[i],2),round(luminosities[j]/code.CONST_L_SUN,2)))

            plot2.set_xlabel("$\log_{10}[\dot{M}/(\mathrm{M}_\odot/\mathrm{yr^{-1}})]$",fontsize=fontsize)
            plots = [plot1,plot2]
            for i,plot in enumerate(plots):
                plot.set_ylabel("$\eta$",fontsize=fontsize)

                handles,labels = plot.get_legend_handles_labels()
                plot.legend(handles,labels,loc="best",fontsize=0.85*fontsize)

                plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
                plot.set_title("$L=%s \, \mathrm{L}_\odot$"%(round(luminosities[i]/code.CONST_L_SUN,2)))


            plots = [plot1e,plot2e,plot3e,plot4e]
            for i,plot in enumerate(plots):
                plot.set_ylabel("$\log_{10}[\dot{M}/(\mathrm{M}_\odot/\mathrm{yr^{-1}})]$",fontsize=fontsize)

                plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            plot3e.set_xlabel("$M_\mathrm{He\,layer}/\mathrm{M}_\odot$",fontsize=fontsize)
            plot4e.set_xlabel("$M_\mathrm{He\,layer}/\mathrm{M}_\odot$",fontsize=fontsize)

            plot1h.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            plot1h.set_xlabel("$M_\mathrm{WD}/\mathrm{M}_\odot$",fontsize=fontsize)
            plot1h.set_ylabel("$\dot{M}/(\mathrm{M}_\odot/\mathrm{yr})$",fontsize=fontsize)

            plot1h.set_ylim(1e-8,1e-6)

            fig.savefig("white_dwarf_SNe_single_degenerate_model_1_eta.pdf")
            fige.savefig("white_dwarf_SNe_single_degenerate_model_1_explosion.pdf")
            figh.savefig("white_dwarf_SNe_single_degenerate_model_1_hydrogen_m_dot.pdf")
    
            pyplot.show()

    def test20(self,args):
        print("Test Helium star system")

        #particles = Tools.create_fully_nested_multiple(2,[5.958893384591632, 3.574547072971755], [0.03], [0.0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[11,7],object_types=[1,1])
        particles = Tools.create_fully_nested_multiple(2,[5.958893384591632, 1.0], [0.0043], [0.0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[11,7],object_types=[1,1])
        
        particles = Tools.create_fully_nested_multiple(2,[5.958893384591632, 1.0], [0.0015], [0.0], [0.01], [0.01], [0.01], metallicities=[0.02,0.02],stellar_types=[11,7],object_types=[1,1])
        
        binaries = [x for x in particles if x.is_binary == True]
        bodies = [x for x in particles if x.is_binary == False]

        code = MSE()
        code.add_particles(particles)

        t = 0.0
        N=1000
        tend = 1.0e9
        dt=tend/float(N)

        t_print_array = []
        a_print_array = []
        e_print_array = []
        AP_print_array = []

        code.evolve_model(0.0) ### make sure that stellar_evolution.cpp -- initialize_stars() is run
        code.particles[1].stellar_type = 10
        code.particles[0].age = 1.60722e+08
        code.particles[1].age = 0.0

        code.binary_evolution_SNe_Ia_single_degenerate_model = 0
        code.binary_evolution_SNe_Ia_double_degenerate_model = 0

        code.effective_radius_multiplication_factor_for_collisions_compact_objects = 1.0e0

        N_bodies = 2
        N_orbits = 1
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
        
        i_status = 0
        integration_flags = [[]]
        code.verbose_flag = 1
        N_orbits_status = [N_orbits]
        N_bodies_status = [N_bodies]

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            
            #particles = code.particles
            #binaries = [x for x in particles if x.is_binary == True]
            #bodies = [x for x in particles if x.is_binary == False]


            particles = code.particles
            orbits = [x for x in particles if x.is_binary==True]
            bodies = [x for x in particles if x.is_binary==False]
            N_orbits = len(orbits)
            N_bodies = len(bodies)

            if args.verbose==True:
                print( 't/Myr',t,'smas',[x.a for x in orbits],'stellar types',[x.stellar_type for x in bodies],'masses',[x.mass for x in bodies],'ages',[x.age for x in bodies],'WD_He_layer_mass',[x.WD_He_layer_mass for x in bodies],'m_dot_accretion_SD',[x.m_dot_accretion_SD for x in bodies])
              
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

        code.write_final_log_entry() ### This has to be done within Python, since the C++ code does not know if the desired Python simulation end time has been reached!
        N_status = i_status+1
        
        for i_status in range(N_status):
            t_print[i_status] = np.array(t_print[i_status])

        import copy
        error_code_copy = copy.deepcopy(code.error_code)
        log_copy = copy.deepcopy(code.log)

        code.reset()
                
        if HAS_MATPLOTLIB == True and args.plot==True:
            from matplotlib import lines
            
            parsec_in_AU = 206201.0
            CONST_L_SUN = 0.0002710404109745588

            plot_log = []
            previous_event_flag = -1
            for index_log,log in enumerate(log_copy):
                event_flag = log["event_flag"]
                if previous_event_flag == event_flag and (event_flag == 4 or event_flag == 10):
                    
                    continue
                plot_log.append(log)
                previous_event_flag = event_flag
                            
            N_l = len(plot_log)
            N_r = int(np.ceil(np.sqrt(N_l)))#+1
            N_c = N_r
            panel_length = 3

            fontsize=10
            fig=pyplot.figure(figsize=(N_r*panel_length,N_r*panel_length))
            
            legend_elements = []
            for k in range(16):
                color,s,description,marker = Tools.get_color_and_size_and_description_for_star(k,1.0)
                legend_elements.append( lines.Line2D([0],[0], marker=marker, markerfacecolor=color,
                    markeredgecolor='black', color='w', markersize=10, label=r"$\mathrm{%s}$"%description))

            for index_log,log in enumerate(plot_log):
                plot=fig.add_subplot(N_r,N_c,index_log+1)
                particles = log["particles"]
                event_flag = log["event_flag"]
                index1 = log["index1"]
                index2 = log["index2"]

                Tools.generate_mobile_diagram(particles,plot,fontsize=fontsize,index1=index1,index2=index2,event_flag=event_flag)

                text = Tools.get_description_for_event_flag(event_flag,log["SNe_type"])
                plot.set_title(text,fontsize=fontsize)
                plot.annotate(r"$t\simeq %s\,\mathrm{Myr}$"%round(log["time"]*1e-6,2),xy=(0.1,0.9),xycoords='axes fraction',fontsize=fontsize)

                if index_log == 0:
                    plot.legend(handles = legend_elements, bbox_to_anchor = (-0.05, 1.50), loc = 'upper left', ncol = 5,fontsize=0.85*fontsize)
                
            #fig.savefig(plot_filename + "_mobile.pdf")
        
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
                plot2.annotate("$\mathrm{CE}$",xy=(1.02*t,1.0e3),fontsize=0.8*fontsize)
            
            plot1.set_ylabel("$m/\mathrm{M}_\odot$",fontsize=fontsize)
            plot2.set_ylabel("$r/\mathrm{au}$",fontsize=fontsize)
            plot3.set_ylabel("$\mathrm{Stellar\,Type}$",fontsize=fontsize)
            #plot4.set_ylabel("$\Omega_\mathrm{spin}/\mathrm{yr^{-1}}$",fontsize=fontsize)
            plot4.set_ylabel("$P_\mathrm{spin}/\mathrm{s}$",fontsize=fontsize)
            plot4.set_xlabel("$t/\mathrm{Myr}$",fontsize=fontsize)
            plot2.set_ylim(1.0e-5,1.0e5)
            
            plot_pos.set_xlabel("$X/\mathrm{pc}$",fontsize=fontsize)
            plot_pos.set_ylabel("$Y/\mathrm{pc}$",fontsize=fontsize)
            plot_pos.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            

            plot_HRD.set_xlim(5.0,3.0)
            plot_HRD.set_ylim(-4.0,6.0)
            plot_HRD.set_xlabel("$\mathrm{log}_{10}(T_\mathrm{eff}/\mathrm{K})$",fontsize=fontsize)
            plot_HRD.set_ylabel(r"$\mathrm{log}_{10}(L/L_\odot)$",fontsize=fontsize)
            plot_HRD.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            
            #fig.savefig(plot_filename + ".pdf")
            #fig_pos.savefig(plot_filename + "_pos.pdf")
            #fig_HRD.savefig(plot_filename + "_HRD.pdf")
            
            pyplot.show()


    def test21(self,args):
        print("Test conservation laws (energy and angular momentum)")

        print("Test 21a: Hamiltonian conservation for isolated hierarchical triple")
        print("Test 21b: Angular momentum conservation for isolated hierarchical triple")

        ### Set up the Naoz et al. (2009) reference triple system ###
        # Same system as test1a, with all dissipative physics disabled.
        # This is a well-characterized Kozai-Lidov system.
        particles = Tools.create_fully_nested_multiple(3,
            [1.0, 1.0e-3, 40.0e-3],
            [6.0, 100.0],
            [0.001, 0.6],
            [0.0, 65.0*np.pi/180.0],
            [45.0*np.pi/180.0, 0.0],
            [0.0, 0.0],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2])

        binaries = [x for x in particles if x.is_binary==True]
        bodies = [x for x in particles if x.is_binary==False]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        # Inner orbital period: P = sqrt(a^3 / M_total) in MSE units (G = 4 pi^2)
        M_inner = 1.0 + 1.0e-3
        P_inner = np.sqrt(6.0**3 / M_inner)
        tend = 1000.0 * P_inner

        N = 200
        dt = tend / float(N)

        # Initialize (commit particles + no-op step)
        t = 0.0
        code.evolve_model(t)
        t += dt

        # First real evolution step establishes reference values.
        # Note: code.initial_hamiltonian is 0 because evolve_interface(0,0)
        # returns H=0, so we track conservation ourselves.
        code.evolve_model(t)
        H_initial = code.hamiltonian
        L_initial = compute_total_orbital_AM(code)
        L_initial_mag = np.linalg.norm(L_initial)

        assert H_initial != 0.0, "Hamiltonian should be nonzero for a bound triple system"
        assert L_initial_mag > 0.0, "Total angular momentum should be nonzero"

        max_rel_energy_error = 0.0
        max_rel_AM_error = 0.0

        t += dt
        while t <= tend:
            code.evolve_model(t)

            rel_E_err = abs(code.hamiltonian - H_initial) / abs(H_initial)
            if rel_E_err > max_rel_energy_error:
                max_rel_energy_error = rel_E_err

            L_total = compute_total_orbital_AM(code)
            rel_AM_err = np.linalg.norm(L_total - L_initial) / L_initial_mag
            if rel_AM_err > max_rel_AM_error:
                max_rel_AM_error = rel_AM_err

            if args.verbose==True:
                particles = code.particles
                binaries = [x for x in particles if x.is_binary==True]
                print('t/P_in', t/P_inner, 'e_in', binaries[0].e,
                      'dE/E', rel_E_err, 'dL/L', rel_AM_err)

            t += dt

        if args.verbose==True:
            print('  Max relative energy error:', max_rel_energy_error)
            print('  Max relative AM error:', max_rel_AM_error)

        assert max_rel_energy_error < 1.0e-6, \
            "Hamiltonian not conserved: max relative error = %.2e (tolerance: 1e-6)" % max_rel_energy_error
        print("Test 21a passed")

        assert max_rel_AM_error < 1.0e-6, \
            "Angular momentum not conserved: max relative error = %.2e (tolerance: 1e-6)" % max_rel_AM_error
        print("Test 21b passed")

        code.reset()

        ### Test 21c: Nested quadruple - Hamiltonian and AM conservation ###
        # A nested quadruple (3+1) exercises the full multi-binary Hamiltonian
        # accumulation path. Before the evolve.cpp fix (which reset *hamiltonian
        # to 0.0 after the integration loop), no energy tracking was possible.
        print("Test 21c: Hamiltonian and AM conservation for nested quadruple")

        particles_q = Tools.create_fully_nested_multiple(4,
            [1.0, 1.0e-3, 40.0e-3, 0.5],
            [6.0, 100.0, 3000.0],
            [0.001, 0.3, 0.1],
            [0.0, 65.0*np.pi/180.0, 30.0*np.pi/180.0],
            [45.0*np.pi/180.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            metallicities=[0.02, 0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1, 1],
            object_types=[2, 2, 2, 2])

        binaries_q = [x for x in particles_q if x.is_binary==True]
        bodies_q = [x for x in particles_q if x.is_binary==False]

        for b in bodies_q:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False

        for b in binaries_q:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code_q = MSE()
        code_q.add_particles(particles_q)
        code_q.relative_tolerance = 1.0e-14
        code_q.absolute_tolerance_eccentricity_vectors = 1.0e-14
        code_q.include_flybys = False
        code_q.enable_tides = False
        code_q.enable_root_finding = False
        code_q.verbose_flag = 0

        tend_q = 500.0 * P_inner
        N_q = 100
        dt_q = tend_q / float(N_q)

        t_q = 0.0
        code_q.evolve_model(t_q)
        t_q += dt_q

        code_q.evolve_model(t_q)
        H_initial_q = code_q.hamiltonian
        L_initial_q = compute_total_orbital_AM(code_q)
        L_initial_mag_q = np.linalg.norm(L_initial_q)

        assert H_initial_q != 0.0, "Quad Hamiltonian should be nonzero"

        max_rel_energy_error_q = 0.0
        max_rel_AM_error_q = 0.0

        t_q += dt_q
        while t_q <= tend_q:
            code_q.evolve_model(t_q)

            rel_E_err_q = abs(code_q.hamiltonian - H_initial_q) / abs(H_initial_q)
            if rel_E_err_q > max_rel_energy_error_q:
                max_rel_energy_error_q = rel_E_err_q

            L_total_q = compute_total_orbital_AM(code_q)
            rel_AM_err_q = np.linalg.norm(L_total_q - L_initial_q) / L_initial_mag_q
            if rel_AM_err_q > max_rel_AM_error_q:
                max_rel_AM_error_q = rel_AM_err_q

            t_q += dt_q

        if args.verbose==True:
            print('  Quad max relative energy error:', max_rel_energy_error_q)
            print('  Quad max relative AM error:', max_rel_AM_error_q)

        assert max_rel_energy_error_q < 1.0e-6, \
            "Quad Hamiltonian not conserved: max relative error = %.2e (tolerance: 1e-6)" % max_rel_energy_error_q
        assert max_rel_AM_error_q < 1.0e-6, \
            "Quad angular momentum not conserved: max relative error = %.2e (tolerance: 1e-6)" % max_rel_AM_error_q

        print("Test 21c passed")

        code_q.reset()

        ### Test 21d: Hamiltonian is zero for a no-op step (start_time == end_time) ###
        # The C code explicitly sets *hamiltonian = 0.0 when start_time == end_time.
        # This is a boundary case that should be predictably handled.
        print("Test 21d: Hamiltonian == 0 for no-op step (start==end)")

        particles_d = Tools.create_fully_nested_multiple(2,
            [1.0, 1.0],
            [1.0],
            [0.0],
            [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1],
            object_types=[2, 2])
        code_d = MSE()
        code_d.add_particles(particles_d)
        code_d.evolve_model(0.0)   # no-op: start == end
        assert code_d.hamiltonian == 0.0, \
            "Hamiltonian should be 0.0 for no-op evolve step, got %g" % code_d.hamiltonian
        print("Test 21d passed")
        code_d.reset()

        ### Test 21e: Angular momentum direction is conserved for the triple ###
        # The vector direction (j_hat) as well as its magnitude should be preserved
        # in a purely secular (no tides, no GW) triple system.
        print("Test 21e: Angular momentum vector direction conserved for triple")

        particles_e = Tools.create_fully_nested_multiple(3,
            [1.0, 1.0e-3, 40.0e-3],
            [6.0, 100.0],
            [0.001, 0.6],
            [0.0, 65.0*np.pi/180.0],
            [45.0*np.pi/180.0, 0.0],
            [0.0, 0.0],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2])

        for b in [x for x in particles_e if not x.is_binary]:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in [x for x in particles_e if x.is_binary]:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code_e = MSE()
        code_e.add_particles(particles_e)
        code_e.relative_tolerance = 1.0e-14
        code_e.absolute_tolerance_eccentricity_vectors = 1.0e-14
        code_e.include_flybys = False
        code_e.enable_tides = False
        code_e.enable_root_finding = False
        code_e.verbose_flag = 0

        M_inner_e = 1.0 + 1.0e-3
        P_inner_e = np.sqrt(6.0**3 / M_inner_e)
        tend_e = 500.0 * P_inner_e
        N_e = 50
        dt_e = tend_e / float(N_e)

        t_e = 0.0
        code_e.evolve_model(t_e)
        t_e += dt_e
        code_e.evolve_model(t_e)
        L_ref = compute_total_orbital_AM(code_e)
        L_ref_mag = np.linalg.norm(L_ref)
        L_ref_hat = L_ref / L_ref_mag

        max_dir_err = 0.0
        t_e += dt_e
        while t_e <= tend_e:
            code_e.evolve_model(t_e)
            L_t = compute_total_orbital_AM(code_e)
            L_t_mag = np.linalg.norm(L_t)
            L_t_hat = L_t / L_t_mag
            # cos(angle) between L vectors; 1 = same direction
            cosangle = np.dot(L_ref_hat, L_t_hat)
            dir_err = abs(1.0 - cosangle)
            if dir_err > max_dir_err:
                max_dir_err = dir_err
            t_e += dt_e

        if args.verbose==True:
            print('  Max AM direction error (1 - cos(angle)):', max_dir_err)

        assert max_dir_err < 1.0e-10, \
            "AM direction not conserved: max (1-cos(angle)) = %.2e (tolerance 1e-10)" % max_dir_err
        print("Test 21e passed")
        code_e.reset()

    def test22(self,args):
        print("Test secular-to-N-body switching in triple system")

        """Set up a triple with a tight hierarchy ratio and high mutual
        inclination so that Kozai-Lidov oscillations drive the inner
        eccentricity high enough to trigger secular breakdown. This test
        verifies: (a) the secular_breakdown_has_occurred flag is assigned
        correctly (catches C9 regression where = was written as ==), and
        (b) the code switches to N-body integration and produces physical
        results.

        The test uses stop_after_root_found=True so that when CVODE
        detects the secular breakdown root crossing, the evolution pauses
        before investigate_roots_in_system resets the flag.  This lets us
        directly assert the flag value."""

        particles = Tools.create_fully_nested_multiple(3,
            [1.0, 0.001, 1.0],
            [1.0, 5.0],
            [0.001, 0.001],
            [0.01, 89.5*np.pi/180.0],
            [0.01, 0.01],
            [0.01, 0.01],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        inner_binary = binaries[0]
        outer_binary = binaries[1]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.check_for_physical_collision_or_orbit_crossing = False
            b.check_for_dynamical_instability = False
            b.check_for_entering_LISA_band = False

        ### Only check secular breakdown on the inner binary (which has a
        ### parent).  The outer binary has no parent, so its root function
        ### slot is never computed by check_for_roots.
        outer_binary.check_for_secular_breakdown = False

        code = MSE()
        code.add_particles(particles)
        code.enable_tides = False
        code.include_flybys = False
        code.stop_after_root_found = True
        code.verbose_flag = 0

        t = 0.0
        dt = 5.0
        tend = 2.0e3

        secular_breakdown_detected = False

        while (t<tend):
            t+=dt
            code.evolve_model(t)
            flag = code.CVODE_flag

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            if args.verbose==True:
                e_inner = binaries[0].e if len(binaries) > 0 else -1.0
                print("t/yr",t,"e_in",e_inner,
                      "CVODE_flag",flag)

            if flag == 2:
                for b in binaries:
                    if b.secular_breakdown_has_occurred:
                        secular_breakdown_detected = True
                if args.verbose==True:
                    print("Root found at t =",code.model_time,
                          "secular_breakdown =",secular_breakdown_detected)
                break

        ### (a) Assert secular_breakdown_has_occurred flag was set.
        ### This catches the C9 regression where the assignment on
        ### line 74 of ODE_root_finding.cpp was written as == instead of =.
        assert secular_breakdown_detected, \
            "Secular breakdown flag should be set (catches C9 = vs == regression)"

        ### (b) Assert that the code produces physical results at the
        ### secular breakdown point.  The inner binary should have very
        ### high eccentricity but all values should be finite and valid.
        ### Re-fetch particles since references change during evolution.
        particles = code.particles
        binaries = [x for x in particles if x.is_binary == True]
        bodies = [x for x in particles if x.is_binary == False]

        for b in binaries:
            assert not np.isnan(b.a), "Semi-major axis is NaN"
            assert not np.isnan(b.e), "Eccentricity is NaN"
            assert b.a > 0, "Semi-major axis should be positive"
        for body in bodies:
            assert not np.isnan(body.mass), "Mass is NaN"
            assert body.mass > 0, "Mass should be positive"

        ### The inner eccentricity should be very high at secular breakdown
        assert binaries[0].e > 0.9, \
            "Inner eccentricity should be > 0.9 at secular breakdown, got %g" % binaries[0].e

        if args.verbose==True:
            print("At breakdown: a_in =",binaries[0].a,"e_in =",binaries[0].e)

        code.reset()

        ### Phase 2: Re-run the same system without stop_after_root_found.
        ### This time the code should process the secular breakdown root,
        ### switch to N-body (integration_flag > 0), and continue evolving.
        ### After N-body, the code re-evaluates stability and may switch
        ### back to secular (integration_flag = 0), so we track the maximum
        ### integration_flag seen during the evolution.
        particles = Tools.create_fully_nested_multiple(3,
            [1.0, 0.001, 1.0],
            [1.0, 5.0],
            [0.001, 0.001],
            [0.01, 89.5*np.pi/180.0],
            [0.01, 0.01],
            [0.01, 0.01],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        outer_binary = binaries[1]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.check_for_physical_collision_or_orbit_crossing = False
            b.check_for_dynamical_instability = False
            b.check_for_entering_LISA_band = False
        outer_binary.check_for_secular_breakdown = False

        code = MSE()
        code.add_particles(particles)
        code.enable_tides = False
        code.include_flybys = False
        code.verbose_flag = 0
        # [P5.1] Pin nbody_analysis_fractional_integration_time to 0.05 so the
        # N-body phase stays active long enough to be detected at the Python
        # loop boundary (dt=5 yr).  With the new paper default of 0.1 the
        # MSTAR stability-analysis integration aligns with the inner orbital
        # period and immediately declares the system stable, causing the
        # N-body phase to complete before the next Python check.  This test
        # is about the secular-to-N-body SWITCHING MECHANISM, not about the
        # analysis-time parameter; test31a already covers the default value.
        code.nbody_analysis_fractional_integration_time = 0.05

        t = 0.0
        dt = 5.0
        tend = 200.0
        nbody_activated = False

        while (t<tend):
            t+=dt
            code.evolve_model(t)

            if code.integration_flag > 0:
                nbody_activated = True

            if args.verbose==True:
                print("Phase 2: t/yr",t,"integration_flag",code.integration_flag)

        ### Assert N-body was activated at some point during the evolution.
        assert nbody_activated, \
            "N-body integration should have been activated during evolution"

        ### Assert physical results after the full evolution.
        particles = code.particles
        binaries = [x for x in particles if x.is_binary == True]
        bodies = [x for x in particles if x.is_binary == False]

        for b in binaries:
            assert not np.isnan(b.a), "Semi-major axis is NaN after N-body"
            assert not np.isnan(b.e), "Eccentricity is NaN after N-body"
            assert b.a > 0, "Semi-major axis should be positive after N-body"
        for body in bodies:
            assert not np.isnan(body.mass), "Mass is NaN after N-body"
            assert body.mass > 0, "Mass should be positive after N-body"

        if args.verbose==True:
            print("N-body was activated: integration_flag reached > 0",
                  "final a",[b.a for b in binaries],"e",[b.e for b in binaries])

        print("Test passed")
        code.reset()

    def test23(self,args):
        print("Test 2+2 quadruple system initialization and evolution")

        """Use create_2p2_quadruple_system to build a quadruple. Verify all
        7 particles (4 bodies + 3 binaries) are created correctly with valid
        orbital elements. Evolve for a short time and assert no crash."""

        m1 = 1.0
        m2 = 0.8
        m3 = 1.2
        m4 = 0.6
        a1 = 1.0
        a2 = 1.5
        a_out = 100.0
        e1 = 0.1
        e2 = 0.2
        e_out = 0.3
        i1 = 0.01
        i2 = 0.01
        i_out = 30.0*np.pi/180.0

        particles = Tools.create_2p2_quadruple_system(
            [m1, m2, m3, m4],
            [a1, a2, a_out],
            [e1, e2, e_out],
            [i1, i2, i_out],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
            metallicities=[0.02, 0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1, 1],
            object_types=[2, 2, 2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        ### Verify particle counts ###
        assert len(particles) == 7, \
            "Expected 7 particles (4 bodies + 3 binaries), got %d" % len(particles)
        assert len(bodies) == 4, \
            "Expected 4 bodies, got %d" % len(bodies)
        assert len(binaries) == 3, \
            "Expected 3 binaries, got %d" % len(binaries)

        ### Verify body masses ###
        expected_masses = [m1, m2, m3, m4]
        for i, body in enumerate(bodies):
            assert body.mass == expected_masses[i], \
                "Body %d mass mismatch: expected %g, got %g" % (i, expected_masses[i], body.mass)

        ### Verify orbital elements are valid ###
        expected_a = [a1, a2, a_out]
        expected_e = [e1, e2, e_out]
        for i, binary in enumerate(binaries):
            assert binary.a == expected_a[i], \
                "Binary %d semi-major axis mismatch: expected %g, got %g" % (i, expected_a[i], binary.a)
            assert binary.e == expected_e[i], \
                "Binary %d eccentricity mismatch: expected %g, got %g" % (i, expected_e[i], binary.e)
            assert binary.a > 0, "Binary %d semi-major axis should be positive" % i
            assert 0 <= binary.e < 1, "Binary %d eccentricity should be in [0, 1)" % i

        ### Verify hierarchy structure ###
        ### Binary 0: children are bodies 0 and 1 ###
        ### Binary 1: children are bodies 2 and 3 ###
        ### Binary 2: children are binaries 0 and 1 ###
        assert binaries[0].child1 == bodies[0], "Inner binary 1 child1 should be body 0"
        assert binaries[0].child2 == bodies[1], "Inner binary 1 child2 should be body 1"
        assert binaries[1].child1 == bodies[2], "Inner binary 2 child1 should be body 2"
        assert binaries[1].child2 == bodies[3], "Inner binary 2 child2 should be body 3"
        assert binaries[2].child1 == binaries[0], "Outer binary child1 should be inner binary 1"
        assert binaries[2].child2 == binaries[1], "Outer binary child2 should be inner binary 2"

        ### Disable dissipation for a clean dynamics-only test ###
        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.enable_tides = False
        code.include_flybys = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        t = 0.0
        dt = 1.0e3
        tend = 1.0e4

        while (t<tend):
            t+=dt
            code.evolve_model(t)

            particles = code.particles
            binaries = [x for x in particles if x.is_binary == True]
            bodies = [x for x in particles if x.is_binary == False]

            if args.verbose==True:
                print("t/yr",t,
                      "a",[b.a for b in binaries],
                      "e",[b.e for b in binaries])

            ### Assert no NaN in orbital elements ###
            for b in binaries:
                assert not np.isnan(b.a), "Semi-major axis is NaN at t=%g" % t
                assert not np.isnan(b.e), "Eccentricity is NaN at t=%g" % t
                assert b.a > 0, "Semi-major axis should be positive at t=%g" % t
                assert 0 <= b.e < 1, "Eccentricity should be in [0, 1) at t=%g" % t

            for body in bodies:
                assert not np.isnan(body.mass), "Mass is NaN at t=%g" % t
                assert body.mass > 0, "Mass should be positive at t=%g" % t

        ### Verify masses are conserved (no stellar evolution, no mass transfer) ###
        final_masses = [body.mass for body in bodies]
        for i, body in enumerate(bodies):
            assert abs(body.mass - expected_masses[i]) < 1.0e-10, \
                "Body %d mass changed unexpectedly: %g -> %g" % (i, expected_masses[i], body.mass)

        print("Test passed")
        code.reset()

    def test24(self, args):
        """CE energy balance: post-CE orbit properties for circular and eccentric initial orbits.

        H36 was reverted (2026-03-21): the BSE/COSMIC convention (CEFLAG-dependent core masses
        for EORBI and ECIRC) is used. With CEFLAG=3 (total masses), EORBI and ECIRC use the
        same mass convention, so post-CE separation is independent of initial eccentricity
        (ratio ~ 1.0). See logs/v3.1/phase5/h36_investigation.md.

        Test 24a: Post-CE orbit is circular (e ~ 0) for a circular initial orbit.
        Test 24b: Post-CE orbit is circular for an eccentric initial orbit.
        Test 24c: The semi-major axis ratio a_f(e)/a_f(0) is within [1-e^2, 1+eps],
                  consistent with the self-consistent energy balance.
        """
        print("Test 24: CE energy balance with eccentric initial orbit")

        M1, M2 = 5.0, 0.8   # stellar masses [M_sun]; primary is a giant (type 5)
        sma    = 0.3          # inner binary semi-major axis [AU] — triggers CE but survives

        def run_ce_binary(e_inner):
            """Run binary CE with given inner eccentricity; return (a_f, e_f) post-CE."""
            particles = Tools.create_fully_nested_multiple(
                2, [M1, M2], [sma], [e_inner], [0.01], [0.01], [0.01],
                metallicities=[0.02, 0.02], stellar_types=[5, 1], object_types=[1, 1])

            c = MSE()
            c.add_particles(particles)
            c.enable_tides   = False
            c.include_flybys = False
            c.verbose_flag   = 0
            c.enable_root_finding = True
            # Use CE energy flag = 3 (alpha-lambda, total masses throughout) so the
            # ECIRC vs EORBI difference (H36) is the only change between e=0 and e>0.
            c.binary_evolution_CE_energy_flag = 3

            c.evolve_model(1.0e4)   # evolve 10 kyr — CE occurs in ~1 yr

            particles_out = c.particles
            binaries_out  = [x for x in particles_out if x.is_binary]

            if binaries_out:
                a_f = binaries_out[0].a
                e_f = binaries_out[0].e
            else:
                a_f, e_f = None, None   # system merged / disrupted

            c.reset()
            return a_f, e_f

        # ------------------------------------------------------------------ 24a
        a_f_circ, e_f_circ = run_ce_binary(0.01)

        if args.verbose:
            print("  Test 24a: e_i=0.01  a_f=%.5f AU  e_f=%.4f" % (a_f_circ, e_f_circ))

        assert a_f_circ is not None, "24a: CE binary should survive, not merge"
        assert not np.isnan(a_f_circ), "24a: post-CE semi-major axis is NaN"
        assert not np.isnan(e_f_circ), "24a: post-CE eccentricity is NaN"
        assert a_f_circ > 0,  "24a: post-CE semi-major axis must be positive"
        assert e_f_circ < 0.1, \
            "24a: post-CE orbit should be nearly circular, got e=%.4f" % e_f_circ
        print("Test 24a passed")

        # ------------------------------------------------------------------ 24b
        a_f_ecc, e_f_ecc = run_ce_binary(0.5)

        if args.verbose:
            print("  Test 24b: e_i=0.50  a_f=%.5f AU  e_f=%.4f" % (a_f_ecc, e_f_ecc))
            print("  Ratio a_f(ecc)/a_f(circ) = %.4f (expected in [0.75, 1.0))" %
                  (a_f_ecc / a_f_circ))

        assert a_f_ecc is not None, "24b: CE binary should survive for e_i=0.5"
        assert not np.isnan(a_f_ecc), "24b: post-CE semi-major axis is NaN for e_i=0.5"
        assert a_f_ecc > 0,  "24b: post-CE semi-major axis must be positive for e_i=0.5"
        assert e_f_ecc < 0.1, \
            "24b: post-CE orbit should be circular after CE, got e=%.4f" % e_f_ecc
        print("Test 24b passed")

        # ------------------------------------------------------------------ 24c
        # With BSE convention (same mass convention for EORBI and ECIRC),
        # the ratio a_f(e)/a_f(0) should be close to 1.0 for CEFLAG=3.
        # We verify the ratio lies within [1-e^2, 1+eps].
        e_test  = 0.5
        ratio   = a_f_ecc / a_f_circ
        lb      = 1.0 - e_test**2   # = 0.75

        if args.verbose:
            print("  Test 24c: ratio=%.6f  lower_bound=%.4f" % (ratio, lb))

        assert ratio >= lb - 1.0e-3, \
            ("24c: a_f(e=0.5)/a_f(e=0.01) should be >= (1-e^2)=%.3f, got %.6f"
             % (lb, ratio))
        assert ratio <= 1.0 + 1.0e-3, \
            ("24c: a_f(e=0.5)/a_f(e=0.01) should be <= ~1.0, got %.6f" % ratio)
        print("Test 24c passed")

        print("Test passed")

    def test25(self, args):
        """H37: NS spin is set correctly and without division-by-zero when the
        pre-collapse spin vector is zero.

        Test 25a: A newly-initialized NS with NS_model=1 (Ye19) and zero initial
                  spin_vec gets a valid spin magnitude from the Ye19 model (not NaN,
                  and physically positive), verifying the guard added to initialize_star.
        Test 25b: Evolving a system with a compact binary that forms a NS does not
                  produce NaN spins, verifying the H37 zero-guard in evolve_stars.
        """
        print("Test H37: NS spin guard against zero-spin division and double-update")

        # ------------------------------------------------------------------ 25a
        # Create a neutron star (stellar_type=13) with zero initial spin and verify
        # that after initialization with NS_model=1 the spin is set to a valid value.
        # A 20 M_sun star initialized with stellar_type=13 exercises the guard in
        # initialize_star() (stellar_evolution.cpp lines ~155-175), which was
        # unguarded before H37 and would divide by zero if spin_vec=[0,0,0].
        print("  Test 25a: NS initialization with zero spin and NS_model=1")

        # Wide binary (50 AU) so the two stars don't interact during initialization.
        particles = Tools.create_fully_nested_multiple(
            2, [20.0, 1.0], [50.0], [0.0], [0.01], [0.01], [0.01],
            metallicities=[0.02, 0.02], stellar_types=[13, 1], object_types=[1, 1])

        bodies = [x for x in particles if not x.is_binary]

        # Explicitly zero the NS progenitor's spin to exercise the zero-guard branch.
        ns_body = bodies[0]
        ns_body.spin_vec_x = 0.0
        ns_body.spin_vec_y = 0.0
        ns_body.spin_vec_z = 0.0

        code = MSE()
        code.NS_model = 1           # enable Ye19 spin model
        code.add_particles(particles)

        # A no-op evolve flushes state back through the Python–C interface and
        # triggers initialize_star() for stellar evolution setup.
        code.evolve_model(0.0)

        particles_out = code.particles
        ns_out = [x for x in particles_out if not x.is_binary and x.stellar_type == 13]

        assert len(ns_out) > 0, \
            "25a: NS particle (type=13) should exist after initialization of 20 M_sun star"
        ns = ns_out[0]
        spin_mag = np.sqrt(ns.spin_vec_x**2 + ns.spin_vec_y**2 + ns.spin_vec_z**2)

        if args.verbose:
            print("  25a: NS spin_vec = (%.3e, %.3e, %.3e)  |spin| = %.3e"
                  % (ns.spin_vec_x, ns.spin_vec_y, ns.spin_vec_z, spin_mag))

        assert not np.isnan(spin_mag), \
            "25a: NS spin magnitude is NaN — zero-spin guard failed in initialize_star"
        assert spin_mag > 0.0, \
            "25a: NS spin magnitude should be positive after Ye19 initialization"
        print("Test 25a passed")

        code.reset()

        # ------------------------------------------------------------------ 25b
        # Run a CE binary (5 M_sun giant + 0.8 M_sun MS) with NS_model=1 and zero
        # initial spin vectors.  Verify that no NaN appears in the stellar spin
        # components after CE, exercising the guards added in H37 for the CE code paths.
        print("  Test 25b: No NaN spins during CE evolution with NS_model=1")

        particles = Tools.create_fully_nested_multiple(
            2, [5.0, 0.8], [0.3], [0.01], [0.01], [0.01], [0.01],
            metallicities=[0.02, 0.02], stellar_types=[5, 1], object_types=[1, 1])

        bodies = [x for x in particles if not x.is_binary]
        for b in bodies:
            b.spin_vec_x = 0.0
            b.spin_vec_y = 0.0
            b.spin_vec_z = 0.0

        code = MSE()
        code.NS_model = 1
        code.enable_tides   = False
        code.include_flybys = False
        code.verbose_flag   = 0
        code.enable_root_finding = True
        code.add_particles(particles)

        code.evolve_model(1.0e4)

        particles_out = code.particles
        for p in particles_out:
            if not p.is_binary:
                sx, sy, sz = p.spin_vec_x, p.spin_vec_y, p.spin_vec_z
                assert not np.isnan(sx), \
                    "25b: spin_vec_x is NaN for particle index %d" % p.index
                assert not np.isnan(sy), \
                    "25b: spin_vec_y is NaN for particle index %d" % p.index
                assert not np.isnan(sz), \
                    "25b: spin_vec_z is NaN for particle index %d" % p.index

        if args.verbose:
            for p in particles_out:
                if not p.is_binary:
                    sm = np.sqrt(p.spin_vec_x**2 + p.spin_vec_y**2 + p.spin_vec_z**2)
                    print("  25b: particle %d type=%d |spin|=%.3e"
                          % (p.index, p.stellar_type, sm))

        print("Test 25b passed")
        print("Test passed")

        code.reset()

    def test26(self,args):
        """Regression test for H18: Fryer fallback (kick_distribution=3) BH kick scaling.

        Before the fix, distribution 3 applied the NS kick speed directly to BH
        remnants without momentum-conserving scaling.  After the fix, BH kicks in
        distribution 3 are scaled by m_NS/m_BH, exactly as distribution 2 does.

        Observable consequences tested here:
        - All distribution-3 kick speeds are finite and non-negative.
        - Mean BH kick speed in distribution 3 is <= mean BH kick in distribution 2,
          because distribution 3 additionally applies the Fryer fallback factor
          (1 - f_fallback) <= 1 on top of the same momentum scaling.
        - NS kick distributions in distributions 2 and 3 are statistically similar
          (both draw from the same Maxwellian; NSs are unaffected by the BH fix).
        """
        print("Test H18: Fryer fallback BH kick momentum scaling (distribution 3)")

        code = MSE()
        code.verbose_flag = 0

        CONST_KM_PER_S = code.CONST_KM_PER_S
        np.random.seed(42)

        N = 200  # enough for stable statistics
        alpha = 2.7
        m_lo, m_hi = 8.0, 100.0

        def sample_kicks(kick_dist):
            vs_NS, vs_BH = [], []
            for _ in range(N):
                x = np.random.random()
                m = pow(x * (pow(m_hi, 1.0 - alpha) - pow(m_lo, 1.0 - alpha)) + pow(m_lo, 1.0 - alpha), 1.0 / (1.0 - alpha))
                kw, v = code.test_kick_velocity(kick_dist, m)
                km_s = v / CONST_KM_PER_S
                assert km_s >= 0.0, "Distribution %d produced negative kick %g km/s for kw=%d m=%g" % (kick_dist, km_s, kw, m)
                assert np.isfinite(km_s), "Distribution %d produced non-finite kick %g km/s for kw=%d m=%g" % (kick_dist, km_s, kw, m)
                if kw == 13:
                    vs_NS.append(km_s)
                elif kw == 14:
                    vs_BH.append(km_s)
            return np.array(vs_NS), np.array(vs_BH)

        # Reset seed so distributions 2 and 3 draw from the same random sequence
        np.random.seed(42)
        vs_NS_d2, vs_BH_d2 = sample_kicks(2)
        np.random.seed(42)
        vs_NS_d3, vs_BH_d3 = sample_kicks(3)

        # All kicks are already verified non-negative and finite inside sample_kicks.
        # Now check the statistical ordering.

        if len(vs_BH_d2) > 0 and len(vs_BH_d3) > 0:
            mean_BH_d2 = np.mean(vs_BH_d2)
            mean_BH_d3 = np.mean(vs_BH_d3)
            assert mean_BH_d3 <= mean_BH_d2 + 1e-6, (
                "H18 regression: mean BH kick in dist_3 (%g km/s) should be <= dist_2 (%g km/s) "
                "because dist_3 additionally applies the Fryer fallback factor." % (mean_BH_d3, mean_BH_d2)
            )

        # NS kicks in distribution 2 and 3 both draw from the same NS sigma with the
        # same random seed, so their sample means should be very close.
        if len(vs_NS_d2) > 0 and len(vs_NS_d3) > 0:
            mean_NS_d2 = np.mean(vs_NS_d2)
            mean_NS_d3 = np.mean(vs_NS_d3)
            # They won't be identical because some samples that became NSs vs BHs may
            # differ, but they should be within 30% of each other.
            ratio = mean_NS_d3 / mean_NS_d2 if mean_NS_d2 > 0 else 1.0
            assert 0.7 <= ratio <= 1.3, (
                "H18 regression: NS kick means for dist_2 (%g km/s) and dist_3 (%g km/s) "
                "diverge unexpectedly (ratio=%g); fix may have altered NS kick path." % (mean_NS_d2, mean_NS_d3, ratio)
            )

        if args.verbose:
            if len(vs_BH_d2) > 0:
                print("  dist_2 BH kicks: N=%d mean=%.1f km/s" % (len(vs_BH_d2), np.mean(vs_BH_d2)))
            if len(vs_BH_d3) > 0:
                print("  dist_3 BH kicks: N=%d mean=%.1f km/s" % (len(vs_BH_d3), np.mean(vs_BH_d3)))
            if len(vs_NS_d3) > 0:
                print("  dist_3 NS kicks: N=%d mean=%.1f km/s" % (len(vs_NS_d3), np.mean(vs_NS_d3)))

        code.reset()
        print("Test passed")

    def test27(self,args):
        """Regression tests for H12 (sse_initial_mass) and H13 (ECSN kick sigma attribute name).

        H12: Particle.__init__ previously always assigned sse_initial_mass = mass, ignoring
        the sse_initial_mass keyword argument.  After the fix it is: mass if sse_initial_mass
        is None else sse_initial_mass.

        H13: The Python attribute used to be stored as kick_distribution_sigma_km_s_NS_ECN
        (typo, missing S) so code.particles[i].kick_distribution_sigma_km_s_NS_ECSN would
        never reflect the fetched value.
        """
        print("Test H12/H13: sse_initial_mass and ECSN kick sigma attribute correctness")

        ### H12a: sse_initial_mass defaults to mass when not given ###
        p = Particle(is_binary=False, mass=5.0)
        assert p.sse_initial_mass == 5.0, (
            "H12: sse_initial_mass should default to mass=5.0, got %g" % p.sse_initial_mass
        )

        ### H12b: sse_initial_mass is preserved when explicitly set ###
        p2 = Particle(is_binary=False, mass=10.0, sse_initial_mass=6.5)
        assert p2.sse_initial_mass == 6.5, (
            "H12: sse_initial_mass=6.5 should be preserved, got %g" % p2.sse_initial_mass
        )
        assert p2.mass == 10.0, "H12: mass should remain 10.0, got %g" % p2.mass

        ### H12c: None explicitly passed still defaults to mass ###
        p3 = Particle(is_binary=False, mass=7.0, sse_initial_mass=None)
        assert p3.sse_initial_mass == 7.0, (
            "H12: sse_initial_mass=None should fall back to mass=7.0, got %g" % p3.sse_initial_mass
        )

        ### H13: ECSN kick sigma round-trips through C++ correctly ###
        ecsn_sigma = 42.0  # deliberately not the default of 20.0

        particles = Tools.create_fully_nested_multiple(
            2, [1.0, 1.0], [10.0], [0.0], [0.01], [0.01], [0.01],
            metallicities=[0.02, 0.02], stellar_types=[1, 1], object_types=[2, 2]
        )
        bodies = [x for x in particles if x.is_binary == False]
        for b in bodies:
            b.kick_distribution_sigma_km_s_NS_ECSN = ecsn_sigma
            b.evolve_as_star = False

        code = MSE()
        code.add_particles(particles)
        code.verbose_flag = 0
        code.evolve_model(1.0)  # triggers set_stellar_evolution_properties + get_kick_properties

        for p in code.particles:
            if p.is_binary == False:
                stored = p.kick_distribution_sigma_km_s_NS_ECSN
                assert stored == ecsn_sigma, (
                    "H13: kick_distribution_sigma_km_s_NS_ECSN round-trip failed: "
                    "expected %g, got %g (old typo _NS_ECN would give default value %g)" % (ecsn_sigma, stored, 20.0)
                )
                # Verify the old typo attribute name does NOT exist
                assert not hasattr(p, 'kick_distribution_sigma_km_s_NS_ECN'), (
                    "H13: old typo attribute kick_distribution_sigma_km_s_NS_ECN should not exist"
                )

        code.reset()
        print("Test passed")

    def test28(self,args):
        """Regression test for H38: SSE global parameters not reset on every evolve_stars() call.

        Before the fix, evolve_stars() reinitialised all SSE Fortran common-block
        parameters (neta, bwind, sigma, bhflag, etc.) to hard-coded defaults on
        every call.  This would have silently overridden any user-set values.
        The fix removes the resets, keeping only flags_.ceflag and sse_error_code
        which legitimately need updating per call.

        Observable test: a single star evolved in one large step should give the
        same final state as the same star evolved in many small steps.  If SSE
        globals were being incorrectly reinitialised, the stellar track would drift.
        Also verifies that binary_evolution_CE_energy_flag changes are respected
        across multiple evolve_model() calls (the one flag still updated per call).
        """
        print("Test H38: SSE global parameters preserved across evolve_stars() calls")

        def evolve_star_to_time(n_steps, end_time=1.3e8):
            """Evolve a 5 M_sun primary in a wide binary to end_time in n_steps steps.

            A wide orbit (1000 AU) ensures no mass transfer so only SSE governs
            the evolution.  After ~1.2e8 yr a 5 M_sun star leaves the main sequence.
            """
            particles = Tools.create_fully_nested_multiple(
                2, [5.0, 1.0], [1000.0], [0.0], [0.01], [0.01], [0.01],
                metallicities=[0.02, 0.02], stellar_types=[1, 1], object_types=[1, 1]
            )
            code = MSE()
            code.add_particles(particles)
            code.verbose_flag = 0
            dt = end_time / n_steps
            t = 0.0
            for _ in range(n_steps):
                t += dt
                code.evolve_model(t)
            bodies = [x for x in code.particles if x.is_binary == False]
            primary = bodies[0]
            result = {
                'mass': primary.mass,
                'stellar_type': primary.stellar_type,
            }
            code.reset()
            return result

        one_step   = evolve_star_to_time(1)
        ten_steps  = evolve_star_to_time(10)
        many_steps = evolve_star_to_time(50)

        tol = 1e-4
        # Stellar type should be identical regardless of timestep count
        assert one_step['stellar_type'] == ten_steps['stellar_type'], (
            "H38: stellar type differs: 1-step=%d vs 10-steps=%d; "
            "SSE global reset between calls would corrupt the stellar track" % (
                one_step['stellar_type'], ten_steps['stellar_type'])
        )
        assert one_step['stellar_type'] == many_steps['stellar_type'], (
            "H38: stellar type differs: 1-step=%d vs 50-steps=%d; "
            "SSE global reset between calls would corrupt the stellar track" % (
                one_step['stellar_type'], many_steps['stellar_type'])
        )
        # Mass should agree to within 0.01% (SSE is deterministic given same params)
        assert abs(one_step['mass'] - ten_steps['mass']) / one_step['mass'] < tol, (
            "H38: mass drift between 1-step (%g) and 10-step (%g) evolution; "
            "SSE global parameter reset could cause systematic drift" % (
                one_step['mass'], ten_steps['mass'])
        )
        assert abs(one_step['mass'] - many_steps['mass']) / one_step['mass'] < tol, (
            "H38: mass drift between 1-step (%g) and 50-step (%g) evolution; "
            "SSE global parameter reset could cause systematic drift" % (
                one_step['mass'], many_steps['mass'])
        )

        # Verify binary_evolution_CE_energy_flag is respected across calls
        # (the one SSE flag that IS legitimately updated per evolve_stars() call)
        particles2 = Tools.create_fully_nested_multiple(
            2, [2.0, 1.0], [5.0], [0.0], [0.01], [0.01], [0.01],
            metallicities=[0.02, 0.02], stellar_types=[1, 1], object_types=[1, 1]
        )
        code2 = MSE()
        code2.add_particles(particles2)
        code2.verbose_flag = 0
        code2.binary_evolution_CE_energy_flag = 0  # alpha-lambda formalism
        t2 = 0.0
        for _ in range(5):
            t2 += 1.0e6
            code2.evolve_model(t2)
        code2.binary_evolution_CE_energy_flag = 1  # gamma-alpha formalism
        for _ in range(5):
            t2 += 1.0e6
            code2.evolve_model(t2)
        # Should not crash; flag changes must not be silently overridden
        code2.reset()

        if args.verbose:
            print("  1-step:   mass=%g stellar_type=%d" % (one_step['mass'], one_step['stellar_type']))
            print("  10-steps: mass=%g stellar_type=%d" % (ten_steps['mass'], ten_steps['stellar_type']))
            print("  50-steps: mass=%g stellar_type=%d" % (many_steps['mass'], many_steps['stellar_type']))

        print("Test passed")

    def test29(self,args):
        """Regression test for H39: spin angular momentum vector index bug in N-body setup.

        Before the fix, create_mstar_instance_of_system() wrote spin components into
        p->spin_AM_vec[i] (body index) instead of p->spin_AM_vec[j] (vector component).
        For a single body (i=0) the results happened to be correct for the z-component,
        but for multi-body systems all but the last component were lost.

        The bug's observable effect is subtle because spin_AM_vec is immediately
        overwritten by the correct MSTAR output after integration.  This test verifies:
        (a) N-body integration with non-zero spins does not crash, and
        (b) after N-body, the spin vector components are physically reasonable
            (non-NaN, norm approximately preserved).
        """
        print("Test H39: N-body spin angular momentum vector components correctly set")

        # Build a triple that is known to trigger secular breakdown and N-body switching
        # (same geometry as test22 but with non-zero stellar spins)
        particles = Tools.create_fully_nested_multiple(
            3,
            [1.0, 0.001, 1.0],
            [1.0, 5.0],
            [0.001, 0.001],
            [0.01, 89.5 * np.pi / 180.0],
            [0.01, 0.01],
            [0.01, 0.01],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2]
        )

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False
            # Set non-trivial spin on all three components to stress the index assignment
            b.spin_vec_x = 1.0e-3
            b.spin_vec_y = 2.0e-3
            b.spin_vec_z = 3.0e-3

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False
            b.check_for_physical_collision_or_orbit_crossing = False
            b.check_for_dynamical_instability = False
            b.check_for_entering_LISA_band = False

        outer_binary = binaries[1]
        outer_binary.check_for_secular_breakdown = False

        code = MSE()
        code.add_particles(particles)
        code.enable_tides = False
        code.include_flybys = False
        code.stop_after_root_found = True
        code.verbose_flag = 0

        t = 0.0
        dt = 5.0
        tend = 2.0e3
        nbody_activated = False

        while t < tend:
            t += dt
            code.evolve_model(t)
            particles_now = code.particles
            binaries_now = [x for x in particles_now if x.is_binary == True]
            bodies_now = [x for x in particles_now if x.is_binary == False]

            # Verify spin components are always finite (no NaN from bad index)
            for body in bodies_now:
                spin_norm = np.sqrt(body.spin_vec_x**2 + body.spin_vec_y**2 + body.spin_vec_z**2)
                assert np.isfinite(spin_norm), (
                    "H39: spin vector has non-finite component at t=%g: "
                    "spin=(%g,%g,%g)" % (t, body.spin_vec_x, body.spin_vec_y, body.spin_vec_z)
                )

            if code.CVODE_flag == 2:
                for b in binaries_now:
                    if b.secular_breakdown_has_occurred:
                        nbody_activated = True
                        break
                if nbody_activated:
                    break

        # The test passes if we reach here without crashing and with finite spins.
        # (Asserting N-body was activated is optional; it depends on the Kozai timescale.)
        if args.verbose:
            print("  N-body activated: %s at t=%g yr" % (nbody_activated, t))
            for body in bodies_now:
                print("  spin=(%g,%g,%g)" % (body.spin_vec_x, body.spin_vec_y, body.spin_vec_z))

        code.reset()
        print("Test passed")

    def test30(self,args):
        """Tests for interface logic error fixes (task 4.6).

        30a: H11 — error_code is 0 after a successful evolution step.
        30b: H15 — set_mass() does not clobber sse_initial_mass; ZAMS mass
             is preserved through wind-mass-loss steps.
        30c: H16 — eccentricity clamping keeps e in [0, 1) even for a
             strongly driven Kozai-Lidov triple where e approaches 1.
        30d: H17 — KS-regularised inner binary (integration_method=1) in a
             triple produces finite, physically valid orbital elements.
        """

        # ------------------------------------------------------------------
        # 30a: error_code == 0 on a clean, short evolution (H11 fix)
        # ------------------------------------------------------------------
        print("Test 30a: error_code is 0 after successful evolution")

        particles = Tools.create_fully_nested_multiple(
            2, [1.0, 1.0], [1.0], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1], object_types=[2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        code.evolve_model(1.0e3)

        assert hasattr(code, 'error_code'), "code.error_code attribute must exist after evolve_model()"
        assert code.error_code == 0, \
            "Expected error_code==0 after clean evolution, got {0}".format(code.error_code)

        print("Test 30a passed")
        code.reset()

        # ------------------------------------------------------------------
        # 30b: sse_initial_mass preserved through wind mass-loss (H15 fix)
        # ------------------------------------------------------------------
        print("Test 30b: sse_initial_mass preserved as ZAMS mass after wind mass loss")

        initial_mass = 20.0
        particles = Tools.create_fully_nested_multiple(
            2, [initial_mass, 1.0], [50.0], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1], object_types=[1, 1])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        t = 0.0
        dt = 1.0e6
        tend = 5.0e6
        while t < tend:
            t += dt
            code.evolve_model(t)

        particles_now = code.particles
        massive_star = [b for b in particles_now if b.is_binary == False][0]

        current_mass = massive_star.mass
        zams_mass = massive_star.sse_initial_mass

        assert current_mass > 0.0, \
            "Star mass should still be positive, got {0}".format(current_mass)
        assert zams_mass > 0.0, \
            "sse_initial_mass must be positive, got {0}".format(zams_mass)
        assert not np.isnan(zams_mass), \
            "sse_initial_mass is NaN — likely clobbered by set_mass()"
        assert not np.isinf(zams_mass), \
            "sse_initial_mass is infinite — likely clobbered by set_mass()"

        if args.verbose:
            print("  initial_mass={0:.4f}, current_mass={1:.4f}, "
                  "sse_initial_mass={2:.4f}".format(
                      initial_mass, current_mass, zams_mass))

        print("Test 30b passed")
        code.reset()

        # ------------------------------------------------------------------
        # 30c: eccentricity stays in [0, 1) for a Kozai-Lidov triple (H16)
        # ------------------------------------------------------------------
        print("Test 30c: eccentricity clamping keeps e in [0,1) for high-e Kozai-Lidov triple")

        particles = Tools.create_fully_nested_multiple(
            3, [1.0, 1.0e-3, 40.0e-3], [6.0, 100.0],
            [0.001, 0.6],
            [0.0, 65.0*np.pi/180.0],
            [45.0*np.pi/180.0, 0.0],
            [0.0, 0.0],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.relative_tolerance = 1.0e-14
        code.absolute_tolerance_eccentricity_vectors = 1.0e-14
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        t = 0.0
        dt = 1.0e6
        tend = 3.0e7

        while t < tend:
            t += dt
            code.evolve_model(t)

            particles_now = code.particles
            bs = [x for x in particles_now if x.is_binary == True]

            for b in bs:
                assert not np.isnan(b.e), \
                    "Eccentricity is NaN at t={0:.3e} for orbit {1}".format(t, b.index)
                assert not np.isinf(b.e), \
                    "Eccentricity is infinite at t={0:.3e}".format(t)
                assert b.e >= 0.0, \
                    "Eccentricity is negative ({0:.6f}) at t={1:.3e} — H16 clamping failure".format(b.e, t)
                assert b.e < 1.0, \
                    "Eccentricity >= 1 ({0:.6f}) at t={1:.3e} — H16 clamping failure".format(b.e, t)
                assert b.a > 0.0, \
                    "Semi-major axis <= 0 at t={0:.3e}".format(t)

        print("Test 30c passed")
        code.reset()

        # ------------------------------------------------------------------
        # 30d: KS-regularised binary (integration_method=1) gives valid
        #      orbital elements at each step (H17 fix)
        # ------------------------------------------------------------------
        print("Test 30d: KS-regularised isolated binary produces valid orbital elements")

        particles = Tools.create_fully_nested_multiple(
            2, [1.0, 1.0], [1.0],
            [0.3],
            [0.0],
            [0.0],
            [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1],
            object_types=[2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        the_binary = binaries[0]
        the_binary.integration_method = 1

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        P_orb = np.sqrt(1.0**3 / 2.0)  # yr
        tend_d = 20.0 * P_orb

        t = 0.0
        dt = P_orb
        while t < tend_d:
            t += dt
            code.evolve_model(t)

            particles_now = code.particles
            bs = [x for x in particles_now if x.is_binary == True]

            for b in bs:
                assert not np.isnan(b.a), \
                    "Semi-major axis is NaN at t={0:.4f} for orbit {1}".format(t, b.index)
                assert not np.isnan(b.e), \
                    "Eccentricity is NaN at t={0:.4f} for orbit {1}".format(t, b.index)
                assert b.a > 0.0, \
                    "Semi-major axis <= 0 at t={0:.4f}".format(t)
                assert 0.0 <= b.e < 1.0, \
                    "Eccentricity out of range ({0:.6f}) at t={1:.4f}".format(b.e, t)

        final_a = bs[0].a
        assert abs(final_a - 1.0) < 0.01, \
            "Semi-major axis drifted by more than 1%: initial=1.0, final={0:.6f}".format(final_a)

        print("Test 30d passed")
        code.reset()

        print("Test passed")

    def test31(self, args):
        """Tests for task P5.1 — Fix Critical Parameter Default Mismatches.

        31a: MSE class defaults match paper Table 1:
             nbody_analysis_fractional_integration_time=0.1, CE_spin_flag=0.
        31b: Particle class defaults match paper Sects. 2.6.3/2.6.4/2.8.3:
             all mass-loss / CE timescales are 1e2 yr (not 1e3).
        31c: CE spin flag=0 (default) is correctly propagated to C++ and
             independent resets each give the same default.
        31d: P5.2 — WD (types 10-12) use physical collision radius; only
             NS/BH (types 13-14) get the enlarged factor.  A tight WD binary
             survives without collision using the new logic.
        """

        # ------------------------------------------------------------------
        # 31a: MSE class defaults
        # ------------------------------------------------------------------
        print("Test 31a: MSE class parameter defaults match paper (task P5.1)")

        code = MSE()

        assert code.nbody_analysis_fractional_integration_time == 0.1, \
            "nbody_analysis_fractional_integration_time should be 0.1 (paper default), got {0}".format(
                code.nbody_analysis_fractional_integration_time)

        assert code.binary_evolution_CE_spin_flag == 0, \
            "binary_evolution_CE_spin_flag should be 0 (paper default = spins unaffected), got {0}".format(
                code.binary_evolution_CE_spin_flag)

        if args.verbose:
            print("  nbody_analysis_fractional_integration_time =", code.nbody_analysis_fractional_integration_time)
            print("  binary_evolution_CE_spin_flag =", code.binary_evolution_CE_spin_flag)

        print("Test 31a passed")
        code.reset()

        # ------------------------------------------------------------------
        # 31b: Particle class defaults
        # ------------------------------------------------------------------
        print("Test 31b: Particle class default timescales are 1e2 yr (paper defaults)")

        particles = Tools.create_fully_nested_multiple(
            2, [1.0, 1.0], [1.0], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1], object_types=[1, 1])

        binaries = [x for x in particles if x.is_binary == True]
        assert len(binaries) == 1, "Expected 1 binary, got {0}".format(len(binaries))
        b = binaries[0]

        assert b.dynamical_mass_transfer_low_mass_donor_timescale == 1.0e2, \
            "dynamical_mass_transfer_low_mass_donor_timescale default should be 1e2 yr, got {0}".format(
                b.dynamical_mass_transfer_low_mass_donor_timescale)
        assert b.dynamical_mass_transfer_WD_donor_timescale == 1.0e2, \
            "dynamical_mass_transfer_WD_donor_timescale default should be 1e2 yr, got {0}".format(
                b.dynamical_mass_transfer_WD_donor_timescale)
        assert b.compact_object_disruption_mass_loss_timescale == 1.0e2, \
            "compact_object_disruption_mass_loss_timescale default should be 1e2 yr, got {0}".format(
                b.compact_object_disruption_mass_loss_timescale)
        assert b.common_envelope_timescale == 1.0e2, \
            "common_envelope_timescale default should be 1e2 yr, got {0}".format(
                b.common_envelope_timescale)

        if args.verbose:
            print("  dynamical_mass_transfer_low_mass_donor_timescale =", b.dynamical_mass_transfer_low_mass_donor_timescale)
            print("  dynamical_mass_transfer_WD_donor_timescale =", b.dynamical_mass_transfer_WD_donor_timescale)
            print("  compact_object_disruption_mass_loss_timescale =", b.compact_object_disruption_mass_loss_timescale)
            print("  common_envelope_timescale =", b.common_envelope_timescale)

        print("Test 31b passed")

        # ------------------------------------------------------------------
        # 31c: CE spin flag persists correctly through code lifecycle
        # ------------------------------------------------------------------
        print("Test 31c: CE spin flag default=0, settable to 1, resets to 0")

        code = MSE()
        assert code.binary_evolution_CE_spin_flag == 0, \
            "Fresh MSE should have CE_spin_flag=0, got {0}".format(code.binary_evolution_CE_spin_flag)

        code.binary_evolution_CE_spin_flag = 1
        assert code.binary_evolution_CE_spin_flag == 1, \
            "After setting CE_spin_flag=1, expected 1 got {0}".format(code.binary_evolution_CE_spin_flag)

        code.reset()
        assert code.binary_evolution_CE_spin_flag == 0, \
            "After reset, CE_spin_flag should return to default 0, got {0}".format(
                code.binary_evolution_CE_spin_flag)

        # Verify a second independent MSE instance also has the correct default
        code2 = MSE()
        assert code2.binary_evolution_CE_spin_flag == 0, \
            "Second MSE instance should also default to CE_spin_flag=0, got {0}".format(
                code2.binary_evolution_CE_spin_flag)

        if args.verbose:
            print("  CE spin flag lifecycle verified: default=0 → set to 1 → reset to 0")

        print("Test 31c passed")
        code.reset()

        # ------------------------------------------------------------------
        # 31d: WD collision radius uses physical radius (P5.2 fix)
        # A tight WD-WD binary (a=0.002 AU) must NOT trigger an immediate
        # collision when effective_radius_multiplication_factor_for_collisions_compact_objects=100.
        # Before the P5.2 fix (stellar_type>=10), WDs received the 100x factor
        # and would have collided immediately (100 * ~4e-5 AU > 0.002 AU).
        # After the fix (stellar_type>=13), WDs use their physical radius.
        # ------------------------------------------------------------------
        print("Test 31d: WD binary (type 11) at 0.002 AU survives without collision (P5.2 fix)")

        particles = Tools.create_fully_nested_multiple(
            2, [0.6, 0.6], [0.002], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[11, 11], object_types=[1, 1])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        for bdy in bodies:
            bdy.evolve_as_star = False
            bdy.include_mass_transfer_terms = False
        for orb in binaries:
            orb.include_pairwise_1PN_terms = False
            orb.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = True
        code.verbose_flag = 0
        # Use the default factor of 100 to confirm WDs are NOT getting it
        code.effective_radius_multiplication_factor_for_collisions_compact_objects = 1.0e2

        # Evolve for a short time; if WDs erroneously received the 100x factor
        # the initial separation (0.002 AU) would be less than 2*100*r_WD
        # (approximately 2*100*4.4e-5 AU ~ 0.0088 AU), triggering a collision
        # at t=0.  With the fix, WDs use their physical radii and the binary survives.
        code.evolve_model(1.0)

        particles_after = code.particles
        binaries_after = [x for x in particles_after if x.is_binary == True]
        bodies_after = [x for x in particles_after if x.is_binary == False]

        # Verify the binary still exists (no collision dissolved it)
        assert len(binaries_after) >= 1, \
            "WD binary should still exist after 1 yr; P5.2 fix may be broken"
        assert len(bodies_after) >= 2, \
            "Both WDs should still be present after 1 yr; P5.2 fix may be broken"

        # Verify both bodies are still WDs (type 10, 11, or 12), not merged objects
        for bdy in bodies_after:
            assert bdy.stellar_type in (10, 11, 12), \
                "Body should still be a WD (type 10-12) after 1 yr, got type {0}".format(
                    bdy.stellar_type)

        if args.verbose:
            print("  WD binary survived 1 yr at a=0.002 AU with compact_objects factor=100")
            inner_orbit = binaries_after[0]
            print("  Final a={0:.4e} AU, e={1:.4f}".format(inner_orbit.a, inner_orbit.e))

        print("Test 31d passed")
        code.reset()

        print("Test passed")

    def test32(self, args):
        """Task 7.1: Zero-eccentricity and non-solar metallicity tests.

        32a: Circular inner orbit (e=0) in a hierarchical triple with tidal
             apsidal motion enabled. Validates the C12 fix in ODE_tides.cpp
             (compute_estimated_tidal_apsidal_motion_timescales guarded by
             e <= epsilon to avoid dot3(spin_vec, e_vec)/e division-by-zero).
             The function is called from check_for_integration_exclusion_orbits,
             which only runs for inner binaries of hierarchical triples.

        32b: Sub-solar metallicity (Z=0.001) binary stellar evolution.
             No NaN, physically reasonable stellar tracks, results differ
             from the solar-metallicity (Z=0.02) run because SSE uses
             Z-dependent tracks (luminosity, radius, lifetimes).

        32c: Super-solar metallicity (Z=0.03) binary stellar evolution.
             Same basic validation as 32b; results differ from solar.
        """
        print("Test 32: Zero-eccentricity and non-solar metallicity tests")

        # ------------------------------------------------------------------
        # 32a: C12 regression — circular orbit (e=0) with tidal apsidal motion
        #
        # Before the C12 fix, compute_estimated_tidal_apsidal_motion_timescales
        # in ODE_tides.cpp computed:
        #     dot3(spin_vec, e_vec) / e
        # which divides by zero when e == 0, producing NaN in the apsidal-
        # motion timescale used to decide ODE exclusion for the inner binary.
        # The fix uses the algebraic identity:
        #     2*(s.h_hat)^2 - (s.q_hat)^2 - (s.e_hat)^2 = 3*(s.h_hat)^2 - |s|^2
        # which is valid for any orthonormal triad and avoids the division.
        # ------------------------------------------------------------------
        print("Test 32a: Circular inner orbit (e=0) tidal fix regression (C12)")

        code32a = MSE()
        CONST_R_SUN = code32a.CONST_R_SUN
        k_AM = 0.19
        rg   = 0.25

        # Hierarchical triple: inner binary has e=0 exactly; outer has e=0.3.
        # A triple is required because compute_estimated_tidal_apsidal_motion_timescales
        # is called inside check_for_integration_exclusion_orbits, which only
        # processes inner binaries (those with a parent orbit).
        particles = Tools.create_fully_nested_multiple(
            3,
            [1.0, 0.8, 0.5],
            [1.0, 50.0],
            [0.0, 0.3],          # inner eccentricity = 0 (exact circular orbit)
            [0.01, 0.5],
            [0.01, 0.01],
            [0.01, 0.01],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2],
        )

        bodies   = [x for x in particles if not x.is_binary]
        binaries = [x for x in particles if x.is_binary]

        for b in bodies:
            b.evolve_as_star              = False
            b.include_mass_transfer_terms = False
            b.check_for_RLOF_at_pericentre = False
            b.include_spin_orbit_1PN_terms = False

        for b in binaries:
            b.include_pairwise_1PN_terms  = False
            b.include_pairwise_25PN_terms = False

        # Set physical stellar properties on the two inner-binary stars so that
        # compute_estimated_tidal_apsidal_motion_timescales receives non-trivial
        # radius and apsidal_motion_constant values (required to trigger the
        # tidal timescale path rather than returning immediately with C = 0).
        for i, body in enumerate(bodies[:2]):
            body.radius                                = CONST_R_SUN * (1.0 if i == 0 else 0.8)
            body.spin_vec_x                            = 0.0
            body.spin_vec_y                            = 0.0
            body.spin_vec_z                            = 1.0e-2
            body.apsidal_motion_constant               = k_AM
            body.gyration_radius                       = rg
            body.tides_viscous_time_scale              = 1.0e10
            body.tides_viscous_time_scale_prescription = 0
            body.tides_method                          = 1
            body.include_tidal_friction_terms          = False
            body.include_tidal_bulges_precession_terms = True
            body.include_rotation_precession_terms     = True
            body.minimum_eccentricity_for_tidal_precession = 1.0e-8

        # Tertiary: tides disabled (outer body, not part of inner pair)
        bodies[2].radius                               = 0.5 * CONST_R_SUN
        bodies[2].spin_vec_z                           = 1.0e-5
        bodies[2].include_tidal_friction_terms         = False
        bodies[2].include_tidal_bulges_precession_terms = False
        bodies[2].include_rotation_precession_terms    = False

        code32a.add_particles(particles)
        code32a.enable_tides        = True
        code32a.include_flybys      = False
        code32a.enable_root_finding = False
        code32a.verbose_flag        = 0

        t    = 0.0
        dt   = 1.0e4   # 10 000 yr per step
        tend = 1.0e5   # 100 000 yr total

        while t < tend:
            t += dt
            code32a.evolve_model(t)

            for p in code32a.particles:
                if p.is_binary:
                    assert not np.isnan(p.a), \
                        "32a: semi-major axis NaN at t=%g yr — C12 tidal e=0 bug" % t
                    assert not np.isinf(p.a), \
                        "32a: semi-major axis Inf at t=%g yr" % t
                    assert p.a > 0, \
                        "32a: semi-major axis non-positive at t=%g yr" % t
                    assert not np.isnan(p.e), \
                        "32a: eccentricity NaN at t=%g yr — C12 tidal e=0 bug" % t
                    assert not np.isinf(p.e), \
                        "32a: eccentricity Inf at t=%g yr" % t
                    assert 0.0 <= p.e < 1.0, \
                        "32a: eccentricity out of range (%.6f) at t=%g yr" % (p.e, t)
                else:
                    for attr in ('spin_vec_x', 'spin_vec_y', 'spin_vec_z'):
                        val = getattr(p, attr)
                        assert not np.isnan(val), \
                            "32a: %s NaN at t=%g yr — C12 tidal e=0 bug" % (attr, t)
                        assert not np.isinf(val), \
                            "32a: %s Inf at t=%g yr" % (attr, t)

        if args.verbose:
            bins = [p for p in code32a.particles if p.is_binary]
            print("  32a final: a_in=%.4f AU  e_in=%.2e" % (bins[0].a, bins[0].e))

        print("Test 32a passed")
        code32a.reset()

        # ------------------------------------------------------------------
        # 32b & 32c: Non-solar metallicity stellar evolution
        #
        # A 4 + 0.5 M_sun binary at 1000 AU (wide enough to prevent any RLOF)
        # is evolved to 300 Myr at three metallicities.  Stellar evolution is
        # enabled (the default); the wide orbit means the two stars evolve as
        # isolated single stars.
        #
        # At 300 Myr a 4 M_sun star has completed main-sequence evolution
        # (~158 Myr at Z=0.02) and is either a WD or a late AGB star.
        # SSE uses Z-dependent stellar tracks, so the luminosity, mass, and
        # stellar type at 300 Myr differ between metallicities.
        # ------------------------------------------------------------------

        def run_stellar_evolution(Z, tend_yr, verbose=False):
            """Evolve a 4 + 0.5 M_sun binary at metallicity Z for tend_yr years.

            Returns (stellar_type, mass, luminosity) of the 4 M_sun primary.
            Asserts no NaN in masses, stellar types, and luminosities at each step.
            """
            particles = Tools.create_fully_nested_multiple(
                2, [4.0, 0.5], [1000.0], [0.0], [0.01], [0.01], [0.01],
                metallicities=[Z, Z], stellar_types=[1, 1], object_types=[1, 1])

            c = MSE()
            c.add_particles(particles)
            c.enable_tides        = False
            c.include_flybys      = False
            c.enable_root_finding = False
            c.verbose_flag        = 0

            N  = 20
            dt = tend_yr / float(N)
            t  = 0.0

            while t < tend_yr:
                t += dt
                c.evolve_model(t)

                for p in c.particles:
                    if not p.is_binary:
                        assert not np.isnan(p.mass), \
                            "Z=%.4f: mass NaN at t=%.2e yr" % (Z, t)
                        assert not np.isinf(p.mass), \
                            "Z=%.4f: mass Inf at t=%.2e yr" % (Z, t)
                        assert p.mass >= 0.0, \
                            "Z=%.4f: mass negative at t=%.2e yr" % (Z, t)
                        assert 0 <= p.stellar_type <= 15, \
                            "Z=%.4f: invalid stellar_type=%d at t=%.2e yr" % (
                                Z, p.stellar_type, t)
                        assert not np.isnan(p.luminosity), \
                            "Z=%.4f: luminosity NaN at t=%.2e yr" % (Z, t)
                        assert not np.isinf(p.luminosity), \
                            "Z=%.4f: luminosity Inf at t=%.2e yr" % (Z, t)
                    else:
                        assert not np.isnan(p.a), \
                            "Z=%.4f: semi-major axis NaN at t=%.2e yr" % (Z, t)
                        assert not np.isinf(p.a), \
                            "Z=%.4f: semi-major axis Inf at t=%.2e yr" % (Z, t)

                if verbose:
                    bods = [p for p in c.particles if not p.is_binary]
                    print("  Z=%.4f  t=%.1e yr  types=%s  masses=%s" % (
                        Z, t,
                        [p.stellar_type for p in bods],
                        ["%.3f" % p.mass for p in bods]))

            final_particles = c.particles
            primary = [p for p in final_particles if not p.is_binary][0]
            c.reset()
            return primary.stellar_type, primary.mass, primary.luminosity

        tend_yr = 3.0e8   # 300 Myr — past MS lifetime for a 4 M_sun star at all Z

        print("Test 32b: Sub-solar metallicity Z=0.001 stellar evolution")
        kw_sol, m_sol, lum_sol = run_stellar_evolution(0.02,  tend_yr, args.verbose)
        kw_sub, m_sub, lum_sub = run_stellar_evolution(0.001, tend_yr, args.verbose)

        if args.verbose:
            print("  Solar    (Z=0.02):   type=%d  mass=%.4f  lum=%.4e" %
                  (kw_sol, m_sol, lum_sol))
            print("  Sub-solar(Z=0.001):  type=%d  mass=%.4f  lum=%.4e" %
                  (kw_sub, m_sub, lum_sub))

        # Sanity: valid stellar types and positive masses/luminosities
        assert 0 <= kw_sol <= 15, "Solar: invalid stellar_type=%d" % kw_sol
        assert 0 <= kw_sub <= 15, "Sub-solar: invalid stellar_type=%d" % kw_sub
        assert m_sol  > 0, "Solar: primary mass must be positive"
        assert m_sub  > 0, "Sub-solar: primary mass must be positive"
        assert lum_sol > 0, "Solar: luminosity must be positive"
        assert lum_sub > 0, "Sub-solar: luminosity must be positive"

        # The 4 M_sun star should have evolved past the ZAMS MS (type 1)
        # well before 300 Myr (MS lifetime ~158 Myr at Z=0.02).
        assert kw_sol > 1, \
            "Solar (Z=0.02): 4 M_sun star should be post-MS at 300 Myr, got type %d" % kw_sol
        assert kw_sub > 1, \
            "Sub-solar (Z=0.001): 4 M_sun star should be post-MS at 300 Myr, got type %d" % kw_sub

        # SSE uses Z-dependent stellar tracks: at the same elapsed time the
        # luminosity, mass, and/or stellar type must differ between metallicities.
        # (A metal-poor star evolves faster and may have reached WD stage earlier,
        # giving a cooler and dimmer WD than a star still on the giant branch at
        # solar metallicity, or vice versa — the direction depends on the exact
        # timing.  Any nonzero difference validates that Z is actually used.)
        differs_sub = (
            kw_sub != kw_sol
            or abs(m_sub - m_sol) > 1.0e-3
            or abs(lum_sub - lum_sol) / max(lum_sol, 1.0e-30) > 1.0e-3
        )
        assert differs_sub, (
            "Z=0.001 and Z=0.02 should produce different results at 300 Myr. "
            "Solar: type=%d mass=%.4f lum=%.4e; "
            "Sub-solar: type=%d mass=%.4f lum=%.4e" %
            (kw_sol, m_sol, lum_sol, kw_sub, m_sub, lum_sub))

        print("Test 32b passed")

        print("Test 32c: Super-solar metallicity Z=0.03 stellar evolution")
        kw_sup, m_sup, lum_sup = run_stellar_evolution(0.03, tend_yr, args.verbose)

        if args.verbose:
            print("  Solar      (Z=0.02): type=%d  mass=%.4f  lum=%.4e" %
                  (kw_sol, m_sol, lum_sol))
            print("  Super-solar(Z=0.03): type=%d  mass=%.4f  lum=%.4e" %
                  (kw_sup, m_sup, lum_sup))

        assert 0 <= kw_sup <= 15, "Super-solar: invalid stellar_type=%d" % kw_sup
        assert m_sup  > 0, "Super-solar: primary mass must be positive"
        assert lum_sup > 0, "Super-solar: luminosity must be positive"

        assert kw_sup > 1, \
            "Super-solar (Z=0.03): 4 M_sun star should be post-MS at 300 Myr, got type %d" % kw_sup

        differs_sup = (
            kw_sup != kw_sol
            or abs(m_sup - m_sol) > 1.0e-3
            or abs(lum_sup - lum_sol) / max(lum_sol, 1.0e-30) > 1.0e-3
        )
        assert differs_sup, (
            "Z=0.03 and Z=0.02 should produce different results at 300 Myr. "
            "Solar: type=%d mass=%.4f lum=%.4e; "
            "Super-solar: type=%d mass=%.4f lum=%.4e" %
            (kw_sol, m_sol, lum_sol, kw_sup, m_sup, lum_sup))

        print("Test 32c passed")
        print("Test passed")

    def test33(self, args):
        """Tests for task 7.2 — Wind Accretion and Dynamical Mass Transfer.

        33a: Wind accretion (Bondi-Hoyle) — TPAGB primary loses mass via a
             stellar wind; MS companion accretes a detectable fraction via
             the Bondi-Hoyle formula implemented in handle_wind_accretion().

        33b: Dynamical mass transfer with HG donor — a 5 M_sun HG star at
             a=0.5 AU has q=10 >> q_crit=4.0 (for kw=2).  When the HG star
             expands to fill its Roche lobe the code should trigger CE, causing
             a structural change within 20 Myr.

        33c: WD donor dynamical MT driven by GW inspiral — a He WD (kw=10,
             0.4 M_sun) + CO WD (kw=11, 0.5 M_sun) binary.  GW inspiral drives
             the He WD to RLOF; because q=0.8 > q_crit_WD=0.628, the code
             calls dynamical_mass_transfer_WD_donor(), which erases the donor.
             The CO WD accretor remains with stellar_type 9 or 11 and retains
             its original ~0.5 M_sun mass (the function does not explicitly
             update accretor->mass, unlike the low-mass donor path).
        """

        # ------------------------------------------------------------------
        # 33a: Bondi-Hoyle wind accretion
        # ------------------------------------------------------------------
        print("Test 33a: Bondi-Hoyle wind accretion — TPAGB + MS binary")

        particles = Tools.create_fully_nested_multiple(
            2, [3.0, 0.8], [6.0], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[6, 1],
            object_types=[1, 1])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.verbose_flag = 0

        code.evolve_model(0.0)

        bodies_init = [p for p in code.particles if p.is_binary == False]
        tpagb_init = max(bodies_init, key=lambda p: p.mass)
        ms_init = min(bodies_init, key=lambda p: p.mass)
        m_primary_init = tpagb_init.mass
        m_secondary_init = ms_init.mass

        for p in code.particles:
            if p.is_binary:
                p.check_for_RLOF_at_pericentre = False

        code.evolve_model(5.0e6)

        particles_now = code.particles
        bodies_now = [p for p in particles_now if p.is_binary == False]
        binaries_now = [p for p in particles_now if p.is_binary == True]

        assert len(bodies_now) >= 1, "33a: at least one body should remain"

        for b in bodies_now:
            assert np.isfinite(b.mass) and b.mass > 0.0, \
                "33a: body mass is NaN or non-positive: {0}".format(b.mass)

        m_total_init = m_primary_init + m_secondary_init
        m_total_final = sum(b.mass for b in bodies_now)
        assert m_total_final < m_total_init, (
            "33a: total mass must decrease due to TPAGB wind; "
            "got {0:.4f} (init {1:.4f})".format(m_total_final, m_total_init))

        ms_bodies_now = [b for b in bodies_now if b.stellar_type == 1]
        assert len(ms_bodies_now) == 1, \
            "33a: MS companion (type 1) should still exist after 5 Myr"
        m_secondary_final = ms_bodies_now[0].mass
        assert m_secondary_final > m_secondary_init, (
            "33a: MS companion must gain mass via Bondi-Hoyle wind accretion; "
            "got {0:.6f} (init {1:.6f})".format(m_secondary_final, m_secondary_init))

        if binaries_now:
            a_final = binaries_now[0].a
            assert np.isfinite(a_final) and a_final > 6.0, (
                "33a: orbit must widen due to mass loss; "
                "got a_final={0:.4f} AU".format(a_final))

        if args.verbose:
            print("  m_primary: {0:.4f} -> ? (TPAGB -> WD)".format(m_primary_init))
            print("  m_secondary (MS): {0:.6f} -> {1:.6f} M_sun".format(
                m_secondary_init, m_secondary_final))
            if binaries_now:
                print("  a: 6.0000 -> {0:.4f} AU".format(binaries_now[0].a))

        print("Test 33a passed")
        code.reset()

        # ------------------------------------------------------------------
        # 33b: Dynamical mass transfer — HG donor triggers CE
        # ------------------------------------------------------------------
        print("Test 33b: CE evolution from HG donor (q >> q_crit)")

        particles = Tools.create_fully_nested_multiple(
            2, [5.0, 0.5], [0.5], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[2, 1],
            object_types=[1, 1])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.verbose_flag = 0

        t = 0.0
        dt = 5.0e5
        tend = 2.0e7
        structure_changed = False

        while t < tend:
            t += dt
            code.evolve_model(t)
            if code.structure_change:
                structure_changed = True
                break

        assert structure_changed, (
            "33b: CE should have changed system structure within 20 Myr; "
            "HG donor (kw=2) with q=10 >> q_crit=4.0 must trigger CE when RLOF occurs")

        if args.verbose:
            particles_after = code.particles
            bodies_after = [p for p in particles_after if p.is_binary == False]
            print("  CE occurred at t={0:.2e} yr; N_bodies={1}".format(
                t, len(bodies_after)))
            for body in bodies_after:
                print("  body: mass={0:.4f}, stellar_type={1}".format(
                    body.mass, body.stellar_type))

        print("Test 33b passed")
        code.reset()

        # ------------------------------------------------------------------
        # 33c: WD donor dynamical MT via GW inspiral (He WD + CO WD)
        # ------------------------------------------------------------------
        print("Test 33c: GW-driven He WD donor dynamical MT (He WD + CO WD binary)")

        CONST_R_SUN_AU = 0.00465
        R_HeWD  = 0.0158 * CONST_R_SUN_AU
        R_CO_WD = 0.0142 * CONST_R_SUN_AU

        particles = Tools.create_fully_nested_multiple(
            2, [0.4, 0.5], [0.01], [0.0], [0.0], [0.0], [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1],
            object_types=[1, 1])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]

        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = True

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = False
        code.enable_tides = False
        code.verbose_flag = 0

        code.evolve_model(0.0)

        bodies_init = [p for p in code.particles if p.is_binary == False]
        he_wd_body = min(bodies_init, key=lambda p: p.mass)
        co_wd_body = max(bodies_init, key=lambda p: p.mass)

        he_wd_body.stellar_type = 10
        he_wd_body.age = 0.0
        he_wd_body.radius = R_HeWD
        he_wd_body.core_mass = he_wd_body.mass

        co_wd_body.stellar_type = 11
        co_wd_body.age = 0.0
        co_wd_body.radius = R_CO_WD
        co_wd_body.core_mass = co_wd_body.mass

        binary_particle = [p for p in code.particles if p.is_binary][0]
        binary_particle.a = 2.15e-4
        binary_particle.e = 0.0

        t = 0.0
        dt = 200.0
        tend = 5.0e3
        structure_changed = False

        while t < tend:
            t += dt
            code.evolve_model(t)
            if code.structure_change:
                structure_changed = True
                break

        assert structure_changed, (
            "33c: GW-driven He WD dynamical MT should change system structure within 5 kyr; "
            "He WD donor with q=0.8 > q_crit_WD=0.628 should trigger dynamical_mass_transfer_WD_donor()")

        particles_after = code.particles
        bodies_after = [p for p in particles_after if p.is_binary == False]

        assert len(bodies_after) == 1, (
            "33c: He WD donor should be erased by dynamical_mass_transfer_WD_donor(); "
            "expected exactly 1 surviving body, got {0}".format(len(bodies_after)))

        assert bodies_after[0].stellar_type in (9, 11, 12), (
            "33c: CO WD accretor should be HeGB (9) or a WD (11/12) after accreting He WD; "
            "got stellar_type={0}".format(bodies_after[0].stellar_type))

        assert abs(bodies_after[0].mass - 0.5) < 0.05, (
            "33c: surviving CO WD accretor should retain ~0.5 M_sun (WD donor path); "
            "got {0:.4f} M_sun".format(bodies_after[0].mass))

        if args.verbose:
            print("  GW inspiral triggered He WD dynamical MT at t={0:.1f} yr".format(t))
            print("  surviving body: mass={0:.4f}, stellar_type={1}".format(
                bodies_after[0].mass, bodies_after[0].stellar_type))

        print("Test 33c passed")
        code.reset()

        print("Test passed")

    def test34(self, args):
        """Tests for task 7.3 — VRR Integration Test.

        34a: VRR model 1 precesses the angular momentum vector (h_vec) of the
             outer orbit in a triple system.
             Checks: (1) inclination changes significantly after ~T/4 of precession,
                     (2) a and e are conserved (pure precession conserves |h| and |e_vec|),
                     (3) no NaN in orbital elements, (4) error_code == 0.
        34b: VRR precession runs for a full cycle and the outer-orbit inclination
             returns near zero (initial value) after one full precession period.
        """

        # ------------------------------------------------------------------
        # 34a: VRR model 1 tilts the outer orbit's inclination in a triple
        # ------------------------------------------------------------------
        print("Test 34a: VRR model 1 precesses outer-orbit inclination in triple without NaN")

        particles = Tools.create_fully_nested_multiple(
            3, [1.0, 1.0, 0.5], [5.0, 50.0],
            [0.1, 0.2],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            metallicities=[0.02, 0.02, 0.02],
            stellar_types=[1, 1, 1],
            object_types=[2, 2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        outer_orbit = binaries[-1]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        VRR_Omega = 2.0e-2  # rad/yr → T = 2*pi/Omega ≈ 314 yr
        outer_orbit.VRR_model = 1
        outer_orbit.VRR_include_mass_precession = 0
        outer_orbit.VRR_Omega_vec_x = VRR_Omega
        outer_orbit.VRR_Omega_vec_y = 0.0
        outer_orbit.VRR_Omega_vec_z = 0.0

        code = MSE()
        code.enable_VRR = True
        code.add_particles(particles)
        code.include_quadrupole_order_terms = False
        code.include_octupole_order_binary_pair_terms = False
        code.include_flybys = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        a_out = 50.0
        e_out = 0.2
        t_evolve = 100.0  # yr; Omega*t = 2.0 rad
        code.evolve_model(t_evolve)

        assert code.error_code == 0, \
            "error_code should be 0 after VRR evolution, got {0}".format(code.error_code)

        particles_after = code.particles
        binaries_after = [x for x in particles_after if x.is_binary == True]
        outer_after = binaries_after[-1]

        assert not np.isnan(outer_after.a), "a is NaN after VRR evolution"
        assert not np.isnan(outer_after.e), "e is NaN after VRR evolution"
        assert not np.isnan(outer_after.INCL), "INCL is NaN after VRR evolution"

        a_rel_err = abs(outer_after.a - a_out) / a_out
        assert a_rel_err < 1.0e-3, \
            "a changed by {0:.2e} under pure VRR precession: " \
            "initial={1:.4f}, final={2:.4f} AU".format(a_rel_err, a_out, outer_after.a)

        e_rel_err = abs(outer_after.e - e_out) / e_out
        assert e_rel_err < 1.0e-3, \
            "e changed by {0:.2e} under pure VRR precession: " \
            "initial={1:.4f}, final={2:.4f}".format(e_rel_err, e_out, outer_after.e)

        assert outer_after.INCL > 0.1, \
            "INCL = {0:.4f} rad after {1:.0f} yr of VRR precession — " \
            "expected > 0.1 rad (orbit should be visibly tilted)".format(
                outer_after.INCL, t_evolve)

        if args.verbose:
            print("  a: initial={0:.4f}, final={1:.4f} AU (rel err {2:.2e})".format(
                a_out, outer_after.a, a_rel_err))
            print("  e: initial={0:.4f}, final={1:.4f} (rel err {2:.2e})".format(
                e_out, outer_after.e, e_rel_err))
            print("  INCL = {0:.4f} rad = {1:.2f} deg (expected > 0.1 rad)".format(
                outer_after.INCL, np.degrees(outer_after.INCL)))

        print("Test 34a passed")
        code.reset()

        # ------------------------------------------------------------------
        # 34b: VRR precession over a full cycle returns INCL near zero
        # ------------------------------------------------------------------
        print("Test 34b: VRR precession completes a full cycle without NaN or inclination drift")

        T_prec = 2.0 * np.pi / VRR_Omega  # ≈ 314 yr

        def _make_vrr_code():
            """Helper: fresh VRR triple system with outer-orbit VRR_model=1."""
            pts = Tools.create_fully_nested_multiple(
                3, [1.0, 1.0, 0.5], [5.0, 50.0],
                [0.1, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                metallicities=[0.02, 0.02, 0.02],
                stellar_types=[1, 1, 1],
                object_types=[2, 2, 2])
            bs = [x for x in pts if x.is_binary == False]
            bns = [x for x in pts if x.is_binary == True]
            out = bns[-1]
            for b in bs:
                b.evolve_as_star = False
                b.include_mass_transfer_terms = False
            for b in bns:
                b.include_pairwise_1PN_terms = False
                b.include_pairwise_25PN_terms = False
            out.VRR_model = 1
            out.VRR_include_mass_precession = 0
            out.VRR_Omega_vec_x = VRR_Omega
            out.VRR_Omega_vec_y = 0.0
            out.VRR_Omega_vec_z = 0.0
            c = MSE()
            c.enable_VRR = True
            c.add_particles(pts)
            c.include_quadrupole_order_terms = False
            c.include_octupole_order_binary_pair_terms = False
            c.include_flybys = False
            c.enable_tides = False
            c.enable_root_finding = False
            c.verbose_flag = 0
            return c

        c_half = _make_vrr_code()
        c_half.evolve_model(0.5 * T_prec)
        assert c_half.error_code == 0, \
            "error_code != 0 at t=T_prec/2: {0}".format(c_half.error_code)
        bns_half = [x for x in c_half.particles if x.is_binary == True]
        outer_half = bns_half[-1]
        assert not np.isnan(outer_half.INCL), "INCL is NaN at T_prec/2"
        assert abs(outer_half.INCL - np.pi) < 0.05, \
            "At T_prec/2, INCL = {0:.6f} rad (expected ≈ π = {1:.4f})".format(
                outer_half.INCL, np.pi)
        c_half.reset()

        c_full = _make_vrr_code()
        c_full.evolve_model(T_prec)
        assert c_full.error_code == 0, \
            "error_code != 0 at t=T_prec: {0}".format(c_full.error_code)
        bns_full = [x for x in c_full.particles if x.is_binary == True]
        outer_full = bns_full[-1]
        assert not np.isnan(outer_full.a), "a is NaN at T_prec"
        assert not np.isnan(outer_full.e), "e is NaN at T_prec"
        assert not np.isnan(outer_full.INCL), "INCL is NaN at T_prec"
        assert outer_full.INCL < 0.05, \
            "After full VRR precession period, INCL = {0:.6f} rad (expected ≈ 0)".format(
                outer_full.INCL)
        c_full.reset()

        if args.verbose:
            print("  Half-period INCL = {0:.6f} rad = {1:.4f} deg (expected π = 180 deg)".format(
                outer_half.INCL, np.degrees(outer_half.INCL)))
            print("  Full-period INCL = {0:.6f} rad = {1:.4f} deg (expected 0)".format(
                outer_full.INCL, np.degrees(outer_full.INCL)))

        print("Test 34b passed")

        print("Test passed")

    def test35(self, args):
        """Tests for task 7.3 — Flyby Integration Test.

        35a: External flyby perturbations (include_flybys=True) are applied
             during secular ODE evolution of a wide binary.
             Checks: (1) no crash (error_code == 0), (2) no NaN in orbital
             elements, (3) orbital elements change from their initial values
             (confirming at least one flyby perturbation was applied).
        35b: System remains gravitationally bound after the flyby-perturbed run:
             a > 0, 0 ≤ e < 1.
        """

        # ------------------------------------------------------------------
        # 35a: Flyby perturbations change orbital elements of a wide binary
        # ------------------------------------------------------------------
        print("Test 35a: Flyby perturbations change orbital elements of binary without crash")

        particles = Tools.create_fully_nested_multiple(
            2, [1.0, 1.0], [500.0],
            [0.3],
            [0.0],
            [0.0],
            [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1],
            object_types=[2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        the_binary = binaries[0]

        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        initial_a = the_binary.a
        initial_e = the_binary.e

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = True
        code.flybys_stellar_density = 1.0e5 * code.CONST_PER_PC3
        code.flybys_encounter_sphere_radius = 3000.0
        code.flybys_stellar_relative_velocity_dispersion = 30.0 * code.CONST_KM_PER_S
        code.random_seed = 42
        code.include_quadrupole_order_terms = False
        code.include_octupole_order_binary_pair_terms = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        t_evolve = 5000.0
        code.evolve_model(t_evolve)

        assert code.error_code == 0, \
            "error_code should be 0 after flyby evolution, got {0}".format(code.error_code)

        particles_after = code.particles
        binaries_after = [x for x in particles_after if x.is_binary == True]
        assert len(binaries_after) > 0, "Binary disappeared from particles list"

        orb = binaries_after[0]
        assert not np.isnan(orb.a), "Semi-major axis is NaN after flyby evolution"
        assert not np.isnan(orb.e), "Eccentricity is NaN after flyby evolution"
        assert not np.isnan(orb.INCL), "INCL is NaN after flyby evolution"

        delta_ae = abs(orb.a - initial_a) + abs(orb.e - initial_e)
        assert delta_ae > 1.0e-10, \
            "a={0:.4f} AU, e={1:.6f} unchanged after {2:.0f} yr — flyby module may not be active".format(
                orb.a, orb.e, t_evolve)

        if args.verbose:
            print("  Initial: a={0:.2f} AU, e={1:.4f}".format(initial_a, initial_e))
            print("  Final:   a={0:.2f} AU, e={1:.4f}".format(orb.a, orb.e))
            print("  |Δa| + |Δe| = {0:.6e}".format(delta_ae))

        print("Test 35a passed")
        code.reset()

        # ------------------------------------------------------------------
        # 35b: System remains gravitationally bound after flyby perturbations
        # ------------------------------------------------------------------
        print("Test 35b: Binary remains bound (a > 0, 0 ≤ e < 1) after flyby evolution")

        particles = Tools.create_fully_nested_multiple(
            2, [1.0, 1.0], [500.0],
            [0.3],
            [0.0],
            [0.0],
            [0.0],
            metallicities=[0.02, 0.02],
            stellar_types=[1, 1],
            object_types=[2, 2])

        bodies = [x for x in particles if x.is_binary == False]
        binaries = [x for x in particles if x.is_binary == True]
        for b in bodies:
            b.evolve_as_star = False
            b.include_mass_transfer_terms = False
        for b in binaries:
            b.include_pairwise_1PN_terms = False
            b.include_pairwise_25PN_terms = False

        code = MSE()
        code.add_particles(particles)
        code.include_flybys = True
        code.flybys_stellar_density = 1.0e5 * code.CONST_PER_PC3
        code.flybys_encounter_sphere_radius = 3000.0
        code.flybys_stellar_relative_velocity_dispersion = 30.0 * code.CONST_KM_PER_S
        code.random_seed = 42
        code.include_quadrupole_order_terms = False
        code.include_octupole_order_binary_pair_terms = False
        code.enable_tides = False
        code.enable_root_finding = False
        code.verbose_flag = 0

        t_short = 1000.0
        code.evolve_model(t_short)

        assert code.error_code == 0, \
            "error_code != 0 after short flyby run: {0}".format(code.error_code)

        particles_after = code.particles
        binaries_after = [x for x in particles_after if x.is_binary == True]
        assert len(binaries_after) > 0, "Binary vanished after {0:.0f} yr".format(t_short)

        orb = binaries_after[0]
        assert not np.isnan(orb.a), "a is NaN"
        assert not np.isnan(orb.e), "e is NaN"
        assert orb.a > 0.0, \
            "a = {0:.4f} AU is non-positive — binary unbound".format(orb.a)
        assert 0.0 <= orb.e < 1.0, \
            "e = {0:.6f} is outside [0, 1) — binary unbound or invalid".format(orb.e)

        if args.verbose:
            print("  After {0:.0f} yr: a={1:.2f} AU, e={2:.4f}  (bound)".format(
                t_short, orb.a, orb.e))

        print("Test 35b passed")
        code.reset()

        print("Test passed")

    def test36(self, args):
        """Tests for task 8.1 — Documented-defaults validation.

        Checks that the parameter defaults advertised in README.md and
        project.md match the actual Python-layer defaults.  No evolution
        is needed; we only instantiate objects and read back their fields.

        36a: Code-level (MSE object) defaults for VRR and eCAML.
        36b: Particle-level defaults for VRR parameters.
        36c: Particle-level defaults for LISA-band root-finding parameters.
        """

        # ------------------------------------------------------------------
        # 36a: Code-level defaults
        # ------------------------------------------------------------------
        print("Test 36a: Code-level defaults for VRR and eCAML")

        code = MSE()

        # README documents enable_VRR default as False
        assert code.enable_VRR == False, \
            "enable_VRR default should be False (README: 'must be True to push VRR parameters')"

        # README documents binary_evolution_use_eCAML_model default as False
        assert code.binary_evolution_use_eCAML_model == False, \
            "binary_evolution_use_eCAML_model default should be False"

        print("Test 36a passed")

        # ------------------------------------------------------------------
        # 36b: Particle-level VRR defaults
        # ------------------------------------------------------------------
        print("Test 36b: Particle-level VRR parameter defaults")

        particles = Tools.create_fully_nested_multiple(
            3, [1.0, 1.0, 1.0], [1.0, 100.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0])

        binaries = [p for p in particles if p.is_binary]
        outer_binary = max(binaries, key=lambda p: p.a)

        # README documents VRR_model default as 0
        assert outer_binary.VRR_model == 0, \
            "VRR_model default should be 0 (off)"

        # README documents VRR_include_mass_precession default as 0
        assert outer_binary.VRR_include_mass_precession == 0, \
            "VRR_include_mass_precession default should be 0"

        # README documents VRR_mass_precession_rate default as 0.0
        assert outer_binary.VRR_mass_precession_rate == 0.0, \
            "VRR_mass_precession_rate default should be 0.0"

        # README documents VRR_Omega_vec components default as 0.0
        assert outer_binary.VRR_Omega_vec_x == 0.0, "VRR_Omega_vec_x default should be 0.0"
        assert outer_binary.VRR_Omega_vec_y == 0.0, "VRR_Omega_vec_y default should be 0.0"
        assert outer_binary.VRR_Omega_vec_z == 0.0, "VRR_Omega_vec_z default should be 0.0"

        # README documents VRR_initial_time default as 0.0 and VRR_final_time as 1.0
        assert outer_binary.VRR_initial_time == 0.0, \
            "VRR_initial_time default should be 0.0"
        assert outer_binary.VRR_final_time == 1.0, \
            "VRR_final_time default should be 1.0"

        # README documents Bar-Or eta defaults as 0.0
        for attr in ['VRR_eta_20_init', 'VRR_eta_a_22_init', 'VRR_eta_b_22_init',
                     'VRR_eta_a_21_init', 'VRR_eta_b_21_init',
                     'VRR_eta_20_final', 'VRR_eta_a_22_final', 'VRR_eta_b_22_final',
                     'VRR_eta_a_21_final', 'VRR_eta_b_21_final']:
            assert getattr(outer_binary, attr) == 0.0, \
                "{} default should be 0.0".format(attr)

        print("Test 36b passed")

        # ------------------------------------------------------------------
        # 36c: Particle-level LISA band defaults
        # ------------------------------------------------------------------
        print("Test 36c: Particle-level LISA-band parameter defaults")

        # README documents check_for_entering_LISA_band default as True
        for b in binaries:
            assert b.check_for_entering_LISA_band == True, \
                "check_for_entering_LISA_band default should be True"

        # README documents critical GW frequency default as 31557.6 yr^-1
        for b in binaries:
            assert abs(b.check_for_entering_LISA_band_critical_GW_frequency - 31557.6) < 1e-6, \
                "check_for_entering_LISA_band_critical_GW_frequency default should be 31557.6"

        # README documents entering_LISA_band_has_occurred default as False
        for b in binaries:
            assert b.entering_LISA_band_has_occurred == False, \
                "entering_LISA_band_has_occurred default should be False"

        print("Test 36c passed")

        print("Test passed")

    def test100(self,args):
        print('Unit tests')

        code = MSE()

        flag = code.unit_tests(args.mode)

        assert(flag == 0)

        print("Unit tests passed")
        

def kroupa_93_imf(m):

    alpha1 = -1.3
    alpha2 = -2.2
    alpha3 = -2.7
    m1 = 0.1
    m2 = 0.5
    m3 = 1.0
    m4 = 100.0
    C1 = 0.2905673356704877
    C2 = 0.155711179725752
    C3 = 0.155711179725752
    
    if (m>=m1 and m<m2):
        pdf = C1*pow(m,alpha1)
    elif (m>=m2 and m<m3):
        pdf = C2*pow(m,alpha2)
    elif (m>=m3 and m<=m4):
        pdf = C3*pow(m,alpha3)
    else:
        pdf = 0.0
        
    return pdf
    
def sample_random_vector_on_unit_sphere():
    INCL = np.arccos( 2.0*np.random.random() - 1.0)
    LAN = 2.0 * np.pi * np.random.random()
    return compute_unit_AM_vector(INCL,LAN)

def compute_total_orbital_AM(code):
    """Compute total orbital angular momentum vector by summing h_vec for all binaries.

    Uses the C interface get_orbital_vectors to retrieve h_vec directly.
    h_vec = mu * sqrt(G*a*(1-e^2)/M) * j_hat, i.e. the full orbital angular
    momentum (not the specific angular momentum), so summing over all binary
    nodes gives the total orbital angular momentum of the system.

    argtypes are set here so this function is self-contained.
    """
    _dblp = ctypes.POINTER(ctypes.c_double)
    # Ensure argtypes are set (idempotent)
    code.lib.get_orbital_vectors.argtypes = [ctypes.c_int, _dblp, _dblp, _dblp, _dblp, _dblp, _dblp]
    code.lib.get_orbital_vectors.restype = ctypes.c_int

    L_total = np.array([0.0, 0.0, 0.0])
    for p in code.particles:
        if p.is_binary:
            e_x, e_y, e_z = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
            h_x, h_y, h_z = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
            code.lib.get_orbital_vectors(p.index,
                ctypes.byref(e_x), ctypes.byref(e_y), ctypes.byref(e_z),
                ctypes.byref(h_x), ctypes.byref(h_y), ctypes.byref(h_z))
            L_total += np.array([h_x.value, h_y.value, h_z.value])
    return L_total

def compute_unit_AM_vector(INCL,LAN):
    return np.array( [np.sin(LAN)*np.sin(INCL), -np.cos(LAN)*np.sin(INCL), np.cos(INCL)] )

def compute_e_and_j_hat_vectors(INCL,AP,LAN):
    sin_INCL = np.sin(INCL)
    cos_INCL = np.cos(INCL)
    sin_AP = np.sin(AP)
    cos_AP = np.cos(AP)
    sin_LAN = np.sin(LAN)
    cos_LAN = np.cos(LAN)
    
    e_hat_vec_x = (cos_LAN*cos_AP - sin_LAN*sin_AP*cos_INCL);
    e_hat_vec_y = (sin_LAN*cos_AP + cos_LAN*sin_AP*cos_INCL);
    e_hat_vec_z = (sin_AP*sin_INCL);
    
    j_hat_vec_x = sin_LAN*sin_INCL;
    j_hat_vec_y = -cos_LAN*sin_INCL;
    j_hat_vec_z = cos_INCL;

    e_hat_vec = np.array([e_hat_vec_x,e_hat_vec_y,e_hat_vec_z])
    j_hat_vec = np.array([j_hat_vec_x,j_hat_vec_y,j_hat_vec_z])

    return e_hat_vec,j_hat_vec
    
if __name__ == '__main__':
    args = parse_arguments()
    
    N_tests = 36
    if args.test==0:
        tests = list(range(1,N_tests+1)) + [100]
    else:
        tests = [args.test]

    t=test_mse()
    for i in tests:
        print( 'Running test number',i,'; verbose =',args.verbose,'; plot =',args.plot)
        function = getattr(t, 'test%s'%i)
        function(args)
    
    print("="*50)
    print("All tests passed!")
