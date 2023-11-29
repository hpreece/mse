
import numpy as np
import numpy.random as randomf

#import cPickle as pickle
import pickle
import numpy.random as randomf
import os,argparse,copy
from os import path

import math
from mpi4py import MPI
import scipy

CONST_R_SUN = 0.004649130343817401
CONST_G = 4.0*np.pi**2

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

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",                           type=int,       dest="model",                        default=1,              help="")
    parser.add_argument("--name",                            type=str,       dest="name",                           default="run01",              help="")    
    parser.add_argument("--seed",                            type=int,       dest="seed",                           default=0,              help="")    
    parser.add_argument("--N_MC",                            type=int,       dest="N_MC",                         default=10000,              help="")    
    parser.add_argument("--N_dir",                           type=int,       dest="N_dir",                         default=1000,              help="")    
    parser.add_argument("--istart",                          type=int,       dest="istart",                       default=0,              help="")    
    parser.add_argument("--iend",                            type=int,       dest="iend",                         default=1000000,              help="")    
    parser.add_argument("--logi",                            type=int,       dest="logi",                         default=0,              help="")    
    parser.add_argument("--logf",                            type=int,       dest="logf",                         default=17,              help="")
    ### boolean arguments ###
    add_bool_arg(parser, 'plot',                            default=True,          help="Make plots")
    add_bool_arg(parser, 'debug',                           default=False,         help="debugging mode")
    add_bool_arg(parser, 'specific',                        default=False,         help="run single specific system without MPI (istart)")
    add_bool_arg(parser, 'show',                            default=True,          help="Show plots")
    add_bool_arg(parser, 'plot_fancy',                      default=False,         help="Use LaTeX in plots")
    add_bool_arg(parser, 'verbose',                         default=False,         help="Verbose print output")
    add_bool_arg(parser, 'calc',                            default=True,          help="Make plots")
    add_bool_arg(parser, 'calc_new',                        default=False,         help="Recalculate all systems")
    add_bool_arg(parser, 'reversed',                        default=False,         help="Reverse system list")
    add_bool_arg(parser, 'print_memory_usage',              default=False,         help="Print memory usage info")
    
        
    args = parser.parse_args()

    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/figs')
    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/data')

    args.base_data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/' + args.name + '/'
    args.base_fig_filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/' + args.name + '/'

    mkdir_p(args.base_data_filename)
    mkdir_p(args.base_fig_filename)

    return args


def mkdir_p(path):
    import os,errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def remove_p(path):
    import os,errno
    try:
        os.remove(path)
    except OSError:
        pass
           
def save(data,filename):
    with open(filename,'wb') as file:
        pickle.dump(data,file)

def load(filename):
    with open(filename,'rb') as file:
        data = pickle.load(file)
    return data

def sample_distribution(y_lower,y_upper,y_distribution):
    random_number = randomf.random()
    
    if y_distribution[0] == "flat":
        y = y_lower + random_number*(y_upper-y_lower)
    elif y_distribution[0] == "log_flat":
        log_y = np.log(y_lower) + random_number*(np.log(y_upper) - np.log(y_lower))
        y = np.exp(log_y)
    elif y_distribution[0] == "power_law": ### assumes dN/dy ~ y^y_\alpha 
        alpha = y_distribution[1]
        if alpha==-1.0:
            y = y_lower*pow(y_upper/y_lower,random_number)
        else:
            y = pow( random_number*(pow(y_upper,alpha+1.0) - pow(y_lower,alpha+1.0)) + pow(y_lower,alpha+1.0), 1.0/(alpha + 1.0) )
    elif y_distribution[0] == "random_cosi":
        cos_y = 2.0*random_number - 1.0
        y = np.arccos(cos_y)
    elif y_distribution[0] == "gaussian":
        mu = y_distribution[1]
        sigma = y_distribution[2]
        y = randomf.normal(mu,sigma)
        #print 'mu',mu,'sigma',sigma
    elif y_distribution[0] == "gaussian_times_y": ### dN/dy ~ y exp[-y^2/(2sigma^2)]
        sigma = y_distribution[1]
        y = sigma*np.sqrt( -2.0*np.log(1.0 - random_number) )
    elif y_distribution[0] == "gaussian_pos":
        mu = y_distribution[1]
        sigma = y_distribution[2]

        y = -1.0
        while ( (y<0.0) or (y<y_lower) or (y>y_upper) ):
            y = randomf.normal(mu,sigma)
    elif y_distribution[0] == "maxwellian":

        mu = y_distribution[1]
        sigma = y_distribution[2]
        y = -1.0
        while ( (y<0.0) or (y<y_lower) or (y>y_upper) ):
            y1 = randomf.normal(mu,sigma)
            y2 = randomf.normal(mu,sigma)
            y3 = randomf.normal(mu,sigma)
            y = np.sqrt( y1**2 + y2**2 + y3**2 )

    elif y_distribution[0] == "rayleigh":
        ### dN/dx ~ x exp( -beta x^2 )
        beta = y_distribution[1]
        x = np.log( np.exp(-beta*y_lower**2) + random_number*(np.exp(-beta*y_upper**2) - np.exp(-beta*y_lower**2)) )
        y = np.sqrt( -x/beta )
    elif y_distribution[0] == "q_two_part_constant":
        q_c = y_distribution[1]
        alpha = y_distribution[2]
        x_c = q_c/(q_c + alpha*(1.0-q_c))
        if random_number <= x_c:
            y = random_number*(q_c + alpha*(1.0-q_c))
        else:
            y = ( q_c*(alpha-1.0) + random_number*(q_c + alpha*(1.0-q_c)) )/alpha
    elif y_distribution[0] == "sin_pi":
        y = (1.0/np.pi)*np.arccos( 1.0 - 2.0*random_number )
    elif y_distribution[0] == "three_component_broken_power_law":
        x = random_number
        alpha1 = y_distribution[1]
        alpha2 = y_distribution[2]
        alpha3 = y_distribution[3]
        m1 = y_distribution[4]
        m2 = y_distribution[5]
        m3 = y_distribution[6]
        m4 = y_distribution[7]

        alpha1_plus_1 = 1.0 + alpha1
        alpha2_plus_1 = 1.0 + alpha2
        alpha3_plus_1 = 1.0 + alpha3
        alpha1_plus_1_pm1 = 1.0/alpha1_plus_1
        alpha2_plus_1_pm1 = 1.0/alpha2_plus_1
        alpha3_plus_1_pm1 = 1.0/alpha3_plus_1

        m1_pow_alpha1_plus_one = pow(m1,alpha1_plus_1)
        m2_pow_alpha1_plus_one = pow(m2,alpha1_plus_1)
        m2_pow_alpha2_plus_one = pow(m2,alpha2_plus_1)
        m3_pow_alpha2_plus_one = pow(m3,alpha2_plus_1)
        m3_pow_alpha3_plus_one = pow(m3,alpha3_plus_1)
        m4_pow_alpha3_plus_one = pow(m4,alpha3_plus_1)

        C1 = 1.0/( (1.0/alpha1_plus_1)*(m2_pow_alpha1_plus_one - m1_pow_alpha1_plus_one) + pow(m2,alpha1-alpha2)*(1.0/alpha2_plus_1)*(m3_pow_alpha2_plus_one - m2_pow_alpha2_plus_one) + pow(m2,alpha1-alpha2)*pow(m3,alpha2-alpha3)*(1.0/alpha3_plus_1)*(m4_pow_alpha3_plus_one - m3_pow_alpha3_plus_one) )
        C2 = C1 * pow(m2,alpha1-alpha2)
        C3 = C2 * pow(m3,alpha2-alpha3)

        x1 = (C1/alpha1_plus_1)*( m2_pow_alpha1_plus_one - m1_pow_alpha1_plus_one)
        x2 = x1 + (C2/alpha2_plus_1)*( m3_pow_alpha2_plus_one - m2_pow_alpha2_plus_one)
        x3 = x2 + (C3/alpha3_plus_1)*( m4_pow_alpha3_plus_one - m3_pow_alpha3_plus_one)
        
        if (x >= 0.0 and x < x1):
            y = pow( x*alpha1_plus_1/C1 + m1_pow_alpha1_plus_one, alpha1_plus_1_pm1)
        elif (x >= x1 and x < x2):
            y = pow( (x - x1)*alpha2_plus_1/C2 + m2_pow_alpha2_plus_one, alpha2_plus_1_pm1)
        elif (x >= x2 and x <= x3):
            y = pow( (x - x2)*alpha3_plus_1/C3 + m3_pow_alpha3_plus_one, alpha3_plus_1_pm1)

    else:
        print('sample_distribution -- unknown y_distribution ', y_distribution[0],' -- exiting')
        exit(-1)
    return y


def test_sample_distribution(y_distribution):
    if y_distribution[0] == "three_component_broken_power_law":
        N = 10000
        m_distribution = ["three_component_broken_power_law",-1.3, -2.2, -2.7, 0.1, 0.5, 1.0, 100.0]
        ms = []
        for i in range(N):
            m = sample_distribution(0.1,100.0,m_distribution)
            ms.append(m)

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
        
    else:
        print("test_sample_distribution with  y_distribution[0] = %s not supported; exiting"%y_distribution[0])
        exit(0)

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
    
def compute_Eggleton_fq(q):
    ### q is defined as m_primary/m_secondary ###
    q_pow_one_third = pow(q,1.0/3.0)
    q_pow_two_third = q_pow_one_third*q_pow_one_third
    return 0.49*q_pow_two_third/(0.6*q_pow_two_third + np.log(1.0 + q_pow_one_third))

def estimate_MS_radius_RSun(M_MSun):
    return pow(M_MSun,0.7)

def check_for_dynamical_stability_MA(binaries):
    stable = True
    for index,b in enumerate(binaries):
        #print("Test",b.child1.mass,b.child2.mass,b.child1.is_binary,b.child2.is_binary)
        a_out = b.a
        e_out = b.e
        rp_out = a_out*(1.0-e_out)
        if b.child1.is_binary == True:
            q_out = b.child2.mass/b.child1.mass
            i_rel = compute_mutual_inclination(b.INCL,b.child1.INCL,b.LAN,b.child1.LAN)
            rp_out_crit = b.child1.a*2.8*pow( (1.0 + q_out)*(1.0 + e_out)/np.sqrt(1.0-e_out),2.0/5.0 ) * (1.0 - 0.3*i_rel/np.pi)
            if rp_out < rp_out_crit:
                stable = False
        if b.child2.is_binary == True:
            q_out = b.child1.mass/b.child2.mass
            i_rel = compute_mutual_inclination(b.INCL,b.child2.INCL,b.LAN,b.child2.LAN)
            rp_out_crit = b.child2.a*2.8*pow( (1.0 + q_out)*(1.0 + e_out)/np.sqrt(1.0-e_out),2.0/5.0 ) * (1.0 - 0.3*i_rel/np.pi)
            if rp_out < rp_out_crit:
                stable = False
    return stable

def compute_t_LK_min(binaries):
    t_LK = 1.0e10
    for index,b in enumerate(binaries):
        e_out = b.e
        m_tot = b.mass
        P_out = 2.0*np.pi*np.sqrt(b.a**3/(CONST_G*m_tot))
        #print("e_out",e_out,"m_tot",m_tot,"P_out",P_out)
        t_LK1 = 1.0e10
        if b.child1.is_binary==True:
            m_in = b.child1.mass
            m_out = b.child2.mass
            P_in = 2.0*np.pi*np.sqrt(b.child1.a**3/(CONST_G*m_in))
            t_LK1 = (P_out/P_in)*P_out*(m_tot/m_out)*pow(1.0-e_out**2,3.0/2.0)
            #print("t_LK1",t_LK1,"m_in",m_in,"m_out",m_out,"P_in",P_in)
        t_LK2 = 1.0e10
        if b.child2.is_binary==True:
            m_in = b.child2.mass
            m_out = b.child1.mass
            P_in = 2.0*np.pi*np.sqrt(b.child2.a**3/(CONST_G*m_in))
            t_LK2 = (P_out/P_in)*P_out*(m_tot/m_out)*pow(1.0-e_out**2,3.0/2.0)
            #print("t_LK2",t_LK2,"m_in",m_in,"m_out",m_out,"P_in",P_in)
        t_LK = np.amin([t_LK,t_LK1,t_LK2])
    #print("t_LK",t_LK*1e-6)
    return t_LK

def compute_t_LK(G,m1,m2,m3,a_in,a_out,e_out):
    P_in = compute_orbital_period(G,m1+m2,a_in)
    P_out = compute_orbital_period(G,m1+m2+m3,a_out)
    return (P_out/P_in)*P_out*((m1+m2+m3)/m3)*pow(1.0-e_out**2,3.0/2.0)
    
def compute_1PN_timescale(G,c,M,a,e):
    P = compute_orbital_period(G,M,a)
    rg = G * M/(c**2)
    return (1.0/3.0) * P * (a/rg) * (1.0 - e**2)

def compute_orbital_period(G,M,a):
    return 2.0*np.pi*np.sqrt(a**3/(G*M))
    
def compute_semimajor_axis(G,M,P):
    return pow( (P/(2.0*np.pi))**2 * G * M, 1.0/3.0 )

def extract_sma_ratios_from_system(binaries,use_rp_out = True):

    sma_ratios = []
    for index_binary,b in enumerate(binaries):
        r_out = b.a
        if use_rp_out==True:
            r_out *= (1.0-b.e)

        if b.child1.is_binary==True:
            a_in = b.child1.a
            sma_ratios.append(r_out/a_in)
        if b.child2.is_binary==True:
            a_in = b.child2.a
            sma_ratios.append(r_out/a_in)
            
    return sma_ratios

def extract_sma_ratios_from_parent(binary,use_rp_out = True):
    a_in = binary.a

    parent = binary.parent
    r_out = parent.a
    if use_rp_out==True:
        r_out *= (1.0-parent.e)

    sma_ratios = []
    sma_ratios.append(r_out/a_in)
            
    return sma_ratios
    
def extract_mutual_inclinations_from_system(binaries):

    mutual_INCLs = []
    for index_binary,b in enumerate(binaries):
        INCL_out = b.INCL
        LAN_out = b.LAN

        if b.child1.is_binary==True:
            INCL_in = b.child1.INCL
            LAN_in = b.child1.LAN
            mutual_INCLs.append( compute_mutual_inclination(INCL_in,INCL_out,LAN_in,LAN_out) )
        if b.child2.is_binary==True:
            INCL_in = b.child2.INCL
            LAN_in = b.child2.LAN
            mutual_INCLs.append( compute_mutual_inclination(INCL_in,INCL_out,LAN_in,LAN_out) )

    return mutual_INCLs

def extract_mutual_inclinations_from_parent(binary):

    INCL_in = binary.INCL
    LAN_in = binary.LAN

    mutual_INCLs = []

    parent = binary.parent
    INCL_out = parent.INCL
    LAN_out = parent.LAN
    
    mutual_INCLs.append( compute_mutual_inclination(INCL_in,INCL_out,LAN_in,LAN_out) )
    
    return mutual_INCLs
   

def compute_mutual_inclination(INCL_k,INCL_l,LAN_k,LAN_l):
    cos_INCL_rel = np.cos(INCL_k)*np.cos(INCL_l) + np.sin(INCL_k)*np.sin(INCL_l)*np.cos(LAN_k-LAN_l)
    return np.arccos(cos_INCL_rel)

def give_CDF(data):
    sorted_data = numpy.sort( data )
    yvals = numpy.arange(len(sorted_data))/float(len(sorted_data))
    
    sorted_data = list(sorted_data)
    yvals = list(yvals)
    
    sorted_data.append( sorted_data[-1] )
    yvals.append(1.0)
    
    return numpy.array(sorted_data),numpy.array(yvals)

def compute_unit_AM_vector(INCL,LAN):
    return np.array( [np.sin(LAN)*np.sin(INCL), -np.cos(LAN)*np.sin(INCL), np.cos(INCL)] )

def compute_unit_e_vector(INCL,LAN,AP):
    return np.array( [
        np.cos(LAN) * np.cos(AP) - np.sin(LAN) * np.sin(AP) * np.cos(INCL), \
        np.sin(LAN) * np.cos(AP) + np.cos(LAN) * np.sin(AP) * np.cos(INCL), \
        np.sin(AP) * np.sin(INCL)] )
        

def get_filename_for_system(args,index_system):
    N_dir = args.N_dir
    i_dir = int(index_system/N_dir)
    #imax = int(index_system/1000.0)
    #dirname = args.base_data_filename + 'i_dir_' + str(i_dir) + '_model_' + str(args.model) + '/'
    dirname = args.base_data_filename + 'i_dir_' + str(i_dir) + '/'
    filename = dirname + 'system_' + str(index_system) + '.pkl'
    #print(filename)
    mkdir_p(dirname)
    return dirname,filename

def get_plot_filename_for_system(args,index_system):
    N_dir = args.N_dir
    i_dir = int(index_system/N_dir)
    #imax = int(index_system/1000.0)
    
    #dirname = args.base_data_filename + 'i_dir_' + str(i_dir) + '_model_' + str(args.model) + '/'
    dirname = args.base_fig_filename + 'i_dir_' + str(i_dir) + '/'
    filename = dirname + 'system_' + str(index_system)
    
    mkdir_p(dirname)
    return dirname,filename
    
def sample_random_vector():
    INCL = np.arccos( 2.0*np.random.random() - 1.0)
    LAN = 2.0 * np.pi * np.random.random()
    return compute_unit_AM_vector(INCL,LAN)

def stability_criterion_MA01(m1,m2,m3,a_in,a_out,e_out,i_rel):
    q_out = m3/(m1+m2)
    rp_out = a_out*(1.0 - e_out)
    rp_out_crit = a_in * 2.8*pow( (1.0 + q_out)*(1.0 + e_out)/np.sqrt(1.0-e_out),2.0/5.0 ) * (1.0 - 0.3*i_rel/np.pi)
    if (rp_out < rp_out_crit):
        stable = False
    else:
        stable = True
    
    return stable

def moe_di_stefano_f_log_P(log10_M1,log_P):
    gs = moe_di_stefano_gamma_small(log10_M1,log_P)
    gl = moe_di_stefano_gamma_large(log10_M1,log_P)
    F_twin =  moe_di_stefano_F_twin(log10_M1,log_P)
    
    #gs=0
    #gl=0
    #F_twin=0
    
    gsp1 = gs + 1.0
    glp1 = gl + 1.0
    #print("G",gamma_small,gamma_large,F_twin)
    q1 = 0.1
    q2 = 0.3
    q3 = 1.0
    
    q_factor = (1.0 - F_twin) * (1.0/( pow(q3,glp1) - pow(q2,glp1) )) * (glp1) * pow(q2,gl-gs) * ( (1.0/gsp1) * ( pow(q2,gsp1) - pow(q1,gsp1) ) + pow(q2,gs-gl) * (1.0/(1.0 - F_twin)) * (1.0/glp1) * ( pow(q3,glp1) - pow(q2,glp1) ) )
    f_large = moe_di_stefano_f_log_P_high_q(log10_M1,log_P) 
    
    #print("log10_M1,log_P",log10_M1,log_P,"q",q_factor,"f_large",f_large)
    return f_large * q_factor, f_large, gs, gl, F_twin

def moe_di_stefano_f_log_P_high_q(log10_M1,log_P):
    #alpha = 0.02
    #Delta_log_P = 0.8

    alpha = 0.018
    Delta_log_P = 0.7
    
    mds1 = moe_di_stefano_f_log_P_aux1(log10_M1)
    mds2 = moe_di_stefano_f_log_P_aux2(log10_M1)
    mds3 = moe_di_stefano_f_log_P_aux3(log10_M1)
    
    if log_P >= 0.2 and log_P < 1.0:
        return mds1
    if log_P >= 1.0 and log_P < 2.7 - Delta_log_P:
        return mds1 + (log_P - 1.0)/(1.7 - Delta_log_P) * (mds2 - mds1 - alpha*Delta_log_P)
    if log_P >= 2.7 - Delta_log_P and log_P < 2.7 + Delta_log_P:
        return mds2 + alpha * (log_P - 2.7)
    if log_P > 2.7 + Delta_log_P and log_P < 5.5:
        return mds2 + alpha*Delta_log_P + ((log_P - 2.7 - Delta_log_P)/(2.8 - Delta_log_P)) * ( mds3 - mds2 - alpha*Delta_log_P )
    if log_P >= 5.5 and log_P <= 8.0:
        return mds3 * np.exp(-0.3 * (log_P - 5.5) )
"""
def moe_di_stefano_f_log_P_aux1(log10_M1):
    return 0.018 + 0.04*log10_M1 + 0.07*log10_M1**2

def moe_di_stefano_f_log_P_aux2(log10_M1):
    return 0.034 + 0.09*log10_M1

def moe_di_stefano_f_log_P_aux3(log10_M1):
    return 0.081 - 0.08*log10_M1 + 0.06*log10_M1**2
"""

def moe_di_stefano_f_log_P_aux1(log10_M1):
    return 0.020 + 0.04*log10_M1 + 0.07*log10_M1**2

def moe_di_stefano_f_log_P_aux2(log10_M1):
    return 0.039 + 0.07*log10_M1 + 0.01*log10_M1**2

def moe_di_stefano_f_log_P_aux3(log10_M1):
    return 0.078 - 0.05*log10_M1 + 0.04*log10_M1**2
    
def moe_di_stefano_F_twin(log10_M1,log_P):
    M1 = pow(10.0,log10_M1)
    
    if M1 <= 6.5:
        log_P_twin = 8.0 - M1
    else:
        log_P_twin = 1.5

    F_twin_small_P = 0.3 - 0.15*log10_M1
    
    if log_P < 1.0:
        return F_twin_small_P
    if log_P >= 1.0 and log_P < log_P_twin:
        return F_twin_small_P * (1.0 - (log_P - 1.0)/(log_P_twin - 1.0) )
    if log_P >= log_P_twin:
        return 0.0

def moe_di_stefano_gamma_large(log10_M1,log_P):
    M1 = pow(10.0,log10_M1)
    
    if M1 > 0.8 and M1 <= 1.2:
        return moe_di_stefano_gamma_large_aux1(log_P)
    if M1 > 1.2 and M1 < 3.5:
        y1 = moe_di_stefano_gamma_large_aux1(log_P)
        y2 = moe_di_stefano_gamma_large_aux2(log_P)
        x1 = 1.2
        x2 = 3.5
        return y1 + (M1 - x1) * (y2-y1)/(x2-x1)
    if M1 >= 3.5 and M1 < 6.0:
        y1 = moe_di_stefano_gamma_large_aux2(log_P)
        y2 = moe_di_stefano_gamma_large_aux3(log_P)
        x1 = 3.5
        x2 = 6.0
        return y1 + (M1 - x1) * (y2-y1)/(x2-x1)
    
    if M1 > 6.0:
        return moe_di_stefano_gamma_large_aux3(log_P)

def moe_di_stefano_gamma_large_aux1(log_P):
    if log_P >= 0.2 and log_P < 5.0:
        return -0.5
    if log_P >= 5.0 and log_P <= 8.0:
        return -0.5 - 0.3*(log_P - 5.0)

def moe_di_stefano_gamma_large_aux2(log_P):
    if log_P >= 0.2 and log_P < 1.0:
        return -0.5
    if log_P >= 1.0 and log_P < 4.5:
        return -0.5 - 0.2*(log_P - 1.0)
    if log_P >= 4.5 and log_P < 6.5:
        return -1.2 - 0.4*(log_P - 4.5)
    if log_P >= 6.5 and log_P <= 8.0:
        return -2.0
    
def moe_di_stefano_gamma_large_aux3(log_P):
    if log_P >= 0.0 and log_P < 1.0:
        return -0.5
    if log_P >= 1.0 and log_P < 2.0:
        return -0.5 - 0.9*(log_P - 1.0)
    if log_P >= 2.0 and log_P < 4.0:
        return -1.4 - 0.3*(log_P - 2.0)
    if log_P >= 4.0 and log_P <= 8.0:
        return -2.0   

def moe_di_stefano_gamma_small(log10_M1,log_P):
    M1 = pow(10.0,log10_M1)
    
    if M1 > 0.8 and M1 <= 1.2:
        return moe_di_stefano_gamma_small_aux1(log_P)
    if M1 > 1.2 and M1 < 3.5:
        y1 = moe_di_stefano_gamma_small_aux1(log_P)
        y2 = moe_di_stefano_gamma_small_aux2(log_P)
        x1 = 1.2
        x2 = 3.5
        return y1 + (M1 - x1) * (y2-y1)/(x2-x1)
    if M1 >= 3.5 and M1 < 6.0:
        y1 = moe_di_stefano_gamma_small_aux2(log_P)
        y2 = moe_di_stefano_gamma_small_aux3(log_P)
        x1 = 3.5
        x2 = 6.0
        return y1 + (M1 - x1) * (y2-y1)/(x2-x1)
    
    if M1 > 6.0:
        return moe_di_stefano_gamma_small_aux3(log_P)

def moe_di_stefano_gamma_small_aux1(log_P):
    if log_P >= 0.2 and log_P <= 8.0:
        return 0.3

def moe_di_stefano_gamma_small_aux2(log_P):
    if log_P >= 0.2 and log_P < 2.5:
        return 0.2
    if log_P >= 2.5 and log_P < 5.5:
        return 0.2 - 0.3*(log_P - 2.5)
    if log_P >= 5.5 and log_P <= 8.0:
        return -0.7 - 0.2*(log_P - 5.5)
    
def moe_di_stefano_gamma_small_aux3(log_P):
    if log_P >= 0.0 and log_P < 1.0:
        return 0.1
    if log_P >= 1.0 and log_P < 3.0:
        return 0.1 - 0.15*(log_P - 1.0)
    if log_P >= 3.0 and log_P < 5.6:
        return -0.2 - 0.5*(log_P - 3.0)
    if log_P >= 5.6 and log_P <= 8.0:
        return -1.5

def moe_di_stefano_sample_binary_properties(log10_M1):
    log_P,gs,gl,F_twin = moe_di_stefano_sample_log_P(log10_M1)
        
    q = moe_di_stefano_sample_q(gs,gl,F_twin)[0]
        
    e = moe_di_stefano_sample_e(log10_M1,log_P)[0]
    
    return log_P,q,e

def moe_di_stefano_sample_log_P(log10_M1,method="rejection"):
    
    method="rejection"
    #method="inverse_cdf_num"
    #method="inverse_cdf_spline"
    
    if method == "rejection":
        ### Sample log_P using rejection method ###

        log_P_min = 0.2
        log_P_max = 8.0
        
        N=100
        points = np.linspace(log_P_min,log_P_max,N)
        
        f_max = -1
        log_P_maximum = -1
        for i,x in enumerate(points):
            
            #f_x = f(x)
            f_x = moe_di_stefano_f_log_P(log10_M1,x)[0]
            #print("x",x,f_x)
            if f_x > f_max:
                f_max = f_x
                log_P_maximum = x
        #print("f_max",f_max,"log_P_maximum",log_P_maximum)
        sampled = False
        N=0

        while sampled == False:
            y = randomf.random()
            
            log_P = log_P_min + randomf.random()*(log_P_max-log_P_min)
            f_x,f_high,gs,gl,F_twin = moe_di_stefano_f_log_P(log10_M1,log_P)
            if f_x/f_max >= y:
                sampled = True
                
    elif method == "inverse_cdf_num":
        log_P_min = 0.2
        log_P_max = 8.0
        
        N=200
        points = np.linspace(log_P_min,log_P_max,N)
        dp = (points[1] - points[0])
        
        integral = 0.0
        fs = []
        for i,log_P in enumerate(points):
            f = moe_di_stefano_f_log_P(log10_M1,log_P)[0]
            fs.append(f)
            integral += f * dp
        fs = np.array(fs)
        pdf = fs/integral
        
        y = randomf.random()
        
        I = 0.0
        cdf = []
        
        for i,log_P in enumerate(points):
            I += pdf[i] * dp
            cdf.append(I)
            #print(y,I)
            if I > y:
                break
        #print("P",log_P)
        f_x,f_high,gs,gl,F_twin = moe_di_stefano_f_log_P(log10_M1,log_P)

        if 1==0:
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.plot(points,cdf)
            pyplot.show()


    elif method == "inverse_cdf_spline":
        log_P_min = 0.2
        log_P_max = 8.0
        
        N=100
        points = np.linspace(log_P_min,log_P_max,N)
        dp = (points[1] - points[0])
        
        integral = 0.0
        fs = []
        for i,log_P in enumerate(points):
            f = moe_di_stefano_f_log_P(log10_M1,log_P)[0]
            fs.append(f)
            integral += f * dp
        fs = np.array(fs)
        pdf = fs/integral
        
        from scipy import interpolate
        
        tck = interpolate.splrep(points, pdf, s=0)
        
        y = randomf.random()
        if y > 0.998:
            y = 0.998
        I = 0.0
        cdf = []
        
        #for i,log_P in enumerate(points):
        #    I += pdf[i] * dp
        #    cdf.append(I)
            #print(y,I)
            #if I > y:
            #    break
        #print("P",log_P)
        #f_x,f_high,gs,gl,F_twin = moe_di_stefano_f_log_P(log10_M1,log_P)

        f = lambda x: interpolate.splint(log_P_min, x, tck)
        f_root = lambda x: interpolate.splint(log_P_min, x, tck) - y
        
        f_root_prime = lambda x: interpolate.splev(x, tck, der=0)
        f_root_prime2 = lambda x: interpolate.splev(x, tck, der=1)
        #tck_int = interpolate.splantider(tck, n=1)
        #interp_fn2 = lambda x: interpolate.splev(x, tck, der=0)
        #print("T",interp_fn2(log_P_max))

        from scipy import optimize

        try:
        
            root = optimize.ridder(f_root, log_P_min, log_P_max)
        except ValueError:
            try:
                root = optimize.newton(f_root, log_P_min, tol=0.01, fprime=f_root_prime,fprime2 = f_root_prime2,maxiter=1000)
            except RuntimeError:
                fig=pyplot.figure()
                plot=fig.add_subplot(1,1,1)
                #plot.scatter(points,cdf)
                
                xnew = np.linspace(log_P_min,log_P_max,10000)
                #ynew = interpolate.splev(xnew, tck, der=0)
                plot.plot(xnew,[f(x) for x in xnew],color='r')
                #plot.plot(xnew,[f_root(x) for x in xnew],color='y')
                plot.axhline(y=y,color='g')
                #plot.axvline(x=root,color='r')
                
                pyplot.show()
        
        #print("R",root)    
        
        
        if 1==0:
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.scatter(points,cdf)
            
            xnew = np.linspace(log_P_min,log_P_max,10000)
            #ynew = interpolate.splev(xnew, tck, der=0)
            plot.plot(xnew,[f(x) for x in xnew],color='r')
            #plot.plot(xnew,[f_root(x) for x in xnew],color='y')
            plot.axhline(y=y,color='g')
            plot.axvline(x=root,color='r')
            
            pyplot.show()
                
        #print("f",f(log_P_max))

        #roots = interpolate.sproot(tck_int, mest=1)

        log_P = root
        
        if log_P>8.0:
            log_P = 8.0
        f_x,f_high,gs,gl,F_twin = moe_di_stefano_f_log_P(log10_M1,log_P)

        #exit(0)
        
        
    return log_P,gs,gl,F_twin
    
    
def moe_di_stefano_sample_q(gs,gl,F_twin):
    
    ### Sample q using inverse CDF method ###
    ### Preamble
    gsp1 = gs + 1.0
    glp1 = gl + 1.0
    gsp1_inv = 1.0/gsp1
    glp1_inv = 1.0/glp1
    
    q1 = 0.1
    q2 = 0.3
    q3 = 0.95
    q4 = 1.0
    
    q1_pow_gsp1 = pow(q1,gsp1)
    q2_pow_gsp1 = pow(q2,gsp1)
    q2_pow_glp1 = pow(q2,glp1)
    q3_pow_glp1 = pow(q3,glp1)
    q4_pow_glp1 = pow(q4,glp1)
    
    I1 = gsp1_inv * ( q2_pow_gsp1 - q1_pow_gsp1)
    I2 = glp1_inv * ( q3_pow_glp1 - q2_pow_glp1)
    I3 = glp1_inv * ( q4_pow_glp1 - q3_pow_glp1)
    I4 = I2 + I3
    
    C1 = 1.0/(I1 + I4*(q2_pow_gsp1/q2_pow_glp1) * (1.0/(1.0 - F_twin)) )
    C2 = C1 * q2_pow_gsp1/q2_pow_glp1
    C3 = C2 * ( I3 + F_twin * I2 )/( I3 * (1.0 - F_twin) )
    
    A1 = C1 * I1
    A2 = A1 + C2 * I2
    A3 = A2 + C3 * I3
    
    #print("As",A1,A2,A3)
    
    ### Sample q
    y = randomf.random()
    if y >= 0.0 and y < A1:
        q = pow( (y*gsp1/C1) + q1_pow_gsp1, gsp1_inv)
    if y >= A1 and y < A2:
        q = pow( ((y - A1)*glp1/C2) + q2_pow_glp1, glp1_inv)
    if y >= A2 and y < A3:
        q = pow( ((y - A2)*glp1/C3) + q3_pow_glp1, glp1_inv)
    
    return q, C1, C2, C3, q1, q2, q3, q4
    
def moe_di_stefano_sample_e(log10_M1,log_P):
    
    e_max = moe_di_stefano_sample_e_e_max(log_P)
    
    eta = moe_di_stefano_sample_e_eta(log10_M1,log_P)
    if (eta < -5.0):
        eta = -5.0
    
    etap1 = eta + 1.0
    
    y = randomf.random()

    e1 = 1.0e-10
    e2 = e_max

    e = pow( y*(pow(e2,etap1) - pow(e1,etap1)) + pow(e1,etap1), 1.0/etap1)
    #e = e_max * pow(y, 1.0/etap1)

    if (e>=1.0):
        print("ERROR in moe_di_stefano_sample_e")
        print("emax",e_max,"eta",eta,e)
        exit(0)
    return e,eta,e_max
    
def moe_di_stefano_sample_e_e_max(log_P):
    P_day = pow(10.0,log_P)
    if P_day < 2.0:
        return 0.0
    else:
        return 1.0 - pow(P_day/2.0,-2.0/3.0)
    
def moe_di_stefano_sample_e_eta(log10_M1,log_P):
    M1 = pow(10.0,log10_M1)

    if M1 > 0.8 and M1 <= 3.0:
        return moe_di_stefano_sample_e_aux1(log_P)
    if M1 > 3.0 and M1 <= 7.0:
        y1 = moe_di_stefano_sample_e_aux1(log_P)
        y2 = moe_di_stefano_sample_e_aux2(log_P)
        x1 = 3.0
        x2 = 7.0
        return y1 + (M1 - x1) * (y2-y1)/(x2-x1)
    if M1 > 7.0:
        return moe_di_stefano_sample_e_aux2(log_P)
    
def moe_di_stefano_sample_e_aux1(log_P):
    return 0.6 - 0.7/(log_P - 0.5)

def moe_di_stefano_sample_e_aux2(log_P):
    return 0.9 - 0.2/(log_P - 0.5)

def moe_di_stefano_test_sample_q(gs,gl,F_twin):
    
    qs = []
    N_MC = 100000
    for i in range(N_MC):
        q, C1, C2, C3, q1, q2, q3, q4 = moe_di_stefano_sample_q(gs,gl,F_twin)
        qs.append(q)
    
    labelsize=20
    fontsize=20

    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="linear")

    plot.hist(qs,bins=np.linspace(0.0,1.0,50),histtype='step',color='tab:red',density=True,label="Monte Carlo")

    N=100
    q_points1 = np.linspace(q1,q2,N)
    q_points2 = np.linspace(q2,q3,N)
    q_points3 = np.linspace(q3,q4,N)
    pdf1s = C1 * pow(q_points1,gs)
    pdf2s = C2 * pow(q_points2,gl)
    pdf3s = C3 * pow(q_points3,gl)

    plot.plot(q_points1,pdf1s,color='k',label="Analytic")
    plot.plot(q_points2,pdf2s,color='k')
    plot.plot(q_points3,pdf3s,color='k')
    #plot.axvline(x = q3, ymin = C2*pow(q3,gl), ymax = C3*pow(q3,gl), color='k')
    plot.plot((q1, q1), (0, C1*pow(q1,gs)), color='k')
    plot.plot((q3, q3), (C2*pow(q3,gl), C3*pow(q3,gl)), color='k')

    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.55*fontsize)
    
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot.set_xlabel("$q$",fontsize=fontsize)
    plot.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)

    pyplot.show()

def moe_di_stefano_test_sample_e():
    lms = np.log10(np.array([1.0,5.0]))
    
    etas = [[] for x in range(len(lms))]
    es = [[] for x in range(len(lms))]
    es_eta = [[] for x in range(len(lms))]
    es_e_max = [[] for x in range(len(lms))]
    log_Ps = np.linspace(0.6,4.5,100)
    
    N_MC = 100000
    for j,log10_M1 in enumerate(lms):

        for i,log_P in enumerate(log_Ps):
            e,eta,e_max = moe_di_stefano_sample_e(log10_M1,log_P)
            etas[j].append(eta)

    
        log_P = 1.0
        for i in range(N_MC):
            e,eta,e_max = moe_di_stefano_sample_e(log10_M1,log_P)
            es[j].append(e)
            es_eta[j] = eta
            es_e_max[j] = e_max

    labelsize=20
    fontsize=20

    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="linear")

    colors = ['tab:red','tab:blue']
    for j in range(len(lms)):
        plot.plot(log_Ps,etas[j],color=colors[j])


    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.55*fontsize)
    
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot.set_xlabel("$\mathrm{log}_{10}(P/\mathrm{d})$",fontsize=fontsize)
    plot.set_ylabel("$\eta$",fontsize=fontsize)

    plot.set_ylim(-1.0,1.5)
    
    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="log")
    
    for j in range(len(lms)):
        plot.hist(es[j],bins=np.linspace(0.0,1.0,50),histtype='step',density=True,color=colors[j])
        eta = es_eta[j]
        e_max = es_e_max[j]
        points = np.linspace(0.0,e_max,1000)
        plot.plot(points,(eta+1.0)*pow(points/e_max,eta)/e_max,color='k')
        
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot.set_xlabel("$e$",fontsize=fontsize)
    plot.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)

    pyplot.show()


def test_moe_di_stefano():
    import time
    t0 = time.time()
    
    #moe_di_stefano_test_sample_q(0.3,-0.5,0.01)
    #moe_di_stefano_test_sample_q(0.0,0.0,0.0) ### completely flat
    
    #moe_di_stefano_test_sample_e()
    
    #lms = np.log10(np.array([1.0,3.5,7.5,12.5,20]))
    lms = np.log10(np.array([1.0,11.0,17]))
    #lms = np.log10(np.array([17.0])) ### log masses
    
    fs = [[] for x in range(len(lms))]
    f_highs = [[] for x in range(len(lms))]
    lPs = np.linspace(0.2,7.9,100)
    
    F_twins = [[] for x in range(len(lms))]
    gammas_small = [[] for x in range(len(lms))]
    gammas_large = [[] for x in range(len(lms))]

    int_f = [0.0 for x in range(len(lms))]
    int_f_high = [0.0 for x in range(len(lms))]
    for i,log10_M1 in enumerate(lms):
        for log_P in lPs:
            f,f_high,gs,gl,F_twin = moe_di_stefano_f_log_P(log10_M1,log_P)
            fs[i].append( f )
            f_highs[i].append( f_high )
            
            int_f[i] += f * (lPs[1] - lPs[0])
            int_f_high[i] += f_high * (lPs[1] - lPs[0])
            
            F_twin = moe_di_stefano_F_twin(log10_M1,log_P)
            F_twins[i].append(F_twin)
    
            gamma_large = moe_di_stefano_gamma_large(log10_M1,log_P)
            gammas_large[i].append(gamma_large)

            gamma_small = moe_di_stefano_gamma_small(log10_M1,log_P)
            gammas_small[i].append(gamma_small)

    labelsize=20
    fontsize=20

    colors = ['tab:red','tab:orange','tab:green','tab:blue','tab:purple']

    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="linear")

    fig2=pyplot.figure(figsize=(8,6))
    plot2=fig2.add_subplot(1,1,1,yscale="linear")

    fig3=pyplot.figure(figsize=(8,6))
    plot3=fig3.add_subplot(1,1,1,yscale="linear")
    
    for i,log10_M1 in enumerate(lms):
        label = r"$M_1 = %s\,\mathrm{M}_\odot$"%round(pow(10.0,log10_M1),1)
        plot.plot(lPs,np.array(f_highs)[i]/int_f_high[i],color=colors[i],linestyle='dotted')
        plot.plot(lPs,np.array(fs)[i]/int_f[i] ,color=colors[i],linestyle='solid')
        
        log_Ps = []
        qs = []
        es = []
        N_MC = 10000
        for i_MC in range(N_MC):
            log_P,q,e = moe_di_stefano_sample_binary_properties(log10_M1)
            log_Ps.append(log_P)
            qs.append(q)
            es.append(e)
            
        plot.hist(log_Ps,bins=np.linspace(0.0,8.0,30),histtype='step',density=True,color=colors[i],label=label)
        plot2.hist(qs,bins=np.linspace(0.0,1.0,30),histtype='step',density=True,color=colors[i],label=label)
        plot3.hist(es,bins=np.linspace(0.0,1.0,30),histtype='step',density=True,color=colors[i],label=label)
    

    plot.set_xlabel("$\mathrm{log}_{10}(P/\mathrm{d})$",fontsize=fontsize)
    plot.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)

    plot2.set_xlabel("$q$",fontsize=fontsize)
    plot2.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)

    plot3.set_xlabel("$e$",fontsize=fontsize)
    plot3.set_ylabel("$\mathrm{PDF}$",fontsize=fontsize)

    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.55*fontsize)

    handles,labels = plot2.get_legend_handles_labels()
    plot2.legend(handles,labels,loc="best",fontsize=0.55*fontsize)

    handles,labels = plot3.get_legend_handles_labels()
    plot3.legend(handles,labels,loc="best",fontsize=0.55*fontsize)
    
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot2.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot3.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    
    """
    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="linear")
    for i,log10_M1 in enumerate(lms):
        plot.plot(lPs,F_twins[i],color=colors[i],linestyle='solid')
    plot.set_title("$F_\mathrm{twin}$")


    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="linear")
    for i,log10_M1 in enumerate(lms):
        plot.plot(lPs,gammas_large[i],color=colors[i],linestyle='solid')
    plot.set_title("$\gamma_\mathrm{large}$")

    fig=pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,yscale="linear")
    for i,log10_M1 in enumerate(lms):
        plot.plot(lPs,gammas_small[i],color=colors[i],linestyle='solid')
    plot.set_title("$\gamma_\mathrm{small}$")
    """


    print("wall time/s",time.time() - t0)
    pyplot.show()


if __name__ == '__main__':
    test_moe_di_stefano()
