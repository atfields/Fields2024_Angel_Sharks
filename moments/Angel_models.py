#!/bin/python3
import numpy
import moments

def null_model(params, ns):
        #Moments model
        #4 parameters
        nu1, nu2, nu3, T = params
        sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1] + ns[2])
        fs = moments.Spectrum(sts)

        fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1] + ns[2])
        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])
        nomig = numpy.array([[0,0,0], [0,0,0], [0,0,0]])
        fs.integrate([nu1, nu2, nu3], T, m=nomig, dt_fac=0.01)

        return fs

def pop_model_sym(params, ns):
        #Moments model
        #13 parameters
        Pop1_0, Pop1, An_0, An, Pop2_0, Pop2, Pop3_0, Pop3, m_Pop1_An, m_Pop1_Pop2, m_Pop2_Pop3, T1, T2 = params
        #Pop1_0, Pop1, nAn_0, nAn, Pop2_0, Pop2, Pop3_0, Pop3, m_Pop1_Pop2, m_Pop2_Pop1, m_Pop2_Pop3, m_Pop3_Pop2, T1, T2 = params
        #Define Ancestrial FS
        sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1] + ns[2])
        fs = moments.Spectrum(sts)
        T=T1+T2

        #Split 1
        fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1]+ns[2])

        Pop1_func = lambda t : Pop1_0 * (Pop1/Pop1_0) ** (t/(T))
        An_func = lambda t : An_0 * (An/An_0) ** (t/(T1))
        nu_func = lambda t: [Pop1_func(t), An_func(t)]

        sym_mig = numpy.array([[0,m_Pop1_An], [m_Pop1_An,0]])
        fs.integrate(nu_func, T1, m=sym_mig, dt_fac=0.05)

        #Split 2
        Pop1_0 = Pop1_func(T2)

        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])

        Pop2_func = lambda t : Pop2_0 * (Pop2/Pop2_0) ** (t/T2)
        Pop3_func = lambda t : Pop3_0 * (Pop3/Pop3_0) ** (t/T2)
        nu_func2 = lambda t: [Pop1_func(t), Pop2_func(t), Pop3_func(t)]
        sym_mig2 = numpy.array([[0,m_Pop1_Pop2,0], [m_Pop1_Pop2,0,m_Pop2_Pop3], [0,m_Pop2_Pop3,0]])
        fs.integrate(nu_func2, T2, m=sym_mig2, dt_fac=0.05)
        return fs

def pop_model_asym(params, ns):
        #Moments model
        #16 parameters
        Pop1_0, Pop1, An_0, An, Pop2_0, Pop2, Pop3_0, Pop3, m_Pop1_An, m_An_Pop1, m_Pop1_Pop2, m_Pop2_Pop1, m_Pop2_Pop3, m_Pop3_Pop2, T1, T2 = params
        #Pop1_0, Pop1, nAn_0, nAn, Pop2_0, Pop2, Pop3_0, Pop3, m_Pop1_Pop2, m_Pop2_Pop1, m_Pop2_Pop3, m_Pop3_Pop2, T1, T2 = params
        #Define Ancestrial FS
        sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1] + ns[2])
        fs = moments.Spectrum(sts)
        T=T1+T2

        #Split 1
        fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1]+ns[2])

        Pop1_func = lambda t : Pop1_0 * (Pop1/Pop1_0) ** (t/(T))
        An_func = lambda t : An_0 * (An/An_0) ** (t/(T1))
        nu_func = lambda t: [Pop1_func(t), An_func(t)]

        asym_mig = numpy.array([[0,m_Pop1_An], [m_An_Pop1,0]])
        fs.integrate(nu_func, T1, m=asym_mig, dt_fac=0.05)

        #Split 2
        Pop1_0 = Pop1_func(T2)

        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])

        Pop2_func = lambda t : Pop2_0 * (Pop2/Pop2_0) ** (t/T2)
        Pop3_func = lambda t : Pop3_0 * (Pop3/Pop3_0) ** (t/T2)
        nu_func2 = lambda t: [Pop1_func(t), Pop2_func(t), Pop3_func(t)]
        asym_mig2 = numpy.array([[0,m_Pop1_Pop2,0], [m_Pop2_Pop1,0,m_Pop2_Pop3], [0,m_Pop3_Pop2,0]])
        fs.integrate(nu_func2, T2, m=asym_mig2, dt_fac=0.05)
        return fs

def pop_model_second_sym(params, ns):
        #Moments model
        #13 parameters
        Pop1_0, Pop1, An_0, An, Pop2_0, Pop2, Pop3_0, Pop3, m_Pop1_An, m_Pop1_Pop2, m_Pop2_Pop3, T1, T2, T3 = params
        #Pop1_0, Pop1, nAn_0, nAn, Pop2_0, Pop2, Pop3_0, Pop3, m_Pop1_Pop2, m_Pop2_Pop1, m_Pop2_Pop3, m_Pop3_Pop2, T1, T2 = params
        #Define Ancestrial FS
        sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1] + ns[2])
        fs = moments.Spectrum(sts)
        T=T1+T2+T3

        #Split 1
        fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1]+ns[2])

        Pop1_func = lambda t : Pop1_0 * (Pop1/Pop1_0) ** (t/(T))
        An_func = lambda t : An_0 * (An/An_0) ** (t/(T1))
        nu_func = lambda t: [Pop1_func(t), An_func(t)]

        sym_mig = numpy.array([[0,m_Pop1_An], [m_Pop1_An,0]])
        fs.integrate(nu_func, T1, m=sym_mig, dt_fac=0.05)

        #Split 2
        Pop1_0 = Pop1_func(T2)

        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])

        Pop2_func = lambda t : Pop2_0 * (Pop2/Pop2_0) ** (t/(T2+T3))
        Pop3_func = lambda t : Pop3_0 * (Pop3/Pop3_0) ** (t/(T2+T3))
        nu_func2 = lambda t: [Pop1_func(t), Pop2_func(t), Pop3_func(t)]
        no_mig = numpy.array([[0,0,0], [0,0,0]])
        fs.integrate(nu_func2, T2, m=no_mig, dt_fac=0.05)
        Pop1_0 = Pop1_func(T3)
        Pop2_0 = Pop2_func(T3)
        Pop3_0 = Pop3_func(T3)
        sym_mig2 = numpy.array([[0,m_Pop1_Pop2,0], [m_Pop1_Pop2,0,m_Pop2_Pop3], [0,m_Pop2_Pop3,0]])
        fs.integrate(nu_func2, T3, m=sym_mig2, dt_fac=0.05)
        return fs

