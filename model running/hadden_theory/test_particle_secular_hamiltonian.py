import numpy as np
import celmech as cm
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict
from celmech.poisson_series import PSTerm, PoissonSeries, bracket
from celmech.disturbing_function import list_resonance_terms

def list_multinomial_exponents(pwr,ndim):
    r"""
    List exponents appearing in the expansion of the multinomial 
    
    .. math:: 
    (x_1 + x_2 + ... + x_\mathrm{ndim})^pwr

    Parameters
    ----------
    pwr : int
        Exponent of multinomial
    ndim : int
        Number of variables in multinomial

    Returns
    -------
    list
        A list of lists of length `ndim` representing the exponents of each
        varaible :math:`x_i`
    """
    if ndim==1:
        return [[pwr],]
    else:
        x =[]
        for pow1 in range(0,pwr+1):
            subpows = list_multinomial_exponents(pwr-pow1,ndim-1)
            x+=[[pow1]+y for y in subpows]
        return x
    
from math import factorial
def multinomial_coefficient(p, ks):
    """Calculate multinomial coefficient for given p and ks"""
    num = factorial(p)
    denom = 1
    for k in ks:
        denom *= factorial(k)
    return num // denom

def calc_g0_and_s0(semi_major_axis,synthetic_secular_theory,GM=1.0):
    """
    Evaluate the lowest-order prediction for the secular free apsidal and nodal precession frequencies for a 
    test particle.

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis of test particle
    ss_theory : SyntheticSecularTheory
        Representation of planetary system in which test particle resides.
    GM : float, optional
        Newton's constant times stellar mass, by default 1.0

    Returns
    -------
    _type_
        _description_
    """
    n = np.sqrt(GM / semi_major_axis**3)
    z6 = [0 for _ in range(6)]
    C_e_in = df_coefficient_C(*z6,0,0,1,0)
    C_e_out = df_coefficient_C(*z6,0,0,0,1)
    C_I_in = df_coefficient_C(*z6,1,0,0,0)
    C_I_out = df_coefficient_C(*z6,0,1,0,0)
    g0,s0 = 0,0
    for m_i,a_i in zip(synthetic_secular_theory.masses,synthetic_secular_theory.semi_major_axes):
        if a_i>semi_major_axis:
            alpha = semi_major_axis/a_i
            C_e = evaluate_df_coefficient_dict(C_e_in,alpha)
            C_I = evaluate_df_coefficient_dict(C_I_in,alpha)
            g0 += m_i * alpha * C_e
            s0 += m_i * alpha * C_I
        else:
            alpha = a_i / semi_major_axis
            C_e = evaluate_df_coefficient_dict(C_e_out,alpha)
            C_I = evaluate_df_coefficient_dict(C_I_out,alpha)
            g0 += m_i *  C_e
            s0 += m_i *  C_I
    g0 *= 2*n
    s0 *= 0.5*n
    return g0,s0
    
class SyntheticSecularTheory():
    def __init__(self,masses,semi_major_axes,omega_vector,x_dicts,y_dicts):
        self.masses = masses
        self.semi_major_axes = semi_major_axes
        self.N_planets = len(self.masses)
        self.N_freq = len(omega_vector)
        self.x_dicts = x_dicts
        self.y_dicts = y_dicts
        self.M_i = np.array([len(x_d) for x_d in x_dicts],dtype=int)
        self.N_i = np.array([len(y_d) for y_d in y_dicts],dtype=int)
        self.m_ls = [list(x_d.keys()) for x_d in x_dicts]
        self.n_ls = [list(y_d.keys()) for y_d in y_dicts]
        self.omega_vector = omega_vector
        
    @property
    def Nfreq(self):
        return len(self.omega_vector)
    
    def Xi_to_pow_poisson_series(self,i,pow):
        if pow==0:
            return PSTerm(1,[0,0],[0,0],[0 for _ in range(self.N_freq)],[0 for _ in range(self.N_freq)]).as_series()
        M_i = self.M_i[i]
        x_dict = self.x_dicts[i]
        m_ls = self.m_ls[i]
        
        pvec = np.zeros(self.Nfreq,dtype = int)
        terms = []
        for ks in list_multinomial_exponents(pow,M_i):
            qvec = np.zeros(len(self.omega_vector),dtype = int)
            coeff = multinomial_coefficient(pow,ks)
            for m_l,k in zip(m_ls,ks):
                if k>0:
                    coeff *= (x_dict[m_l])**k
                    qvec += k * np.array(m_l)
            terms.append(PSTerm(coeff,[0,0],[0,0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)

    def Yi_to_pow_poisson_series(self,i,pow):
        if pow==0:
            return PSTerm(1,[0,0],[0,0],[0 for _ in range(self.N_freq)],[0 for _ in range(self.N_freq)]).as_series()
        N_i = self.N_i[i]
        y_dict = self.y_dicts[i]
        n_ls = self.n_ls[i]
        
        pvec = np.zeros(self.Nfreq,dtype = int)
        terms = []
        for ks in list_multinomial_exponents(pow,N_i):
            qvec = np.zeros(len(self.omega_vector),dtype = int)
            coeff = multinomial_coefficient(pow,ks)
            for m_l,k in zip(n_ls,ks):
                if k>0:
                    coeff *= (y_dict[m_l])**k
                    qvec += k * np.array(m_l)
            terms.append(PSTerm(coeff,[0,0],[0,0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)

    def Xbari_to_pow_poisson_series(self,i,pow):
        return self.Xi_to_pow_poisson_series(i,pow).conj
    
    def Ybari_to_pow_poisson_series(self,i,pow):
        return self.Yi_to_pow_poisson_series(i,pow).conj
        
from collections import defaultdict

def mathcal_X_dictionary(semi_major_axis,synthetic_secular_theory,GM=1):
    n = np.sqrt(GM / semi_major_axis**3)
    z4 = [0 for _ in range(4)]
    C_e_mixed = df_coefficient_C(*[0,0,1,-1,0,0],*z4)
    Xcal_dict = defaultdict(float)
    for i in range(synthetic_secular_theory.N_planets):
        a_i = synthetic_secular_theory.semi_major_axes[i]
        m_i = synthetic_secular_theory.masses[i]
        if a_i>semi_major_axis:
            C  = (semi_major_axis/a_i) * evaluate_df_coefficient_dict(C_e_mixed,semi_major_axis/a_i)
        else:
            C  = evaluate_df_coefficient_dict(C_e_mixed,a_i/semi_major_axis)
        x_dict = synthetic_secular_theory.x_dicts[i]
        for mvec,amplitude in x_dict.items():
            Xcal_dict[mvec] += -m_i*n*C*amplitude
    return Xcal_dict

def mathcal_Y_dictionary(semi_major_axis,synthetic_secular_theory,GM=1):
    n = np.sqrt(GM / semi_major_axis**3)
    z4 = [0 for _ in range(4)]
    C_I_mixed = df_coefficient_C(*[0,0,0,0,1,-1],*z4)
    Ycal_dict = defaultdict(complex)
    for i in range(synthetic_secular_theory.N_planets):
        a_i = synthetic_secular_theory.semi_major_axes[i]
        m_i = synthetic_secular_theory.masses[i]
        if a_i>semi_major_axis:
            C  = 0.5 * (semi_major_axis/a_i) * evaluate_df_coefficient_dict(C_I_mixed,semi_major_axis/a_i)
        else:
            C  = 0.5 * evaluate_df_coefficient_dict(C_I_mixed,a_i/semi_major_axis)
        y_dict = synthetic_secular_theory.y_dicts[i]
        for mvec,amplitude in y_dict.items():
            Ycal_dict[mvec] += -m_i*n*C*amplitude
    return Ycal_dict
from collections import defaultdict
from warnings import warn
from celmech.poisson_series import PoissonSeriesHamiltonian
class TestParticleSecularHamiltonian():
    def __init__(self,semi_major_axis,synthetic_secular_theory,GM=1.0):
        """
        Class representing the Hamiltonian of a test particle subject to secular forcing from a system of planets.

        Parameters
        ----------
        semi_major_axis : float
            The test particle's semi-major axis
        synthetic_secular_theory : SyntheticSecularTheory
            A representation of the planetary system's secular dynamics.
        GM : float, optional
            Newton's constant times the stellar mass, by default 1.0
        """
        self.semi_major_axis = semi_major_axis
        self.n = np.sqrt(GM / semi_major_axis**3)
        self.synthetic_secular_theory = synthetic_secular_theory
        self.g0,self.s0 = calc_g0_and_s0(semi_major_axis,synthetic_secular_theory,GM=GM)
        mathcal_X_dict = mathcal_X_dictionary(semi_major_axis,synthetic_secular_theory)
        mathcal_Y_dict = mathcal_Y_dictionary(semi_major_axis,synthetic_secular_theory)

        # Get forced eccentricity
        self.F_e = defaultdict(complex)
        for m,amp in mathcal_X_dict.items():
            omega_m = np.dot(m,self.omega_vector)
            denom = self.g0 - omega_m
            self.F_e[m] += amp / denom

        # Get forced inclination
        self.F_inc = defaultdict(complex)
        for m,amp in mathcal_Y_dict.items():
            omega_m = np.dot(m,self.omega_vector)
            denom = self.s0 - omega_m
            self.F_inc[m] += amp / denom

        self._mmr_corrections  = []
    def _update_linear_theory_with_second_order_in_mass_terms(self,perturber_resonances_dict):
        second_order_terms = PoissonSeries(2,self.N_freq)
        zero_N = [0 for _ in range(self.N_freq)]
        for planet_id,resonance_jk_list in perturber_resonances_dict.items():
            for j,k in resonance_jk_list:
                if (planet_id,j,k) in self._mmr_corrections:
                    warn("" \
                    "A correction for planet {}'s {}:{} MMR has already been inculded in the linear secular theory. " \
                    "This correction will be skipped".format(planet_id,j,k)
                    )
                    continue
                second_order_terms+=self.MMR_second_order_mass_terms(planet_id,j,k,max_order=2)
        dg_key = second_order_terms._PSTerm_to_key(PSTerm(1,[1,0],[1,0],zero_N, zero_N))
        ds_key = second_order_terms._PSTerm_to_key(PSTerm(1,[0,1],[0,1],zero_N, zero_N))

        dg = -1*second_order_terms._terms_dict.pop(dg_key,0)
        ds = -1*second_order_terms._terms_dict.pop(ds_key,0)
        dF_e = dict()
        dF_inc = dict()
        for term in second_order_terms.terms:
            if np.all(term.kbar==np.array([1,0])):
                m = term.q
                omega_m = np.dot(m,self.omega_vector)
                dF_e[tuple(m)] = term.C / (self.g0 + dg - omega_m)
            if np.all(term.kbar==np.array([0,1])):
                m = term.q
                omega_m = np.dot(m,self.omega_vector)
                dF_inc[tuple(m)] = term.C / (self.s0 + ds - omega_m)
        self.g0 += dg
        self.s0 += ds
        for m,amp in dF_e.items():
            self.F_e[m]+=amp
        for m,amp in dF_inc.items():
            self.F_inc[m]+=amp

        #return dg,ds,dF_e,dF_inc


        
    def H2_poisson_series(self):
        N_freq = self.synthetic_secular_theory.N_freq
        zeroN = np.zeros(N_freq,dtype=int)
        eyeN = np.eye(N_freq,dtype=int) 
        h2_series_terms = []# PoissonSeries(2,N_freq)
        h2_series_terms.append(PSTerm(-1*self.g0,[1,0],[1,0],zeroN,zeroN))
        h2_series_terms.append(PSTerm(-1*self.s0,[0,1],[0,1],zeroN,zeroN))
        for omega_i,o_i in zip(self.omega_vector,eyeN):
            h2_series_terms.append(PSTerm(omega_i,[0,0],[0,0],o_i,zeroN))
        return PoissonSeries.from_PSTerms(h2_series_terms)

    def Xi_to_pow_poisson_series(self,i,p):
        return self.synthetic_secular_theory.Xi_to_pow_poisson_series(i,p)
    def Xbari_to_pow_poisson_series(self,i,p):
        return self.synthetic_secular_theory.Xbari_to_pow_poisson_series(i,p)
    def Yi_to_pow_poisson_series(self,i,p):
        return self.synthetic_secular_theory.Yi_to_pow_poisson_series(i,p)
    def Ybari_to_pow_poisson_series(self,i,p):
        return self.synthetic_secular_theory.Ybari_to_pow_poisson_series(i,p)
    @property
    def N_freq(self):
        return self.synthetic_secular_theory.N_freq
    
    @property
    def omega_vector(self):
        return np.array(self.synthetic_secular_theory.omega_vector)
    
    @property
    def M(self):
        return len(self.F_e)
    
    @property
    def N(self):
        return len(self.F_inc)
    
    def _pows_to_series_inner_perturber(self,i,k3,k4,k5,k6,nu1,nu2,nu3,nu4):
        _p  = lambda k,nu: max(0,k) + nu
        _p1 = lambda k,nu: max(0,-k) + nu 
        xpow_series = self.x_to_pow_poisson_series(_p(k4,nu4))
        xbar_pow_series = self.xbar_to_pow_poisson_series(_p1(k4,nu4))
        Xi_pow_series = self.Xi_to_pow_poisson_series(i,_p(k3,nu3))
        Xbari_pow_series = self.Xbari_to_pow_poisson_series(i,_p1(k3,nu3))
        ypow_series = self.y_to_pow_poisson_series(_p(k6,nu2))
        ybar_pow_series = self.ybar_to_pow_poisson_series(_p1(k6,nu2))
        Yi_pow_series = self.Yi_to_pow_poisson_series(i,_p(k5,nu1))
        Ybari_pow_series = self.Ybari_to_pow_poisson_series(i,_p1(k5,nu1))
        series = xpow_series * xbar_pow_series * Xi_pow_series * Xbari_pow_series * ypow_series * ybar_pow_series * Yi_pow_series * Ybari_pow_series
        return series
        
    def _pows_to_series_outer_perturber(self,i,k3,k4,k5,k6,nu1,nu2,nu3,nu4):
        _p  = lambda k,nu: max(0,k) + nu
        _p1 = lambda k,nu: max(0,-k) + nu 
        xpow_series = self.x_to_pow_poisson_series(_p(k3,nu3))
        xbar_pow_series = self.xbar_to_pow_poisson_series(_p1(k3,nu3))
        Xi_pow_series = self.Xi_to_pow_poisson_series(i, _p(k4,nu4))
        Xbari_pow_series = self.Xbari_to_pow_poisson_series(i, _p1(k4,nu4))
        ypow_series = self.y_to_pow_poisson_series(_p(k5,nu1))
        ybar_pow_series = self.ybar_to_pow_poisson_series(_p1(k5,nu1))
        Yi_pow_series = self.Yi_to_pow_poisson_series(i,_p(k6,nu2))
        Ybari_pow_series = self.Ybari_to_pow_poisson_series(i,_p1(k6,nu2))
        series = xpow_series * xbar_pow_series * Xi_pow_series * Xbari_pow_series * ypow_series * ybar_pow_series * Yi_pow_series * Ybari_pow_series
        return series
    
    def _second_order_mass_terms_inner_perturber(self,i,j,k,max_order = None):
        resonance_terms = list_resonance_terms(j,k,max_order = max_order)
        m_i = self.synthetic_secular_theory.masses[i]
        a_i = self.synthetic_secular_theory.semi_major_axes[i]
        alpha = a_i / self.semi_major_axis
        Pk_dict = defaultdict(lambda: PoissonSeries(2,self.N_freq))
        dPk_ddelta_dict = defaultdict(lambda: PoissonSeries(2,self.N_freq))
        nvec = self.n * np.array((1,alpha**(-1.5)))
        for k,nu in resonance_terms:
            k_tp,k_p,k3,k4,k5,k6 = k
            nu1,nu2,nu3,nu4 = nu
            C = evaluate_df_coefficient_dict(df_coefficient_C(*k,*nu),alpha)
            dC_ddelta = evaluate_df_coefficient_dict(df_coefficient_C(*k,*nu,l2 = 1),alpha)
            kvec = (k_tp,k_p)
            pseries = self._pows_to_series_inner_perturber(i,k3,k4,k5,k6,nu1,nu2,nu3,nu4)
            Pk_dict[kvec] += (0.5)**(2*nu2+abs(k6)) * C * pseries
            dPk_ddelta_dict[kvec] += (0.5)**(2*nu2+abs(k6)) * dC_ddelta * pseries
        #return Pk_dict,dPk_ddelta_dict
        terms = PoissonSeries(2,self.N_freq)
        angle_only_term  = lambda u: (np.all(u.k==0) and np.all(u.kbar==0))
        for kv in Pk_dict.keys():
            print(kv)
            k,k_p = kv
            Pk = Pk_dict[kv]
            dPk_ddelta =  dPk_ddelta_dict[kv]
            omega = np.dot(kv,nvec)
            abs_Pk_sq = Pk*Pk.conj
            d_absPk_sq_ddelta = (Pk*dPk_ddelta.conj + Pk.conj*dPk_ddelta)
            term = (1j/omega)*bracket(Pk,Pk.conj) + (-0.5*k/omega) * d_absPk_sq_ddelta + (-1.5*self.n*k*k/omega/omega)*abs_Pk_sq
            # clip terms that only depend on angles
            term = PoissonSeries.from_PSTerms([x for x in term.terms if not angle_only_term(x)])
            terms+=0.5*self.n*self.n*m_i*m_i*term
        return terms + terms.conj
    
    def _second_order_mass_terms_outer_perturber(self,i,j,k,max_order = None):
        resonance_terms = list_resonance_terms(j,k,max_order = max_order)
        m_i = self.synthetic_secular_theory.masses[i]
        a_i = self.synthetic_secular_theory.semi_major_axes[i]
        alpha = self.semi_major_axis / a_i 
        Pk_dict = defaultdict(lambda: PoissonSeries(2,self.N_freq))
        dPk_ddelta_dict = defaultdict(lambda: PoissonSeries(2,self.N_freq))
        nvec = self.n * np.array((1,alpha**(+1.5)))
        for k,nu in resonance_terms:
            k_p,k_tp,k3,k4,k5,k6 = k
            nu1,nu2,nu3,nu4 = nu
            C = evaluate_df_coefficient_dict(df_coefficient_C(*k,*nu),alpha)
            dC_ddelta = evaluate_df_coefficient_dict(df_coefficient_C(*k,*nu,l1 = 1),alpha)
            kvec = (k_tp,k_p)
            pseries = self._pows_to_series_outer_perturber(i,k3,k4,k5,k6,nu1,nu2,nu3,nu4)
            Pk_dict[kvec] += (0.5)**(2*nu1+abs(k5)) * C * pseries
            dPk_ddelta_dict[kvec] += (0.5)**(2*nu1+abs(k5)) * dC_ddelta * pseries
        terms = PoissonSeries(2,self.N_freq)
        angle_only_term  = lambda u: (np.all(u.k==0) and np.all(u.kbar==0))
        for kv in Pk_dict.keys():
            k,k_p = kv
            Pk = Pk_dict[kv]
            dPk_ddelta =  dPk_ddelta_dict[kv]
            omega = np.dot(kv,nvec)
            abs_Pk_sq = Pk*Pk.conj
            d_absPk_sq_ddelta = (Pk*dPk_ddelta.conj + Pk.conj*dPk_ddelta)
            term = (1j/omega)*bracket(Pk,Pk.conj) + (-0.5*k/omega) * d_absPk_sq_ddelta + (-1.5*self.n*k*k/omega/omega)*abs_Pk_sq
            # clip terms that only depend on angles
            term = PoissonSeries.from_PSTerms([x for x in term.terms if not angle_only_term(x)])
            terms+=0.5*self.n*self.n*m_i*m_i*alpha*alpha*term
        return terms + terms.conj
                
    def _DFTerm_poisson_series_inner_perturber(self,alpha,i,k,nu):
        _p  = lambda k,nu: max(0,k) + nu
        _p1 = lambda k,nu: max(0,-k) + nu 
        m_i = self.synthetic_secular_theory.masses[i]
        C = evaluate_df_coefficient_dict(df_coefficient_C(*k,*nu),alpha)
        _,_,k3,k4,k5,k6 = k
        nu1,nu2,nu3,nu4 = nu
        factor = -(0.5)**(2*nu2+abs(k6)) * m_i * self.n * C
        series = self._pows_to_series_inner_perturber(i,k3,k4,k5,k6,nu1,nu2,nu3,nu4)
        series *= factor
        return series + series.conj

    def _DFTerm_poisson_series_outer_perturber(self,alpha,i,k,nu):
        _p  = lambda k,nu: max(0,k) + nu
        _p1 = lambda k,nu: max(0,-k) + nu 
        m_i = self.synthetic_secular_theory.masses[i]
        C = evaluate_df_coefficient_dict(df_coefficient_C(*k,*nu),alpha)
        _,_,k3,k4,k5,k6 = k
        nu1,nu2,nu3,nu4 = nu
        factor = -(0.5)**(2*nu1+abs(k5)) * alpha * m_i * self.n * C
        series = self._pows_to_series_outer_perturber(i,k3,k4,k5,k6,nu1,nu2,nu3,nu4)
        series*= factor
        return series + series.conj

    def DFTerm_poisson_series(self,i,k,nu):
        a_i = self.synthetic_secular_theory.semi_major_axes[i]
        if a_i<self.semi_major_axis:
            alpha = a_i / self.semi_major_axis
            series = self._DFTerm_poisson_series_inner_perturber(alpha,i,k,nu)
        else:
            alpha = self.semi_major_axis/a_i
            series = self._DFTerm_poisson_series_outer_perturber(alpha,i,k,nu)
        return series
    
    def MMR_second_order_mass_terms(self,i,j,k,max_order = None):
        a_i = self.synthetic_secular_theory.semi_major_axes[i]
        if a_i<self.semi_major_axis:
            series = self._second_order_mass_terms_inner_perturber(i,j,k,max_order)
        else:
            series = self._second_order_mass_terms_outer_perturber(i,j,k,max_order)
        return series

    def x_to_pow_poisson_series(self,p):
        exponents = list_multinomial_exponents(p,self.M+1)
        mvecs = list(self.F_e.keys())
        terms = []
        for ks in exponents:
            coeff = multinomial_coefficient(p,ks)
            k0 = ks[0]
            qvec = np.zeros(self.omega_vector.size,dtype = int)
            pvec = np.zeros(self.omega_vector.size,dtype = int)
            for k,m_l in zip(ks[1:],mvecs):
                if k>0:
                    coeff *= (self.F_e[m_l])**k
                    qvec += k*np.array(m_l)
            terms.append(PSTerm(coeff,[k0,0],[0,0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)
    
    def y_to_pow_poisson_series(self,p):
        exponents = list_multinomial_exponents(p,self.N+1)
        nvecs = list(self.F_inc.keys())
        terms = []
        for ks in exponents:
            coeff = multinomial_coefficient(p,ks)
            k0 = ks[0]
            qvec = np.zeros(self.omega_vector.size,dtype = int)
            pvec = np.zeros(self.omega_vector.size,dtype = int)
            for k,n_l in zip(ks[1:],nvecs):
                if k>0:
                    coeff *= (self.F_inc[n_l])**k
                    qvec += k*np.array(n_l)
            terms.append(PSTerm(coeff,[0,k0],[0,0],pvec,qvec))
        return PoissonSeries.from_PSTerms(terms)
    
    def xbar_to_pow_poisson_series(self,p):
        return self.x_to_pow_poisson_series(p).conj
    
    def ybar_to_pow_poisson_series(self,p):
        return self.y_to_pow_poisson_series(p).conj

    def linear_theory_solution(self,x0,y0,times):
        xForced = np.sum([amp * np.exp(1j * np.dot(m,self.omega_vector) * times) for m, amp in self.F_e.items()],axis=0)            
        yForced = np.sum([amp * np.exp(1j * np.dot(m,self.omega_vector) * times) for m, amp in self.F_inc.items()],axis=0)
        x0_free = x0 - xForced[0]
        y0_free = y0 - yForced[0]
        x_soln = x0_free * np.exp(1j * self.g0 * times) + xForced
        y_soln = y0_free * np.exp(1j * self.s0 * times) + yForced
        return x_soln,y_soln
