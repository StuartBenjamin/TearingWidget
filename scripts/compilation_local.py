#exec(open("/home/sbenjamin/TearingMacroFolder/local_scripts/compilation.py").read())
import sys
import os
import numpy as np
import xarray as xr
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import cfspopcon
from cfspopcon.unit_handling import ureg
import copy

mu0=1.256637061e-6

#TO DO:
#Use Regular Neo and put in order of w_marg (both toroidal and cylindrical w_d cases)
#Break up terms that lie within regular neo 
#Check if Dnc ~ -Hbs/2 (see equ 57, Shi 2024)... if so recalculate island widths with that term also... (and break up terms and order in terms of w_marg)
#Compile em all...


#NEW Checklist:::
#Priority 1: List largest island width numbers
    #HOW? Make separate variable
#Priority 2: Subidivide Dnc into shear, pressure..., 
#Priority 3: Investigate Hbs again... what is the conversion value?

#Linear stability:
    #Is Dr negative...
        #Yes: 
            #Is Delta' negative:
                #Yes => Stable
                #No:
                    #Is Delta' < Delta' crit
                        #Yes => Stable
                        #No => Unstable
        #No => Unstable



#NEW Testing:
#neo_terms_on_psifacs1 <-> Dncs_Shi see what Rscale needs to be....


def get_dcon_xr(eq_filename,nmin=1,nmax=2,individ_ns=True,dboug=False,**kwargs):
    equil_dat_vec=[]
    dcon_ran_vec=[]
    rdcon_ran_vec=[]
    first_run=True
    for i in range(nmin,nmax):
        dcon_ran=False
        rdcon_ran=False
        equil_dat_xr=None
        print("n=",i)
        dcon_ran,rdcon_ran,equil_dat_xr=run_DCON_on_equilibrium2(eq_filename,nmin=i,nmax=i,run_merscan=first_run,**kwargs)
        if dboug:
            return dcon_ran,rdcon_ran,equil_dat_xr
        print("n=",i,"dcon_ran=",dcon_ran,"rdcon_ran=",rdcon_ran)
        equil_dat_xr=equil_dat_xr.assign(dW_total_temp=equil_dat_xr['n']*0.0+equil_dat_xr['dW_total'])
        equil_dat_xr=equil_dat_xr.assign(dW_plasma_temp=equil_dat_xr['n']*0.0+equil_dat_xr['dW_plasma'])
        equil_dat_xr=equil_dat_xr.assign(dW_vacuum_temp=equil_dat_xr['n']*0.0+equil_dat_xr['dW_vacuum'])
        dW_total_new=equil_dat_xr.dW_total_temp.mean(dim='ns_dcon_ran')
        dW_plasma_new=equil_dat_xr.dW_plasma_temp.mean(dim='ns_dcon_ran')
        dW_vacuum_new=equil_dat_xr.dW_vacuum_temp.mean(dim='ns_dcon_ran')
        equil_dat_xr=equil_dat_xr.assign(dW_total=dW_total_new)
        equil_dat_xr=equil_dat_xr.assign(dW_plasma=dW_plasma_new)
        equil_dat_xr=equil_dat_xr.assign(dW_vacuum=dW_vacuum_new)
        if individ_ns:
            equil_dat_xr=equil_dat_xr.drop_dims(['ns_dcon_ran'])
        dcon_ran_vec.append(dcon_ran)
        rdcon_ran_vec.append(rdcon_ran)
        equil_dat_vec.append(equil_dat_xr)
        first_run=False
    equil_dat=xr.concat(equil_dat_vec,dim='n')
    return equil_dat,dcon_ran_vec,rdcon_ran_vec,equil_dat_vec

#Unfinished
def neo_terms_on_psifacs2_local(dconxr,dcon_mcind,tok_xr,valid_check=False):
    #Local tokamak xr (that specific device)
    local_tok_xr=tok_xr

    #Local tokamak xr dim_rho interpolation values, for transferring splines to psifacs
    interp_dim_rho_vals=dconxr.psifacs[0].values*(len(local_tok_xr.dim_0.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)
    
    #Converting key terms to psifac splines:
    Te_Kev_on_psifacs=local_tok_xr.electron_temp_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").values#.pint.to(ureg.keV).values
    Ti_Kev_on_psifacs=local_tok_xr.ion_temp_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").values#.pint.to(ureg.keV).values
    #cfspopcon default saves them as KeV
    #Sanity check incase they're in eV:
    if Te_Kev_on_psifacs[0]>1000:
        Te_Kev_on_psifacs=Te_Kev_on_psifacs/1000
    if Ti_Kev_on_psifacs[0]>1000:
        Ti_Kev_on_psifacs=Ti_Kev_on_psifacs/1000

    #   Wc terms:
    chi_perp=local_tok_xr.elongation_psi95[0].values*local_tok_xr.minor_radius.values**2/(6*local_tok_xr.energy_confinement_time.values) #Units m^2/s, comes from Fitz 1995
    dconxr=dconxr.assign(chi_perp_no_ne=chi_perp)
    e=1.60217663e-19
    me=9.1091e-31
    temp_e=(Te_Kev_on_psifacs*1e3)*e #eV*e=joules
    temp_i=(Ti_Kev_on_psifacs*1e3)*e #eV*e=joules
    v_te=np.sqrt(2*temp_e/me) #sqrt(joules/kg)=v, see Fitzpatrick equ 1.71
    mi=local_tok_xr.average_ion_mass[0].values*1.66054e-27 #default units are in amu
    v_ti=np.sqrt(2*temp_i/mi) #sqrt(joules/kg)=v, see Fitzpatrick equ 1.71
    replica_data_array=dconxr.qs[0].copy()
    replica_data_array2=dconxr.qs[0].copy()
    replica_data_array3=dconxr.qs[0].copy()
    replica_data_array4=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=v_te[i]
        replica_data_array2[i]=v_ti[i]
        replica_data_array3[i]=Te_Kev_on_psifacs[i]
        replica_data_array4[i]=Ti_Kev_on_psifacs[i]
    dconxr=dconxr.assign(v_te_s=0.0*dconxr['qs']+replica_data_array)
    dconxr=dconxr.assign(v_ti_s=0.0*dconxr['qs']+replica_data_array2)
    dconxr=dconxr.assign(Te_Kev_s=0.0*dconxr['qs']+replica_data_array3)
    dconxr=dconxr.assign(Ti_Kev_s=0.0*dconxr['qs']+replica_data_array4)
    one_on_v_te=1.0/v_te
    #lambda_parallel=n*2*(psi)*(q1/qs)*sqrt(w_psinorm)*(sqrt(kappa)*<a>)/R0
    psis=dconxr.psifacs[0].values
    qspln=CubicSpline(dconxr.psifacs[0].values,dconxr.qs[0].values)
    q1_on_q=[qspln(psis[i],1)/dconxr.qs[0].values[i] for i in range(len(psis))]
    #This from Rosenburg 2002 AND Gates 1997:
        #   Now, r = sqrt(psi)*<a> = sqrt(psi)*sqrt(kappa)*a
        #        psi=r^2/(a^2kappa)
        #        r^2=psi(a^2kappa)
        #   lamda_parallel = R_0 q Lq/(mw)
            #                  = R0 (m/n) L_q/(mw)
            #                  = R0 L_q/(nw)
            #                  = (R0/(nw))(q/q') 
            #                  = (R0/(nw))q/(dq/dr) 
            #                  = (R0/(nw))q/(dq/dpsi * dpsi/dr) 
            #                  = (R0/(nw))q/(q1 * 2r/(a^2kappa)) 
            #                  = (R0/(nw))q/(q1 * 2sqrt(psi)*sqrt(kappa)*a/(a^2kappa)) 
            #                  = (R0/(nw))q/(q1 * 2 sqrt(psi)*sqrt(kappa)/(a*kappa)) 
            #                  = (R0/(nw))q/(q1 * 2 sqrt(psi)/(a*sqrt(kappa))) 
            #                  = a*sqrt(kappa)*(q/(2*q1*sqrt(psi)) * (R0/(n w)) 
            #                  = (a*sqrt(kappa)/2) * (q/(q1*sqrt(psi)) * (R0/(n w)) 
            #                  = (a*sqrt(kappa)/2) * (q/(q1*sqrt(psi)) * (R0/rs)(rs/w)/n  [final fac no dims]
            #                  = (a*sqrt(kappa)/2) * (q/(q1*sqrt(psi)) * (R0/sqrt(psi)*sqrt(kappa)*a)(2psi/w_psi_bar)/n  [using rosenburg 33][final fac no dims]
            #                  = (1/2) * (q/(q1*sqrt(psi)) * (R0/sqrt(psi))(2psi/w_psi_bar)/n  
            #                  = (q/(q1))*(R0)*(1/w_psi_bar)/n   #Needs units of m, has them...
            #                  = (q/(q1))*(R0)/(w_psi_bar n)   #Needs units of m, has them...
        #   1/lamda_parallel = n(q1/q) w_psi_bar/R0    [1/m]
    one_on_lambda_parallel_prefac = q1_on_q/(local_tok_xr.major_radius[0].values) #[1/m]
        #multiply by w_psi_bar, n to get 1/lamda_parallel
        # 1/(v_te*lamda_parallel) [s/m^2]
    one_on_chi_parallel_prefac = one_on_lambda_parallel_prefac*one_on_v_te #[s/m^2]
        #multiply by w_psi_bar, n to get 1/chiparralel
    chi_parallel_prefacs = 1/one_on_chi_parallel_prefac #[m^2/s]
        #divide by w_psi_bar, n to get chiparralel
    chifrac_prefac = chi_perp[0]*one_on_chi_parallel_prefac #[unitless]
        #multiply by w_psi_bar, n to get chifrac 

    #chifrac_prefac  
    replica_data_array=dconxr.Wc_prefacs[0].copy()
    assert dconxr.Wc_prefacs[0].ipsis.values[0]==0
    for i in dconxr.Wc_prefacs[0].ipsis.values:
        replica_data_array[i]=chi_parallel_prefacs[i]
    dconxr=dconxr.assign(chi_parallel_prefacs=0.0*dconxr['Wc_prefacs']+replica_data_array)

    #wc_internal_prefac  
        # wc_internal_prefac = chifrac_prefac*dconxr.Wc_prefacs.values
    replica_data_array=dconxr.Wc_prefacs[0].copy()
    assert dconxr.Wc_prefacs.ipsis.values[0]==0
    for i in dconxr.Wc_prefacs.ipsis.values:
        replica_data_array[i]=replica_data_array[i].values*chifrac_prefac[i]
    dconxr=dconxr.assign(Wc_internal_prefacs=0.0*dconxr['Wc_prefacs']+replica_data_array)
    #multiply by w_psi_bar (w_psi/psio = w_psi_norm), (n/m^2) to get thing inside ()**(1/4)
        #Thus to get Wc as a function of w_psi_norm, write Wc = (w_psi_norm)**(1/4)*((n/m**2)*wc_internal_prefac)**(1/4) [this in Weber because Wc_prefacs is Wb**4]
        #My final Wc will be in units of Wb (not normalised...)

    #Island from Rosenberg et al. 2002
    kappa=local_tok_xr.elongation_psi95[0].values
    a=local_tok_xr.minor_radius[0].values
    R0=local_tok_xr.major_radius[0].values
    rs = np.sqrt(psis)*np.sqrt(kappa)*a
    s_s_bar=psis*q1_on_q
    wd_prefacR=7.2**4*(psis**3)*(chi_perp[0])*(R0/(rs**2))*one_on_v_te*(1/s_s_bar)
    #Multiply by w_psi_norm, divide by n to get w_d_bar**4 (normalised w_d_bar=w[Wb]/psio, aka w_d_bar is unitless here, see Rosenberg 2002)

    #wc_Ros_internal_prefac  
    replica_data_array=dconxr.Wc_prefacs[0].copy()
    assert dconxr.Wc_prefacs.ipsis.values[0]==0
    for i in dconxr.Wc_prefacs.ipsis.values:
        replica_data_array[i]=wd_prefacR[i]
    dconxr=dconxr.assign(WcR_internal_prefacs=0.0*dconxr['Wc_prefacs']+replica_data_array)

    #Print it broh:
    if valid_check:
        w=0.01
        m=2
        n=1
        psio=dconxr.psio.values
        print(f"psio = {dconxr.psio.values}, w/psio = {w}")
        print("psifac,          q,          Wc_prefac,          k_para/k_perp (func of w/psio),              wd normed (func of w/psio),           wd normed (Rosen) (func of w/psio)")
        for i in dconxr.Hbs_prefacs[0].ipsis.values:
            print(dconxr.psifacs[0].sel(ipsis=i).values,",",dconxr.qs[0].sel(ipsis=i).values,",",dconxr.Wc_prefacs[0].sel(ipsis=i).values,",",1/(n*w*dconxr.psio.values*1*chifrac_prefac[i]),(w*(n/(m**2))*dconxr.Wc_internal_prefacs[0].isel(ipsis=i).values)**(1/4)/dconxr.psio.values,(w*(1/n)*wd_prefacR[i])**(1/4))
        print("DEBONGE")
        raise

    #print(dconxr.psio.values)
    #print(dconxr.Wc_prefacs.values)
    #print(1/(0.1*dconxr.psio.values*1*chifrac_prefac)) #ka perp etc seem reasonable... not sure why wd is so large
    #print((0.1*dconxr.psio.values*dconxr.Wc_internal_prefacs.values*(1/4))**(1/4)/dconxr.psio.values)
        
    #Generating X0::: to calculate Delta' crit for cases where Drbs < 0
    """
        #Converting key terms to psifac splines:
        #   temp and density terms: 
        #ni_on_psifacs=local_tok_xr.fuel_ion_density_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").pint.to(ureg.m**-3).values
        #ne_on_psifacs=local_tok_xr.electron_density_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").pint.to(ureg.m**-3).values
        #Ti_on_psifacs=local_tok_xr.ion_temp_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").pint.to(ureg.eV).values
        
        #eta_on_psifacs = (2.8e-8) * (Te_Kev_on_psifacs ** (-1.5)) #Ohm-m `wesson_tokamaks_2011,` assuming the Coulomb logarithm = 17 and Z=1.
        

        #(2.8e-8) * (T_electron_temp ** (-1.5))
        #   taua terms: 
        #me=9.1091e-31
        #mi=local_tok_xr.average_ion_mass.pint.to(ureg.kg).values
        #rho_on_psifacs=ne_on_psifacs*me+ni_on_psifacs*mi 
        #sqrt_rho_on_psifacs=np.sqrt(rho_on_psifacs) #Sqrt of mass density
        #   taur terms:
        #Lambda=
        #taue=3.44e5*te**1.5/(ne*Lambda)
        #eta=np.sqrt(2)/()

        #e=1.60217663e-19
        #3e0=8.85418782e-12
        #prefac=(sqrt(2)/(12*np.pi^(3/2)))*(e/e0)^2
                #6.9299845689669565e-18
        #eta_on_psifacts=6.5e-8*Te_on_psifacs #Use F's eta.... 

        #Below: '###################' = ready if necessary
        ################### dconxr=dconxr.assign(tauas_no_n=dconxr['taua_prefacs']*sqrt_rho_on_psifacs)
        #   ^Divide by n to get tau_a
        ################### dconxr=dconxr.assign(taurs=dconxr['taur_prefacs']/eta_on_psifacs)
        ################### dconxr=dconxr.assign(X0_prefacs = lambda dconxr: (dconxr.taurs/dconxr.tauas_no_n)**(-1/3))
        #   ^Divide by n**(1/3) to get X0

        #rho = (ne*(2.5*1.672614e-27)) kg/m^3
        #lambda=24-.5*LOG(ne)+LOG(te)
        #taue=3.44e5*te**1.5/(ne*lambda)
        #ne*mi*1e6

            taua_prefac=SQRT(Mloc*mu0)/ABS(twopi*q1*chi1/v1) 
                !to get taua, multiply by local sqrt(rho) and divide by 
                !toroidal mode number nn (see below)
            taur_prefac=((avg(1)/avg(5))*mu0)
                !to get taur, divide by local resistivity (see eta below)

            !Recipe for taua, taur:
                !   ne in units m^(-3): generic value 1e14
                !   te in units eV: generic value 3e3
                !   mi, me in units kg, note mp=1.672614e-27,me=9.1091e-31.
                !   lambda=24-.5*LOG(ne)+LOG(te)
                !   taue=3.44e5*te**1.5/(ne*lambda)
                !   eta=me/(ne*1e6*e**2*taue*1.96)
                !   rho=ne*mi*1e6
                !taua=SQRT(rho*Mloc*mu0)/
                !           ABS(twopi*nn*q1*chi1/v1)
                !taur=((avg(1)/avg(5))*mu0)/eta
            !Recipe for Sfac, X0, Q0, Vs:
                !sfac=taur/taua
                !X0=sfac**(-1._r8/3._r8)
                !Q0=x0/taua
                !Vs=1 for the outer region (Vs term from Glasser 75, Shi 24)

        #dconxr=dconxr.assign(N1=dconxr['Drs']+dconxr['Hbs']*(dconxr['Hs']-0.5))

        #for i in range(len(dconxr_vec)):
        #    MC_ind_dcon=dconxr_vec[i].MC_index.values
        #    for DCONJfrac in dconxr_vec[i].plasma_current_fraction.values:
        #        local_tok_xr=tok_xr.sel(MC_index=MC_ind_dcon,plasma_current_fraction=DCONJfrac)
        #
        #        interp_dim_rho_vals=dconxr_vec[i].psifacs.values*(len(local_tok_xr.dim_0.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)
        #        Dnc_mufrac_on_psifacs=local_tok_xr.Dnc_mufrac_on_linspace_psi_dynamo.interp(dim_0=interp_dim_rho_vals, method="cubic")
    """
    return dconxr

#Get poloidal beta, need gyroradii... I suppose
def neo_terms_on_psifacs2b_local(dconxr,dcon_mcind,tok_xr,valid_check=False):
    local_tok_xr=tok_xr

    #Local tokamak xr dim_rho interpolation values, for transferring splines to psifacs
    interp_dim_rho_vals=dconxr.psifacs[0].values*(len(local_tok_xr.dim_0.values)-1)

    #Generating X0::: to calculate Delta' crit for cases where Drbs < 0

    #Converting key terms to psifac splines:
    #   temp and density terms:
     
    ni_on_psifacs=local_tok_xr.fuel_ion_density_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").values      #*10**19 #default units on cluster are 10^19 m^-3
    ne_on_psifacs=local_tok_xr.electron_density_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").values      #*10**19 #default untis on cluster are 10^19 m^-3

    #Sanity check incase they're not in 10^19:
    if ni_on_psifacs[0]>10**21 or ni_on_psifacs[0]<10**19:
        raise Exception("check density units")
    if ne_on_psifacs[0]>10**21 or ne_on_psifacs[0]<10**19:
        raise Exception("check density units")

    #Ti_on_psifacs=local_tok_xr.ion_temp_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").pint.to(ureg.eV).values
    Te_Kev_on_psifacs=local_tok_xr.electron_temp_profile.interp(dim_0=interp_dim_rho_vals, method="cubic").values#.pint.to(ureg.keV).values
    #cfspopcon default saves them as KeV
    #Sanity check incase they're in eV:
    if Te_Kev_on_psifacs[0]>1000:
        Te_Kev_on_psifacs=Te_Kev_on_psifacs/1000


    # taua terms: 
    me=9.1091e-31
    mi=local_tok_xr.average_ion_mass[0].values*1.66054e-27 #default units are in amu
    rho_on_psifacs=ne_on_psifacs*me+ni_on_psifacs*mi 
    sqrt_rho_on_psifacs=np.sqrt(rho_on_psifacs) #Sqrt of mass density
    ##  !to get taua, multiply taua_prefac by local sqrt(rho) and divide by toroidal mode number nn 
    tau_a_times_nn_on_psifacs=dconxr.taua_prefacs[0].values*sqrt_rho_on_psifacs

    #   taur terms:
    eta_on_psifacs = (2.8e-8) * (Te_Kev_on_psifacs ** (-1.5)) #Ohm-m `wesson_tokamaks_2011,` assuming the Coulomb logarithm = 17 and Z=1.
    ##  !to get taur, divide taur_prefac by local resistivity 
    tau_r_on_psifacs=dconxr.taur_prefacs[0].values/eta_on_psifacs

    #!Recipe for Sfac, X0, Q0, Vs:
    sfac_on_nn_on_psifacs=tau_r_on_psifacs/tau_a_times_nn_on_psifacs #THIS ON SURFACE FOR CALCULATION ON MODE
    X0_no_nn_on_psifacs=sfac_on_nn_on_psifacs**(-1.0/3.0) #THIS ON SURFACE FOR CALCULATION ON MODE
    #To get X0, multiply X0_no_nn_on_psifacs by (nn)**(-1/3) where nn is toroidal mode number
    #For n <=4, X0 only differs by a factor of 4**(-1/3) ~ 0.63.. ie same order of magnitude

    Hs=dconxr.Hs.values
    Drs=dconxr.Drs.values

    #Qcrit=abs(math.gamma(3/4)*math.gamma(1/2-Hs/4)**2*math.gamma(1/4-Hs/2)*math.sin((1-2*Hs)*math.pi/8)*Drs/
    #    (math.gamma(1/4)*math.gamma(1-Hs/4)**2*math.gamma(3/4-Hs/2)*math.sin((5+2*Hs)*math.pi/8)*4))**(2.0/3.0) #THIS ON SURFACE
    #DeltaPrimeCrit_no_nn=math.pi*(2/X0_no_nn_on_psifacs)**(1-2*Hs)*math.gamma(1/4)*math.gamma(1-Hs/4)**2*math.gamma(3/4-Hs/2)*Qcrit**((2*Hs+5)/4)/
    #    ((np.sqrt(2)*(1-2*Hs)*math.sin((1-2*Hs)*math.pi/8))*(1-Hs/2)*(math.cos(Hs*math.pi/2)*math.gamma((1+Hs)/4)*math.gamma(1-Hs))**2) #THIS ON SURFACE

    Qcrit_cylinder = 0.473*(abs(Drs)**(2.0/3.0)) #THIS ON SURFACE
    DeltaPrimeCrit_no_nn_cylinder = 1.54*(dconxr.psifacs.values/X0_no_nn_on_psifacs)*(abs(Drs)**(5.0/6.0)) #THIS ON SURFACE
 
    #!Vs=1 for the outer region (Vs term from Glasser 75, Shi 24)   

    #Putting sfac_on_nn_on_psifacs, tau_r_on_psifacs on surface FOR CALCULATION ON MODE
    replica_data_array=dconxr.qs[0].copy()
    replica_data_array2=dconxr.qs[0].copy()
    replica_data_array3=dconxr.qs[0].copy()
    replica_data_array4=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=sfac_on_nn_on_psifacs[i]
        replica_data_array2[i]=tau_r_on_psifacs[i]
        replica_data_array3[i]=ni_on_psifacs[i]
        replica_data_array4[i]=ne_on_psifacs[i]
    dconxr=dconxr.assign(sfacs_on_nn=0.0*dconxr['qs']+replica_data_array)
    dconxr=dconxr.assign(tau_rs=0.0*dconxr['qs']+replica_data_array)
    dconxr=dconxr.assign(ni_on_psifacs=0.0*dconxr['qs']+replica_data_array3)
    dconxr=dconxr.assign(ne_on_psifacs=0.0*dconxr['qs']+replica_data_array4)

    #Putting X0_no_nn_on_psifacs on surface FOR CALCULATION ON MODE
    replica_data_array=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=X0_no_nn_on_psifacs[i]
    dconxr=dconxr.assign(X0s_no_nn=0.0*dconxr['qs']+replica_data_array)

    #Putting Qcrit_cylinder on surface
    replica_data_array=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=0.473*abs(Drs[0][i])**(2.0/3.0)
    dconxr=dconxr.assign(Qcrits_cylinder=0.0*dconxr['qs']+replica_data_array)

    #Putting DeltaPrimeCrit_no_nn_cylinder on surface
    replica_data_array=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=1.54*(dconxr.psifacs[0].values[i]/X0_no_nn_on_psifacs[i])*(abs(Drs[0][i])**(5.0/6.0))
    dconxr=dconxr.assign(DeltaPrimeCrits_no_nn_cylinder=0.0*dconxr['qs']+replica_data_array)

    #Putting Qcrit, DeltaPrimeCrit_no_nn on surface 
    replica_data_array=dconxr.qs[0].copy()
    replica_data_array2=dconxr.qs[0].copy()
    assert dconxr.qs[0].ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values: ###FIX THIS
        if Hs[0][i]<0.5 and Hs[0][i]>-5/2:
            Qcrit=abs(math.gamma(3/4)*math.gamma(1/2-Hs[0][i]/4)**2*math.gamma(1/4-Hs[0][i]/2)*math.sin((1-2*Hs[0][i])*math.pi/8)*Drs[0][i]/
            (math.gamma(1/4)*math.gamma(1-Hs[0][i]/4)**2*math.gamma(3/4-Hs[0][i]/2)*math.sin((5+2*Hs[0][i])*math.pi/8)*4))**(2.0/3.0)
            replica_data_array[i]=Qcrit
            replica_data_array2[i]=math.pi*(2*dconxr.psifacs[0].values[i]/X0_no_nn_on_psifacs[i])**(1-2*Hs[0][i])*math.gamma(1/4)*math.gamma(1-Hs[0][i]/4)**2*math.gamma(3/4-Hs[0][i]/2)*Qcrit**((2*Hs[0][i]+5)/4)/((np.sqrt(2)*(1-2*Hs[0][i])*math.sin((1-2*Hs[0][i])*math.pi/8))*(1-Hs[0][i]/2)*(math.cos(Hs[0][i]*math.pi/2)*math.gamma((1+Hs[0][i])/4)*math.gamma(1-Hs[0][i]))**2)
        else:
            replica_data_array[i]=np.nan
            replica_data_array2[i]=np.nan
    dconxr=dconxr.assign(Qcrits=0.0*dconxr['qs']+replica_data_array)
    dconxr=dconxr.assign(DeltaPrimeCrits_no_nn=0.0*dconxr['qs']+replica_data_array2)

    return dconxr

#Untested
#def neo_terms_on_modes(dconxr):
#    Dnc_spln=CubicSpline(dconxr.psifacs.values,dconxr.Dncs.values)
#    Hbss_spln=CubicSpline(dconxr.psifacs.values,dconxr.Hbss.values)

#    dconxr=dconxr.assign(Dnc = lambda dconxr: Dnc_spln(dconxr.psifac.values))
#    dconxr=dconxr.assign(Hbs = lambda dconxr: Hbss_spln(dconxr.psifac.values))

def neo_terms_on_psifacs1_local(dconxr,dcon_mcind,tok_xr):
    #Local tokamak xr (that specific device)
    local_tok_xr=tok_xr

    #Local tokamak xr dim_rho interpolation values, for transferring splines to psifacs
    interp_dim_rho_vals=dconxr.psifacs[0].values*(len(tok_xr.Dnc_mufrac_on_linspace_psi_dynamo.dim_0.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)
    #interp_dim_rho_vals2=dconxr.psifacs.values*(len(tok_xr.psi.dim_0.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)


    #Converting key terms to psifac splines:
    #   Dnc term: 
    Dnc_mufrac_on_psifacs=local_tok_xr.Dnc_mufrac_on_linspace_psi_dynamo.interp(dim_0=interp_dim_rho_vals, method="cubic")
    #   Hbs term:
    #avg_BJbs_on_psifacs=local_tok_xr.avg_BJbs_on_linspace_psi_dynamo.interp(dim_rho=interp_dim_rho_vals, method="cubic")

    #mufracs
    replica_data_array=dconxr.Dnc_prefacs[0].copy()
    assert dconxr.Dnc_prefacs.ipsis.values[0]==0
    for i in dconxr.Dnc_prefacs.ipsis.values:
        replica_data_array[i]=Dnc_mufrac_on_psifacs[i].values
    dconxr=dconxr.assign(mufracs=0.0*dconxr['Dnc_prefacs']+replica_data_array)

    #Dnc 
    replica_data_array=dconxr.Dnc_prefacs[0].copy()
    assert dconxr.Dnc_prefacs.ipsis.values[0]==0
    for i in dconxr.Dnc_prefacs.ipsis.values:
        replica_data_array[i]=replica_data_array[i].values*Dnc_mufrac_on_psifacs[i].values
    dconxr=dconxr.assign(Dncs=0.0*dconxr['Dnc_prefacs']+replica_data_array)

    #Hbss
    #replica_data_array=dconxr.Hbs_prefacs.copy()
    #assert dconxr.Hbs_prefacs.ipsis.values[0]==0
    #for i in dconxr.Hbs_prefacs.ipsis.values:
    #    replica_data_array[i]=replica_data_array[i].values*avg_BJbs_on_psifacs[i].values*mu0
    #dconxr=dconxr.assign(Hbss=0.0*dconxr['Hbs_prefacs']+replica_data_array)
    
    #Hegna Dnc + combined stability term:
    dconxr=dconxr.assign(Dhs=dconxr['Drs']/(-0.5+np.sqrt(-dconxr['Dis'])-dconxr['Hs']))
    dconxr=dconxr.assign(Dh_alts=dconxr['Drs']/(0.5+np.sqrt(-dconxr['Dis'])-dconxr['Hs']))
    dconxr=dconxr.assign(DrbsHs=dconxr['Dncs']+dconxr['Dhs'])
    dconxr=dconxr.assign(DrbsH_alts=dconxr['Dncs']+dconxr['Dh_alts'])

    #Shi combined stability term + rescaling to Hegna:
    #dconxr=dconxr.assign(Drbss=dconxr['Drs']+dconxr['Hbss']*(dconxr['Hs']-0.5))
    #rscale=dconxr.psio.values**2
    #dconxr=dconxr.assign(DrbsRSs=dconxr['Drbss']*rscale)
    
    #Shi Dnc equivalence:
    #rscale2=1.0
    #dconxr=dconxr.assign(Dncs_Shi=-(0.5*rscale2)*dconxr['Hbss'])
    
    #dconxr=dconxr.assign(DrbsRS2s=dconxr['Drbss']*(0.5+np.sqrt(-dconxr['Dis'])-dconxr['Hs'])*(0.5+np.sqrt(-dconxr['Dis'])-dconxr['Hs']))

    #Difference between the two:
    #sumvals=0.5*(np.abs(dconxr.DrbsRSs.values)+np.abs(dconxr.DrbsH_alts.values))
    #oneonsum=1/sumvals
    #dconxr=dconxr.assign(Drbs_rel_Diffs= lambda dconxr: (dconxr.DrbsH_alts-dconxr.DrbsRSs)*oneonsum)

    #^what if thats from incorrect normalisation of <Dnc?>


    #print(dconxr.psio.values)
    #print(dconxr.psio.values**2)
    #print(dconxr.volume.values)

    #for i in dconxr.Hbs_prefacs.ipsis.values:
    #    print(dconxr.psifacs.sel(ipsis=i).values,",",dconxr.qs.sel(ipsis=i).values,",",dconxr.Drbs_rel_Diffs.sel(ipsis=i).values)
    #raise

    #for i in dconxr.Hbs_prefacs.ipsis.values:
    #    print(dconxr.psifacs.sel(ipsis=i).values,",",dconxr.qs.sel(ipsis=i).values,",",dconxr.DrbsRS2s.sel(ipsis=i).values,",",dconxr.DrbsRSs.sel(ipsis=i).values,",",dconxr.DrbsHs.sel(ipsis=i).values,",",dconxr.DrbsH_alts.sel(ipsis=i).values)
    #raise

    #for i in dconxr.Hbs_prefacs.ipsis.values:
    #    print(dconxr.psifacs.sel(ipsis=i).values,",",dconxr.qs.sel(ipsis=i).values,",",-0.5*dconxr.psio.values**2*dconxr.Hbss.sel(ipsis=i).values,",",dconxr.Dncs.sel(ipsis=i).values)
    #raise

    #raise
    #for i in dconxr.Hbs_prefacs.ipsis.values:
    #    print(dconxr.psifacs.sel(ipsis=i).values,",",dconxr.qs.sel(ipsis=i).values,",",dconxr.psio.values**2*dconxr.Drbss.sel(ipsis=i).values,",",dconxr.DrbsHs.sel(ipsis=i).values,",",dconxr.DrbsH_alts.sel(ipsis=i).values)
    #raise
    return dconxr

def neo_terms_on_psifacs1b_local(dconxr,dcon_mcind,tok_xr,verbose=False):
    #Local tokamak xr (that specific device)
    local_tok_xr=tok_xr

    #Local tokamak xr dim_rho interpolation values, for transferring splines to psifacs
    interp_dim_rho_vals=dconxr.psifacs[0].values*(len(local_tok_xr.dim_0.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)

    #mag shear on psifacs:
    psis=dconxr.psifacs[0].values
    qspln=CubicSpline(dconxr.psifacs[0].values,dconxr.qs[0].values)
    #   dconxr = dconxr.assign(q1_on_qs=0.0*dconxr['qs']+q1_on_qs)
    replica_data_array=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=qspln(psis[i],1)/dconxr.qs[0][i].values
        if verbose:
            print(f"psifac={psis[i]}, q1_on_q={qspln(psis[i],1)/dconxr.qs[0][i].values}")
    dconxr=dconxr.assign(q1_on_qs=0.0*dconxr['qs']+replica_data_array)

    #Converting key terms to psifac splines:
    #   <B> term:
    local_tok_xr=local_tok_xr.assign(avg_B_on_linspace_psi_dynamo=local_tok_xr['modb_avgs[0]'])
    #print(dconxr.qs.ipsis.values[-1])
    #print(dconxr.qs.ipsis.values[-2])
    #print(local_tok_xr.avg_B_on_linspace_psi_dynamo)
    #print(local_tok_xr.avg_B_on_linspace_psi_dynamo.values[-1])
    #print(local_tok_xr.avg_B_on_linspace_psi_dynamo.values[-2])
    #print(local_tok_xr.avg_B_on_linspace_psi_dynamo.psi_normalised.values[-2])
    #print(local_tok_xr.avg_B_on_linspace_psi_dynamo.dim_0.values)
    #print(local_tok_xr.avg_B_on_linspace_psi_dynamo.dim_0.values[-1])

   # local_tok_xr.avg_B_on_linspace_psi_dynamo.loc[dict(dim_0=local_tok_xr.avg_B_on_linspace_psi_dynamo.dim_0.values[-1])]=local_tok_xr.avg_B_on_linspace_psi_dynamo.values[-2]
    local_tok_xr=local_tok_xr.assign(avg_Bsq_on_linspace_psi_dynamo=local_tok_xr['avg_B_on_linspace_psi_dynamo']*local_tok_xr['avg_B_on_linspace_psi_dynamo']) 
    avg_Bsq_on_psifacs=local_tok_xr.avg_Bsq_on_linspace_psi_dynamo.interp(dim_0=interp_dim_rho_vals, method="cubic").values

    #if verbose:
        #print("avg_B_on_linspace_psi_dynamo= ",local_tok_xr.avg_B_on_linspace_psi_dynamo.values)
        #print(local_tok_xr.avg_Bsq_on_linspace_psi_dynamo)
        #print("interp_dim_rho_vals = ",interp_dim_rho_vals)
        #print("avg_Bsq_on_psifacs = ",avg_Bsq_on_psifacs)

    #(dp/dr / B^2) on psifacs:
    mu0_p_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.mu0_ps[0].values)
    mu0mu0_p1_on_Bsq=[mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i] for i in range(len(psis))]
    dconxr = dconxr.assign(mu0_p1_on_Bsqs=0.0*dconxr['qs']+mu0mu0_p1_on_Bsq)
    replica_data_array=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    #print(len(psis))
    #print(len(avg_Bsq_on_psifacs))
    #print(len(replica_data_array))
    for i in dconxr.qs.ipsis.values:
        #print(f"psifac={psis[i]}, mu0_p_on_Bsq={mu0_p_spln(psis[i])/avg_Bsq_on_psifacs[i]}, mu0_pprime_on_Bsq={mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i]}")
        replica_data_array[i]=mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i]
        if verbose:
            print(f"psifac={psis[i]}, mu0_p_on_Bsq={mu0_p_spln(psis[i])/avg_Bsq_on_psifacs[i]}, mu0_pprime_on_Bsq={mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i]}")
    dconxr=dconxr.assign(mu0_p1_on_Bsqs=0.0*dconxr['qs']+replica_data_array)

    #(dp/dr / p) on psifacs:
    replica_data_array=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    #print(len(psis))
    #print(len(avg_Bsq_on_psifacs))
    #print(len(replica_data_array))
    for i in dconxr.qs.ipsis.values:
        #print(f"psifac={psis[i]}, mu0_p_on_Bsq={mu0_p_spln(psis[i])/avg_Bsq_on_psifacs[i]}, mu0_pprime_on_Bsq={mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i]}")
        replica_data_array[i]=mu0_p_spln(psis[i],1)/dconxr.mu0_ps[0].values[i]
    dconxr=dconxr.assign(p1_on_ps=0.0*dconxr['qs']+replica_data_array)

    print(dconxr.Avg_minor_rs[0].values)
    p_on_min_radii_spln=CubicSpline(dconxr.Avg_minor_rs[0].values,(1/mu0)*dconxr.mu0_ps[0].values)
    p_on_min_radii2_spln=CubicSpline(np.sqrt(dconxr.psifacs[0].values)*local_tok_xr.minor_radius[0].values,(1/mu0)*dconxr.mu0_ps[0].values)
    q_on_min_radii_spln=CubicSpline(dconxr.Avg_minor_rs[0].values,dconxr.qs[0].values)
    q_on_min_radii2_spln=CubicSpline(np.sqrt(dconxr.psifacs[0].values)*local_tok_xr.minor_radius[0].values,dconxr.qs[0].values)

    dp_drs=[p_on_min_radii_spln(psis[i],1) for i in range(len(psis))]
    dp_drs2=[p_on_min_radii2_spln(psis[i],1) for i in range(len(psis))]
    dq_drs=[q_on_min_radii_spln(psis[i],1) for i in range(len(psis))]
    dq_drs2=[q_on_min_radii2_spln(psis[i],1) for i in range(len(psis))]

    replica_data_array=dconxr.qs[0].copy()
    replica_data_array2=dconxr.qs[0].copy()
    replica_data_array3=dconxr.qs[0].copy()
    replica_data_array4=dconxr.qs[0].copy()
    assert dconxr.qs.ipsis.values[0]==0
    for i in dconxr.qs.ipsis.values:
        replica_data_array[i]=dp_drs[i]
        replica_data_array2[i]=dp_drs2[i]
        replica_data_array3[i]=dq_drs[i]
        replica_data_array4[i]=dq_drs2[i]
    dconxr=dconxr.assign(dp_drs=0.0*dconxr['qs']+replica_data_array)
    dconxr=dconxr.assign(dp_drs2=0.0*dconxr['qs']+replica_data_array2)
    dconxr=dconxr.assign(dq_drs=0.0*dconxr['qs']+replica_data_array3)
    dconxr=dconxr.assign(dq_drs2=0.0*dconxr['qs']+replica_data_array4)

    #Remaining term [Dnc subtracting the shear and p' term...]:::: MAKE IT REPLICA DATA ARRAY
    replica_data_array=dconxr.Dnc_prefacs[0].copy()
    assert dconxr.Dnc_prefacs[0].ipsis.values[0]==0
    for i in dconxr.Dnc_prefacs[0].ipsis.values:
        print(f"psifac={psis[i]}, Dnc_prefacs={dconxr.Dnc_prefacs[0][i].values}, qshear={qspln(psis[i],1)/dconxr.qs[0][i].values}, pshear={mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i]}")
        qshear = qspln(psis[i],1)/dconxr.qs[0][i].values
        pshear = mu0_p_spln(psis[i],1)/avg_Bsq_on_psifacs[i]
        replica_data_array[i]=replica_data_array[i].values*qshear/pshear
        if verbose:
            print(f"psifac={psis[i]}, Dnc_prefacs={dconxr.Dnc_prefacs[0][i].values}, qshear={qshear}, pshear={pshear}, Dnc*qshear/pshear={replica_data_array[i].values}")
    dconxr=dconxr.assign(Dnc_geom_facs=0.0*dconxr['Dnc_prefacs']+replica_data_array)

    return dconxr

#Returns lin_well_hill
def lin_well_hill2(dconxr,local_tok_xr,verbose=False):
    #print("WRF")
    #Is there a well/hill present? Creates a spline of those values I suppose
    Jtor2_on_radius_spln=CubicSpline(local_tok_xr.minor_radius_dim_rho.values,local_tok_xr.Jtor_GSoutput_on_linspace_psi_dynamo.values)

    dense_minor_radii=local_tok_xr.minor_radius_dim_rho.interp(dim_rho=np.linspace(0,1,num=500, endpoint=True)*(len(local_tok_xr.dim_rho.values)-1), method="cubic").values

    high_res_Jtor2_spln=CubicSpline(dense_minor_radii,Jtor2_on_radius_spln(dense_minor_radii))

    high_res_dJtor2_dr_spln=CubicSpline(dense_minor_radii,high_res_Jtor2_spln(dense_minor_radii,1))
    high_res_d2Jtor2_dr2_spln=CubicSpline(dense_minor_radii,high_res_Jtor2_spln(dense_minor_radii,2))

    #dJt2_dr_spline=CubicSpline(local_tok_xr.minor_radius_dim_rho.values, dJt2_dr_on_linspace_psi)
    #dsqJt2_drsq_spline=CubicSpline(local_tok_xr.minor_radius_dim_rho.values, dJt2_dr_spline(local_tok_xr.minor_radius_dim_rho.values,1))

    zero_gradients2=high_res_dJtor2_dr_spln.roots(extrapolate=False)
    inflections_pts=high_res_d2Jtor2_dr2_spln.roots(extrapolate=False)
    
    if verbose:
        print("Max minor radius")
        print(max(dense_minor_radii))
        print("Zeros and inflections::::::::::::::::::")
        print(zero_gradients2)
        print(inflections_pts)
        print("Cutoff=",local_tok_xr.minor_radius_dim_rho.interp(dim_rho=[0.97*(len(local_tok_xr.dim_rho.values)-1)], method="cubic").values)
        print("Zeros and inflections::::::::::::::::::")

    zero_gradients=[]
    for i in range(len(zero_gradients2)-1):
        assert zero_gradients2[i+1]>zero_gradients2[i]
    for i in range(len(zero_gradients2)):
        #print(zero_gradients2[i])
        #print(local_tok_xr.minor_radius_dim_rho.interp(dim_rho=[0.97*(len(local_tok_xr.dim_rho.values)-1)], method="cubic").values)
        if zero_gradients2[i]<local_tok_xr.minor_radius_dim_rho.interp(dim_rho=[0.97*(len(local_tok_xr.dim_rho.values)-1)], method="cubic").values:
            zero_gradients.append(zero_gradients2[i])
            #EXCLUDE SUPER EDGE CASES COS THEYRE CURSED

    #print(zero_gradients)

    assert len(inflections_pts)>=(len(zero_gradients2)-1)

    #print("WAAT")

    zero_gradients_cp=np.full(10,np.nan)
    w_raw_vals=np.zeros([len(zero_gradients),2])
    h_vals=np.zeros([len(zero_gradients),2])
    w_vals=np.zeros(len(zero_gradients))
    is_well=np.full(10,True)
    steepness=np.full(10,np.nan)
    wh_raw_ranges=np.zeros([len(zero_gradients),2])
    for i in range(min(len(zero_gradients),10)):
        zero_gradients_cp[i]=zero_gradients[i]
        if high_res_d2Jtor2_dr2_spln(zero_gradients[i])>0: #Well
            if i+1==len(zero_gradients):
                w_raw_vals[i,1]=local_tok_xr.minor_radius_dim_rho.max().values-zero_gradients[i]
            else:
                w_raw_vals[i,1]=min(zero_gradients[i+1]-zero_gradients[i],local_tok_xr.minor_radius_dim_rho.max().values-zero_gradients[i])

            if i==0:
                w_raw_vals[i,0]=zero_gradients[i]
            else:
                w_raw_vals[i,0]=min(zero_gradients[i]-zero_gradients[i-1],zero_gradients[i])

            assert w_raw_vals[i,0]>0
            assert w_raw_vals[i,1]>0
            w_vals[i]=min(w_raw_vals[i,1],w_raw_vals[i,0])

            assert local_tok_xr.minor_radius_dim_rho.min().values<0.01

            wh_raw_ranges[i,0]=max(zero_gradients[i]-w_vals[i],local_tok_xr.minor_radius_dim_rho.min().values)
            wh_raw_ranges[i,1]=zero_gradients[i]+w_vals[i]
            assert wh_raw_ranges[i,1]>wh_raw_ranges[i,0]

            h_vals[i,0]=high_res_Jtor2_spln(wh_raw_ranges[i,0])-high_res_Jtor2_spln(zero_gradients[i])
            h_vals[i,1]=high_res_Jtor2_spln(wh_raw_ranges[i,1])-high_res_Jtor2_spln(zero_gradients[i])

            if not (h_vals[i,1]>0 and h_vals[i,0]>0):
                continue
                print([wh_raw_ranges[i,0],zero_gradients[i],wh_raw_ranges[i,1]])
                print([high_res_Jtor2_spln(wh_raw_ranges[i,0]),high_res_Jtor2_spln(zero_gradients[i]),high_res_Jtor2_spln(wh_raw_ranges[i,1])])
                plt.close()
                plt.scatter(local_tok_xr.minor_radius_dim_rho,local_tok_xr.Jtor_GSoutput_on_linspace_psi_dynamo.values)
                plt.scatter([wh_raw_ranges[i,0],zero_gradients[i],wh_raw_ranges[i,1]],[high_res_Jtor2_spln(wh_raw_ranges[i,0]),high_res_Jtor2_spln(zero_gradients[i]),high_res_Jtor2_spln(wh_raw_ranges[i,1])])
                plt.savefig(str('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F0funny_Jtorr'+str(i)+'.png'), dpi=600)
                plt.close()
                plt.scatter(local_tok_xr.minor_radius_dim_rho,high_res_dJtor2_dr_spln(local_tok_xr.minor_radius_dim_rho.values))
                plt.scatter(zero_gradients,high_res_dJtor2_dr_spln(zero_gradients))
                plt.savefig(str('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F0funny_JtorrDeriv'+str(i)+'.png'), dpi=600)
                plt.close()
                #plt.scatter(local_tok_xr.minor_radius_dim_rho,local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.values)
                #plt.scatter(zero_gradients,np.zeros(len(zero_gradients)))
                #plt.scatter([minor_radius_spln(0.905),minor_radius_spln(0.92),minor_radius_spln(0.94),minor_radius_spln(0.96)],[0.1,0.1,0.1,0.1])
                #plt.scatter([0.905,0.95,0.96],[0.1,0.1,0.1])
                #plt.savefig('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F0funny_Jbs.png', dpi=600)
                #plt.close()
                #print("hi1)")
                #print(len(dconxr.psifacs.values),len(dconxr.Jbss.values))
                #plt.scatter(dconxr.psifacs.values,dconxr.Jbss.values)
                #plt.scatter([0.905,0.95,0.96],[0.1,0.1,0.1])
                #plt.savefig('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F0funny_Jbs3a.png', dpi=600)
                #plt.close()
                #print("hi2)")
                #print(len(np.linspace(0, 1, num=(len(local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo)))),len(local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.values))
                #plt.scatter(np.linspace(0, 1, num=(len(local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.values)), endpoint=True),local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.values)
                #plt.scatter([0.905,0.95,0.96],[0.1,0.1,0.1])
                #plt.savefig('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F0funny_Jbs3b.png', dpi=600)
                #plt.close()
                #print("hi3)")
                #plt.scatter(local_tok_xr.minor_radius_dim_rho.values,local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.values)
                #plt.scatter(minor_radius_spln([0.905,0.95,0.96]),[0.1,0.1,0.1])
                #plt.savefig('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F0funny_Jbs3c.png', dpi=600)
                assert (h_vals[i,1]>0 and h_vals[i,0]>0)

            steepness[i]=(min(h_vals[i,0],h_vals[i,1])/w_vals[i])/high_res_Jtor2_spln(zero_gradients[i])
        elif high_res_d2Jtor2_dr2_spln(zero_gradients[i])<0: #Hill
            is_well[i]=False
            if i+1==len(zero_gradients):
                w_raw_vals[i,1]=local_tok_xr.minor_radius_dim_rho.max().values-zero_gradients[i]
            else:
                w_raw_vals[i,1]=min(zero_gradients[i+1]-zero_gradients[i],local_tok_xr.minor_radius_dim_rho.max().values-zero_gradients[i])

            if i-1<0:
                w_raw_vals[i,0]=zero_gradients[i]
            else:
                w_raw_vals[i,0]=min(zero_gradients[i]-zero_gradients[i-1],zero_gradients[i])

            w_vals[i]=min(w_raw_vals[i,:])

            wh_raw_ranges[i,0]=max(zero_gradients[i]-w_vals[i],local_tok_xr.minor_radius_dim_rho.min().values)
            wh_raw_ranges[i,1]=zero_gradients[i]+w_vals[i]
            assert wh_raw_ranges[i,1]>wh_raw_ranges[i,0]

            h_vals[i,0]=-(high_res_Jtor2_spln(wh_raw_ranges[i,0])-high_res_Jtor2_spln(zero_gradients[i]))
            h_vals[i,1]=-(high_res_Jtor2_spln(wh_raw_ranges[i,1])-high_res_Jtor2_spln(zero_gradients[i]))

            if not (h_vals[i,1]>0 and h_vals[i,0]>0):
                continue

            steepness[i]=(min(h_vals[i,0],h_vals[i,1])/w_vals[i])/high_res_Jtor2_spln(zero_gradients[i])
        else:
            print('infection point found during well/hill search')

    wh_ranges=np.full([10,2],np.nan)
    for i in range(min(len(zero_gradients)-1,9)):
        assert wh_raw_ranges[i,1]<=wh_raw_ranges[i+1,1] 
        wh_ranges[i,0]=wh_raw_ranges[i,0]
        if wh_raw_ranges[i+1,0]<wh_raw_ranges[i,1]:
            inflec_inbetween=[j for j in inflections_pts if wh_raw_ranges[i+1,0]<j<wh_raw_ranges[i,1]]
            assert len(inflec_inbetween)>0
            if len(inflec_inbetween)==1:
                wh_ranges[i,1]=inflec_inbetween[0]
                wh_raw_ranges[i+1,0]=inflec_inbetween[0]
            elif len(inflec_inbetween)>1:
                wh_ranges[i,1]=np.mean(inflec_inbetween)
                wh_raw_ranges[i+1,0]=np.mean(inflec_inbetween)
                #if not i==len(zero_gradients)-2:
                #    print('Multiple inflection points found between two wells/hills')
        else:
            wh_ranges[i,1]=wh_raw_ranges[i,1]
    if len(zero_gradients)>0:
        wh_ranges[min(len(zero_gradients)-1,9),0]=wh_raw_ranges[min(len(zero_gradients)-1,9),0]
        wh_ranges[min(len(zero_gradients)-1,9),1]=wh_raw_ranges[min(len(zero_gradients)-1,9),1]
    
    for i in range(min(len(zero_gradients)-1,9)):
        assert wh_ranges[i,0]<wh_ranges[i,1]<=wh_ranges[i+1,0]<wh_ranges[i+1,1]

    if False:
        for i in range(min(len(zero_gradients),9)):
            print([wh_ranges[i,0],zero_gradients[i],wh_ranges[i,1]])
            print([high_res_Jtor2_spln(wh_ranges[i,0]),high_res_Jtor2_spln(zero_gradients[i]),high_res_Jtor2_spln(wh_ranges[i,1])])
            plt.close()
            plt.scatter(local_tok_xr.minor_radius_dim_rho,local_tok_xr.Jtor_GSoutput_on_linspace_psi_dynamo.values)
            plt.scatter([wh_ranges[i,0],zero_gradients[i],wh_ranges[i,1]],[high_res_Jtor2_spln(wh_ranges[i,0]),high_res_Jtor2_spln(zero_gradients[i]),high_res_Jtor2_spln(wh_ranges[i,1])])
            plt.savefig(str('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F1funny_Jtorr'+str(i)+'.png'), dpi=600)
            plt.close()
            plt.scatter(local_tok_xr.minor_radius_dim_rho,high_res_dJtor2_dr_spln(local_tok_xr.minor_radius_dim_rho.values))
            plt.scatter(zero_gradients,high_res_dJtor2_dr_spln(zero_gradients))
            plt.savefig(str('/home/sbenjamin/TearingMacroFolder/results/Newfigs/F1funny_JtorrDeriv'+str(i)+'.png'), dpi=600)
            plt.close()

    #We have the ranges, nearest well/hill, steepness and whether its a well or a hill!
    #wh_ranges
    #zero_gradients
    #steepness
    #is_well

    #First check whether well_hill is True
    #Then check whether minor radius on psifacs for a mode sits within wh_ranges[i,:], then evaluate closeness via zero_gradients, put in a cutoff
    #Also put all the current gradient, local bootstrap and BSfraction onto the modes
    #Then that's it I suppose

    if len(zero_gradients)==0:
        dconxr=dconxr.assign(well_hill=False)
    else:
        dconxr=dconxr.assign(well_hill=True)
        if verbose:
            print("zero_gradients=",zero_gradients)
            print("wh_ranges=",wh_ranges[0:(len(zero_gradients))])
            print("steepness=",steepness[0:(len(zero_gradients))])
            print("is_well=",is_well[0:(len(zero_gradients))])
            print("num zero gradients=",len(zero_gradients))

    #print("check0")
    #raise Exception("ELLO")


    dconxr=dconxr.assign(wh_ranges=(['wh_num','lr'],wh_ranges))
    dconxr=dconxr.assign(zero_gradients=(['wh_num'],zero_gradients_cp))
    #print("HEREREE")
    #print(dconxr.zero_gradients)
    dconxr=dconxr.assign(steepness=(['wh_num'],steepness))
    dconxr=dconxr.assign(is_well=(['wh_num'],is_well))
    dconxr=dconxr.assign(num_whs=len(zero_gradients))

    #Just need minor radius on psifacs and then we have everything we need to diagnose modes

    #Plan: Break up psi < 0.905 into regions of well, hill, and normal in current profile 
    #Assume there will be a peak near edge..., (due to BS), and that we won't be analysing pedestal modes
    # Find local min, and local max (near edge), then the point on the inboard side where Jtor is equal to the local max, this describes inner edge of well
    # In_well = to the right of the well boundary
    # We get our closeness to well parameter by physical distance from the local min, and then also divided by max minor radius
    # We get well steepness by [check my old method]

    #For each well: steepness parameter
    #For each mode: in well yes/no, then closeness to well (absolute and relative), then steepness?
        #link this to DP, same as current gradients
    #ADD A MEAN JBS!! on top of JbsFrac
    #THEN put the local gradients and fractions and Jbs on modes, then we are good to go
    return dconxr

def lin_terms_on_psifacs_local(dconxr,dcon_mcind,tok_xr):
    local_tok_xr=tok_xr
    
    #Local tokamak xr dim_rho interpolation values, for transferring splines to psifacs
    interp_dim_rho_vals=dconxr.psifacs[0].values*(len(local_tok_xr.dim_0.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)

    local_tok_xr=local_tok_xr.assign(Jbs_on_Jt2_on_linspace_psi=local_tok_xr['Jbs_GSinput_on_linspace_psi_dynamo']/local_tok_xr['Jtor_GSoutput_on_linspace_psi_dynamo'])

    JBS_on_psifacs=local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.interp(dim_0=interp_dim_rho_vals, method="cubic").values
    Jbs_on_Jt2_on_psifacs=local_tok_xr.Jbs_on_Jt2_on_linspace_psi.interp(dim_0=interp_dim_rho_vals, method="cubic").values

    #Jbs on psifacs
    replica_data_array=dconxr.Dnc_prefacs[0].copy()
    assert dconxr.Dnc_prefacs.ipsis.values[0]==0
    for i in dconxr.Dnc_prefacs.ipsis.values:
        replica_data_array[i]=JBS_on_psifacs[i]
    dconxr=dconxr.assign(Jbss=0.0*dconxr['Dnc_prefacs']+replica_data_array)
    #Jbs_on_Jt2 on psifacs
    replica_data_array=dconxr.Dnc_prefacs[0].copy()
    assert dconxr.Dnc_prefacs.ipsis.values[0]==0
    for i in dconxr.Dnc_prefacs.ipsis.values:
        replica_data_array[i]=Jbs_on_Jt2_on_psifacs[i]
    dconxr=dconxr.assign(Jbs_on_Jt2s=0.0*dconxr['Dnc_prefacs']+replica_data_array)

    #dconxr=lin_well_hill2(dconxr,local_tok_xr)

    return dconxr

def eval_well_hill(approx_minor_radius,dconxr):
    wellhill_minor_radius=np.nan
    minor_radius_closeness=np.nan
    steepness=np.nan
    iswell=np.nan
    for i in range(len(dconxr.zero_gradients)): #These are minor radius locations
        if dconxr.wh_ranges[i,0].values<approx_minor_radius<dconxr.wh_ranges[i,1].values:
            #we are in/on a well/hill
            wellhill_minor_radius=dconxr.zero_gradients[i].values
            minor_radius_closeness=np.abs(approx_minor_radius-wellhill_minor_radius)
            steepness=dconxr.steepness[i].values
            if dconxr.is_well[i].values:
                iswell=1.0
            else:
                iswell=0.0
            break
    return wellhill_minor_radius,minor_radius_closeness,steepness,iswell

#Neo terms on modes... sure plot this 
def neo_terms_on_modes_local(dconxr,tok_xr,verbose=False,k1=1.7):
    dconxr=dconxr.assign(k1=k1)
    #Already got Dh, Dh_alt
    #Just need Dnc, Hbs, Drbs

    #
    #Drbss_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Drbss[0].values)

    DrbsH_alts_spln=CubicSpline(dconxr.psifacs[0].where(dconxr.DrbsH_alts[0].notnull()).dropna(dim="ipsis").values,dconxr.DrbsH_alts[0].where(dconxr.DrbsH_alts[0].notnull()).dropna(dim="ipsis").values)

    #W marginal ingredients
    Dnc_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Dncs[0].values)
    mufrac_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.mufracs[0].values)
    Wc_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Wc_internal_prefacs[0].values)
    Wc_geom_term_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Wc_prefacs[0].values)
    WcR_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.WcR_internal_prefacs[0].values)
    chi_parallel_prefac_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.chi_parallel_prefacs[0].values)
    #Dnc sub components
    qshear_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.q1_on_qs[0].values)
    pshear_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.mu0_p1_on_Bsqs[0].values)
    Dnc_geom_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Dnc_geom_facs[0].values)
    #Linear layer widths & info
    sfacs_on_nn_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.sfacs_on_nn[0].values)
    tau_rs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.tau_rs[0].values)
    X0s_no_nn_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.X0s_no_nn[0].values)
    #Qcrits_spln=CubicSpline(dconxr.psifacs.where().values,dconxr.Qcrits.values)
    #Linear tearing terms
    #approx_minor_radiis_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.approx_minor_radiis[0].values)
    #dJt_drs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.dJt_drs[0].values)
    #dJt2_drs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.dJt2_drs[0].values)
    Jbss_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Jbss[0].values)
    Jbs_on_Jt2s_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Jbs_on_Jt2s[0].values)
    
    Btot_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_Btots[0].values)
    Btor_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_Btors[0].values)
    Bpol_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_Bpols[0].values)
    r_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_minor_rs[0].values)
    R_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_Rs[0].values)
    one_on_R_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_1_on_Rs[0].values)
    JdotBpols_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.JdotBpols[0].values)
    JdotBtors_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.JdotBtors[0].values)
    Jparas_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Jparas[0].values)
    Jpols_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Jpols[0].values)
    Jtors_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Jtors[0].values)
    Jboot_dot_Bs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Jboot_dot_Bs[0].values/mu0)
    Avg_Btot_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Avg_Btots[0].values)
    Dnc_simps_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Dnc_simps[0].values)
    mu0_p_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.mu0_ps[0].values)

    p1_on_p_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.p1_on_ps[0].values)
    Te_Kev_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Te_Kev_s[0].values)
    Ti_Kev_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Ti_Kev_s[0].values)
    v_te_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.v_te_s[0].values)
    v_ti_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.v_ti_s[0].values)
    ni_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.ni_on_psifacs[0].values)
    ne_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.ne_on_psifacs[0].values)
    dp_drs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.dp_drs[0].values)
    dp_drs2_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.dp_drs2[0].values)
    dq_drs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.dq_drs[0].values)
    dq_drs2_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.dq_drs2[0].values)
    qspln=CubicSpline(dconxr.psifacs[0].values,dconxr.qs[0].values)

    #I NEED total B, Bt (get from F), B_poloidal (!!), minor radius, electron and ion temps (all for ion larmor radius), pressure shear legnthscale
    #I can also get the basic trapped fraction
    #   Take these all straight from DCON

    #Dnc 
    if len(dconxr.ns_rdcon_ran.values)==0 or max(dconxr.ns_rdcon_ran.values)<1:
        return dconxr

    replica_Drbss_data_array=dconxr.Re_DeltaPrime.copy()
    replica_DrbsH_alts_data_array=dconxr.Re_DeltaPrime.copy()
    #W marginal ingredients
    replica_Dnc_data_array=dconxr.Re_DeltaPrime.copy()
    replica_mufrac_data_array=dconxr.Re_DeltaPrime.copy()
    replica_Wc_bar_data_array=dconxr.Re_DeltaPrime.copy()
    replica_WcR_bar_data_array=dconxr.Re_DeltaPrime.copy()
    #Dnc sub components
    replica_qshear_data_array=dconxr.Re_DeltaPrime.copy()
    replica_pshear_data_array=dconxr.Re_DeltaPrime.copy()
    replica_Dnc_geom_data_array=dconxr.Re_DeltaPrime.copy()
    #W marginal widths
    replica_marg_wbar_data_array=dconxr.Re_DeltaPrime.copy()
    replica_marg_wbarR_data_array=dconxr.Re_DeltaPrime.copy()
    #W marginal width orderings
    replica_marg_wbar_order_array=dconxr.Re_DeltaPrime.copy()#dconxr.Re_DeltaPrime.astype('int32').copy()
    replica_marg_wbarR_order_array=dconxr.Re_DeltaPrime.copy()#dconxr.Re_DeltaPrime.astype('int32').copy()
    #Pressure delta' scalings
    replica_data_array_DeltaPrime_pscaling=dconxr.Re_DeltaPrime.copy()
    replica_data_array_DeltaPrime_pscalingR=dconxr.Re_DeltaPrime.copy()
    #W fitzpatricks
    replica_data_arrayWc_at_marg_wbar=dconxr.Re_DeltaPrime.copy()
    replica_data_arrayWc_geom_fac=dconxr.Re_DeltaPrime.copy()
    replica_data_arraychi_parallel_no_w=dconxr.Re_DeltaPrime.copy()
    replica_data_arrayWcR_at_wbarR=dconxr.Re_DeltaPrime.copy()
    replica_data_arrayWc_at_wbarR=dconxr.Re_DeltaPrime.copy()
    #Linear layer widths & info
    replica_data_array_sfac=dconxr.Re_DeltaPrime.copy()
    replica_data_array_tau_r=dconxr.Re_DeltaPrime.copy()
    replica_data_array_tau_a=dconxr.Re_DeltaPrime.copy()
    replica_data_array_X0=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Qcrit=dconxr.Re_DeltaPrime.copy()
    replica_data_array_DeltaPrimeCrit=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Qcrit_cylinder=dconxr.Re_DeltaPrime.copy()
    replica_data_array_DeltaPrimeCrit_cylinder=dconxr.Re_DeltaPrime.copy()
    replica_data_array_DeltaPrimeCombined=dconxr.Re_DeltaPrime.copy()
    replica_data_array_DeltaPrimeCombined_cylinder=dconxr.Re_DeltaPrime.copy()
    replica_data_array_linOnWmarg=dconxr.Re_DeltaPrime.copy()
    replica_data_array_linOnWmargR=dconxr.Re_DeltaPrime.copy()
    replica_data_array_linOnWc=dconxr.Re_DeltaPrime.copy()
    replica_data_array_linOnWcR=dconxr.Re_DeltaPrime.copy()
    #Linear tearing local values
    replica_data_array_approx_minor_radius=dconxr.Re_DeltaPrime.copy()
    replica_data_array_dJt_drs=dconxr.Re_DeltaPrime.copy()
    replica_data_array_dJt2_drs=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Jbss=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Jbs_on_Jt2s=dconxr.Re_DeltaPrime.copy()
    #Linear tearing well/hill values
    replica_data_array_wellhill_approx_minor_radius_location=dconxr.Re_DeltaPrime.copy()
    replica_data_array_wellhill_approx_minor_radius_closeness=dconxr.Re_DeltaPrime.copy()
    replica_data_array_wellhill_steepness=dconxr.Re_DeltaPrime.copy()
    replica_data_array_iswell=dconxr.Re_DeltaPrime.copy()
    #LaHaye terms
    replica_data_array_Btot=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Btor=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Bpol=dconxr.Re_DeltaPrime.copy()
    replica_data_array_r=dconxr.Re_DeltaPrime.copy()
    replica_data_array_R=dconxr.Re_DeltaPrime.copy()
    replica_data_array_one_on_R=dconxr.Re_DeltaPrime.copy()
    replica_data_array_JdotBpols=dconxr.Re_DeltaPrime.copy()
    replica_data_array_JdotBtors=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Jparas=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Jpols=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Jtors=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Jboot_dot_Bs=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Avg_Btot=dconxr.Re_DeltaPrime.copy()
    replica_data_array_Dnc_simps=dconxr.Re_DeltaPrime.copy()
    replica_data_array_mu0_p=dconxr.Re_DeltaPrime.copy()
    replica_data_array_p1_on_p=dconxr.Re_DeltaPrime.copy()
    replica_data_array_T_e_Kev=dconxr.Re_DeltaPrime.copy()
    replica_data_array_T_i_Kev=dconxr.Re_DeltaPrime.copy()
    replica_data_array_v_te=dconxr.Re_DeltaPrime.copy()
    replica_data_array_v_ti=dconxr.Re_DeltaPrime.copy()
    replica_data_array_ni=dconxr.Re_DeltaPrime.copy()
    replica_data_array_ne=dconxr.Re_DeltaPrime.copy()
    replica_data_array_dp_drs=dconxr.Re_DeltaPrime.copy()
    replica_data_array_dp_drs2=dconxr.Re_DeltaPrime.copy()
    replica_data_array_dq_drs=dconxr.Re_DeltaPrime.copy()
    replica_data_array_dq_drs2=dconxr.Re_DeltaPrime.copy()
    replica_data_array_q=dconxr.Re_DeltaPrime.copy()
    for n in dconxr.Re_DeltaPrime.n.values:
        for m in dconxr.Re_DeltaPrime.m.values:
            psifac=dconxr.psifac.sel(n=n,m=m).values
            psio=dconxr.psio[0].values

            #Linear tearing values
            #approx_minor_radius=approx_minor_radiis_spln(psifac)
            #if psifac<0.97:
                #dJt_dr=dJt_drs_spln(psifac)
                #dJt2_dr=dJt2_drs_spln(psifac)
            #else:
                #dJt_dr=np.nan
                #dJt2_dr=np.nan
            Jbs=Jbss_spln(psifac)
            Jbs_on_Jt2=Jbs_on_Jt2s_spln(psifac)
            #wellhill_minor_radius,minor_radius_closeness,steepness,iswell=eval_well_hill(approx_minor_radius,dconxr) 

            #Linear layer widths & info section
            H = dconxr.H.sel(n=n,m=m).values
            sfac = sfacs_on_nn_spln(psifac)*n
            tau_r = tau_rs_spln(psifac)
            tau_a = tau_r/sfac
            X0 = X0s_no_nn_spln(psifac)*(n**(-1.0/3.0))

            if H<0.5 and H>-5/2:
                Qcrit = abs(math.gamma(3/4)*math.gamma(1/2-H/4)**2*math.gamma(1/4-H/2)*math.sin((1-2*H)*math.pi/8)*dconxr.Dr.sel(n=n,m=m).values/(math.gamma(1/4)*math.gamma(1-H/4)**2*math.gamma(3/4-H/2)*math.sin((5+2*H)*math.pi/8)*4))**(2.0/3.0)
                DeltaPrimeCrit = (math.pi*(2*psifac/X0)**(1-2*H)*math.gamma(1/4)*math.gamma(1-H/4)**2*math.gamma(3/4-H/2)*Qcrit**((2*H+5)/4)/((np.sqrt(2)*(1-2*H)*math.sin((1-2*H)*math.pi/8))*(1-H/2)*(math.cos(H*math.pi/2)*math.gamma((1+H)/4)*math.gamma(1-H))**2))
            else:
                Qcrit=np.nan
                DeltaPrimeCrit=np.nan

            Qcrit_cylinder = 0.473*abs(dconxr.Dr.sel(n=n,m=m).values)**(2.0/3.0)
            DeltaPrimeCrit_cylinder = 1.54*(psifac/X0)*(abs(dconxr.Dr.sel(n=n,m=m).values)**(5.0/6.0))
            #Linear layer widths end

            DeltaPrimeDimless=dconxr.Re_DeltaPrime.sel(n=n,m=m).values
            Dh_alt=dconxr.Dh_alt.sel(n=n,m=m).values
            Di=dconxr.Di.sel(n=n,m=m).values
            Dnc=Dnc_spln(psifac)
            mufrac=mufrac_spln(psifac)
            qshear=qshear_spln(psifac)
            pshear=pshear_spln(psifac)
            Dnc_geom=Dnc_geom_spln(psifac)
            alpha_l_alt=dconxr.alpha_l_alt.sel(n=n,m=m).values
            alpha_s_alt=dconxr.alpha_s_alt.sel(n=n,m=m).values

            Wc_bar_no_w=(1/psio)*(Wc_spln(psifac)*n/(m**2))**(1/4) #mult by w_psi_norm**(1/4) to get w_d_bar (unitless).
            WcR_bar_no_w=(WcR_spln(psifac)/n)**(1/4) #mult by w_psi_norm**(1/4) to get w_d_bar (unitless).
            chi_parallel_no_w=chi_parallel_prefac_spln(psifac)/n #divide by w_psi_bar to get chiparralel

            marg_wbar=np.nan
            marg_wbarR=np.nan

            marg_wbar_Wc=np.nan
            marg_wbarR_WcR=np.nan
            marg_wbarR_Wc=np.nan
            deltaPrime_pscaling=np.nan
            deltaPrime_pscalingR=np.nan
            if not np.isnan(min([DeltaPrimeDimless,psifac,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt])):
                print(f"n,m={n},{m}: Deltaprime={DeltaPrimeDimless},psifac={psifac}, Dh_alt={Dh_alt}, Di={Di}, Dnc={Dnc}, alpha_l_alt={alpha_l_alt}, alpha_s_alt={alpha_s_alt}, Wc_bar_no_w={Wc_bar_no_w}, WcR_bar_no_w={WcR_bar_no_w}")
            #if psifac<0.905 and (not np.isnan(np.sum(np.array([DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt])))):
            if psifac<0.95 and (not np.isnan(np.sum(np.array([DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt])))):
                if True:
                    if n==1:
                        print(f"n,m={n},{m}: Deltaprime={DeltaPrimeDimless}")
                if not np.isnan(Wc_bar_no_w):
                    marg_wbar,marg_wbar_Wc,deltaPrime_pscaling=solve_marg_width(Wc_bar_no_w,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k1=k1)
                if not np.isnan(WcR_bar_no_w):
                    marg_wbarR,marg_wbarR_WcR,deltaPrime_pscalingR=solve_marg_width(WcR_bar_no_w,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k1=k1)
                    if marg_wbarR<1.0:
                        marg_wbarR_Wc=Wc_bar_no_w*(marg_wbarR**(1/4))  #Confusion... I want Wc at marg_wbar, and WcR at marg_wbarR marg_wbar_Wc

                if False:
                    print(f"m,n={m},{n}: Deltaprime={DeltaPrimeDimless}, wbar1={marg_wbar}, wbar2={marg_wbarR}, Island_width_comparisons: {marg_wbarR_Wc},{marg_wbarR_WcR}")
                
                print(f"          marg_wbarR={marg_wbarR}, Wc_bar_no_w={Wc_bar_no_w}, WcR_bar_no_w={WcR_bar_no_w}")

            #replica_Drbss_data_array.loc[dict(n=n,m=m)]=Drbss_spln(psifac)
            replica_DrbsH_alts_data_array.loc[dict(n=n,m=m)]=DrbsH_alts_spln(psifac)
            replica_Dnc_data_array.loc[dict(n=n,m=m)]=Dnc
            replica_mufrac_data_array.loc[dict(n=n,m=m)]=mufrac
            replica_qshear_data_array.loc[dict(n=n,m=m)]=qshear
            replica_pshear_data_array.loc[dict(n=n,m=m)]=pshear
            replica_Dnc_geom_data_array.loc[dict(n=n,m=m)]=Dnc_geom
            replica_Wc_bar_data_array.loc[dict(n=n,m=m)]=Wc_bar_no_w
            replica_WcR_bar_data_array.loc[dict(n=n,m=m)]=WcR_bar_no_w
            replica_data_arrayWc_geom_fac.loc[dict(n=n,m=m)]=(1/psio)*(Wc_geom_term_spln(psifac))**(1/4)
            replica_data_arraychi_parallel_no_w.loc[dict(n=n,m=m)]=chi_parallel_no_w
            replica_marg_wbar_data_array.loc[dict(n=n,m=m)]=marg_wbar 
            replica_marg_wbarR_data_array.loc[dict(n=n,m=m)]=marg_wbarR
            replica_data_array_DeltaPrime_pscaling.loc[dict(n=n,m=m)]=deltaPrime_pscaling
            replica_data_array_DeltaPrime_pscalingR.loc[dict(n=n,m=m)]=deltaPrime_pscalingR
            replica_data_arrayWc_at_marg_wbar.loc[dict(n=n,m=m)]=marg_wbar_Wc
            replica_data_arrayWcR_at_wbarR.loc[dict(n=n,m=m)]=marg_wbarR_WcR
            replica_data_arrayWc_at_wbarR.loc[dict(n=n,m=m)]=marg_wbarR_Wc
            #Linear layer widths & info section
            replica_data_array_sfac.loc[dict(n=n,m=m)]=sfac
            replica_data_array_tau_r.loc[dict(n=n,m=m)]=tau_r
            replica_data_array_tau_a.loc[dict(n=n,m=m)]=tau_a
            replica_data_array_X0.loc[dict(n=n,m=m)]=X0
            replica_data_array_Qcrit.loc[dict(n=n,m=m)]=Qcrit
            replica_data_array_DeltaPrimeCrit.loc[dict(n=n,m=m)]=DeltaPrimeCrit
            replica_data_array_Qcrit_cylinder.loc[dict(n=n,m=m)]=Qcrit_cylinder
            replica_data_array_DeltaPrimeCrit_cylinder.loc[dict(n=n,m=m)]=DeltaPrimeCrit_cylinder
            replica_data_array_DeltaPrimeCombined.loc[dict(n=n,m=m)]= DeltaPrimeDimless-DeltaPrimeCrit
            replica_data_array_DeltaPrimeCombined_cylinder.loc[dict(n=n,m=m)]= DeltaPrimeDimless-DeltaPrimeCrit_cylinder
            replica_data_array_linOnWmarg.loc[dict(n=n,m=m)]=X0/marg_wbar #Shoud divide or mult. by psifac?! No, because marg_wbar, X0 both dimless
            replica_data_array_linOnWmargR.loc[dict(n=n,m=m)]=X0/marg_wbarR #Shoud divide or mult. by psifac?! No, because marg_wbar, X0 both dimless
            replica_data_array_linOnWc.loc[dict(n=n,m=m)]=X0/marg_wbar_Wc #Shoud divide or mult. by psifac?! No, because marg_wbar, X0 both dimless
            replica_data_array_linOnWcR.loc[dict(n=n,m=m)]=X0/marg_wbarR_WcR #Shoud divide or mult. by psifac?! No, because marg_wbar, X0 both dimless
            #Linear tearing local values
            #replica_data_array_approx_minor_radius.loc[dict(n=n,m=m)]=approx_minor_radius
            #replica_data_array_dJt_drs.loc[dict(n=n,m=m)]=dJt_dr
            #replica_data_array_dJt2_drs.loc[dict(n=n,m=m)]=dJt2_dr
            replica_data_array_Jbss.loc[dict(n=n,m=m)]=Jbs
            replica_data_array_Jbs_on_Jt2s.loc[dict(n=n,m=m)]=Jbs_on_Jt2
            #Linear tearing well/hill values
            #replica_data_array_wellhill_approx_minor_radius_location.loc[dict(n=n,m=m)]=wellhill_minor_radius
            #replica_data_array_wellhill_approx_minor_radius_closeness.loc[dict(n=n,m=m)]=minor_radius_closeness
            #replica_data_array_wellhill_steepness.loc[dict(n=n,m=m)]=steepness
            #replica_data_array_iswell.loc[dict(n=n,m=m)]=iswell
            replica_data_array_Btot.loc[dict(n=n,m=m)]=Btot_spln(psifac)
            replica_data_array_Btor.loc[dict(n=n,m=m)]=Btor_spln(psifac)
            replica_data_array_Bpol.loc[dict(n=n,m=m)]=Bpol_spln(psifac)
            replica_data_array_r.loc[dict(n=n,m=m)]=r_spln(psifac)
            replica_data_array_R.loc[dict(n=n,m=m)]=R_spln(psifac)
            replica_data_array_one_on_R.loc[dict(n=n,m=m)]=one_on_R_spln(psifac)
            replica_data_array_JdotBpols.loc[dict(n=n,m=m)]=JdotBpols_spln(psifac)
            replica_data_array_JdotBtors.loc[dict(n=n,m=m)]=JdotBtors_spln(psifac)
            replica_data_array_Jparas.loc[dict(n=n,m=m)]=Jparas_spln(psifac)
            replica_data_array_Jpols.loc[dict(n=n,m=m)]=Jpols_spln(psifac)
            replica_data_array_Jtors.loc[dict(n=n,m=m)]=Jtors_spln(psifac)
            replica_data_array_Jboot_dot_Bs.loc[dict(n=n,m=m)]=Jboot_dot_Bs_spln(psifac)
            replica_data_array_Avg_Btot.loc[dict(n=n,m=m)]=Avg_Btot_spln(psifac)
            replica_data_array_Dnc_simps.loc[dict(n=n,m=m)]=Dnc_simps_spln(psifac)
            replica_data_array_mu0_p.loc[dict(n=n,m=m)]=mu0_p_spln(psifac)
            replica_data_array_p1_on_p.loc[dict(n=n,m=m)]=p1_on_p_spln(psifac)
            replica_data_array_T_e_Kev.loc[dict(n=n,m=m)]=Te_Kev_spln(psifac)
            replica_data_array_T_i_Kev.loc[dict(n=n,m=m)]=Ti_Kev_spln(psifac)
            replica_data_array_v_te.loc[dict(n=n,m=m)]=v_te_spln(psifac)
            replica_data_array_v_ti.loc[dict(n=n,m=m)]=v_ti_spln(psifac)
            replica_data_array_ni.loc[dict(n=n,m=m)]=ni_spln(psifac)
            replica_data_array_ne.loc[dict(n=n,m=m)]=ne_spln(psifac)
            replica_data_array_dp_drs.loc[dict(n=n,m=m)]=dp_drs_spln(psifac)
            replica_data_array_dp_drs2.loc[dict(n=n,m=m)]=dp_drs2_spln(psifac)
            replica_data_array_dq_drs.loc[dict(n=n,m=m)]=dq_drs_spln(psifac)
            replica_data_array_dq_drs2.loc[dict(n=n,m=m)]=dq_drs2_spln(psifac)
            replica_data_array_q.loc[dict(n=n,m=m)]=qspln(psifac)

            if False:
                print("PSIFAC=",psifac,"m,n=",m,n)
                print("wellhill_minor_radius=",wellhill_minor_radius)
                print("minor_radius_closeness=",minor_radius_closeness)
                print("steepness=",steepness)
                print("iswell=",iswell)
                print("DeltaPrime=",DeltaPrimeDimless)
                print("Jbs_on_Jt2=",Jbs_on_Jt2)
                print("Jbs=",Jbs)
                print("dJt2_dr=",dJt2_dr)
                print("approx_minor_radius=",approx_minor_radius)
                print("mufrac=",mufrac)
                
            
    #Numbering islands by marg. width....:
    marg_wbar_nparray=replica_marg_wbar_data_array.values
    marg_wbarR_nparray=replica_marg_wbarR_data_array.values
    sorted_wbar_indices=np.vstack(np.unravel_index(np.argsort(marg_wbar_nparray, axis=None,kind='mergesort'),marg_wbar_nparray.shape)).T
    sorted_wbarR_indices=np.vstack(np.unravel_index(np.argsort(marg_wbarR_nparray, axis=None,kind='mergesort'),marg_wbarR_nparray.shape)).T
    
    for jj in range(len(sorted_wbar_indices)):
        loc_inds=sorted_wbar_indices[jj]
        
        if replica_marg_wbar_data_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])] != replica_marg_wbar_data_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]:
            replica_marg_wbar_order_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]=np.nan
        else:
            replica_marg_wbar_order_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]=(jj+1) #Not sure if gotta swap n, m values...

    #print(replica_marg_wbar_order_array.values)
    #print(marg_wbar_nparray)
    #print(sorted_wbar_indices)
    #raise Exception("ey0")

    for jj in range(len(sorted_wbarR_indices)):
        loc_inds=sorted_wbarR_indices[jj]

        if replica_marg_wbarR_data_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])] != replica_marg_wbarR_data_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]:
            replica_marg_wbarR_order_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]=np.nan
        else:
            replica_marg_wbarR_order_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]=(jj+1) #Not sure if gotta swap n, m values...
        #replica_marg_wbarR_order_array.loc[dict(n=loc_inds[0]+dconxr.Re_DeltaPrime.n.values[0],m=loc_inds[1]+dconxr.Re_DeltaPrime.m.values[0])]=(jj+1) #Not sure if gotta swap n, m values...

    #print(replica_marg_wbarR_order_array.values)
    #print(marg_wbarR_nparray)
    #print(sorted_wbarR_indices)
    #raise Exception("ey0")

    #Linear layer widths & info section
    dconxr=dconxr.assign(sfac=0.0*dconxr['Re_DeltaPrime']+replica_data_array_sfac)
    dconxr=dconxr.assign(tau_r=0.0*dconxr['Re_DeltaPrime']+replica_data_array_tau_r)
    dconxr=dconxr.assign(tau_a=0.0*dconxr['Re_DeltaPrime']+replica_data_array_tau_a)
    dconxr=dconxr.assign(X0=0.0*dconxr['Re_DeltaPrime']+replica_data_array_X0)
    dconxr=dconxr.assign(Qcrit=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Qcrit)
    dconxr=dconxr.assign(DeltaPrimeCrit=0.0*dconxr['Re_DeltaPrime']+replica_data_array_DeltaPrimeCrit)
    dconxr=dconxr.assign(Qcrit_cylinder=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Qcrit_cylinder)
    dconxr=dconxr.assign(DeltaPrimeCrit_cylinder=0.0*dconxr['Re_DeltaPrime']+replica_data_array_DeltaPrimeCrit_cylinder)
    dconxr=dconxr.assign(DeltaPrimeCombined=0.0*dconxr['Re_DeltaPrime']+replica_data_array_DeltaPrimeCombined)
    dconxr=dconxr.assign(DeltaPrimeCombined_cylinder=0.0*dconxr['Re_DeltaPrime']+replica_data_array_DeltaPrimeCombined_cylinder)

    #Macro stability:
    dconxr=dconxr.assign(Drbs=0.0*dconxr['Re_DeltaPrime']+replica_Drbss_data_array) #this is Shi
    dconxr=dconxr.assign(DrbsH_alt=0.0*dconxr['Re_DeltaPrime']+replica_DrbsH_alts_data_array) #This is Heggy1999

    #Delta_nc terms 
    dconxr=dconxr.assign(Dnc=0.0*dconxr['Re_DeltaPrime']+replica_Dnc_data_array)
    dconxr=dconxr.assign(mufrac=0.0*dconxr['Re_DeltaPrime']+replica_mufrac_data_array)
    dconxr=dconxr.assign(q1_on_q=0.0*dconxr['Re_DeltaPrime']+replica_qshear_data_array)
    dconxr=dconxr.assign(mu0_p1_on_Bsq=0.0*dconxr['Re_DeltaPrime']+replica_pshear_data_array)
    dconxr=dconxr.assign(Dnc_geom_fac=0.0*dconxr['Re_DeltaPrime']+replica_Dnc_geom_data_array)

    #Wc terms and sub-terms
    dconxr=dconxr.assign(Wc_no_w=0.0*dconxr['Re_DeltaPrime']+replica_Wc_bar_data_array)
    dconxr=dconxr.assign(Wc_self_consistent=dconxr['Wc_no_w']**(4/3))
    dconxr=dconxr.assign(Wc_no_w_no_modes=dconxr['Wc_no_w']/((dconxr['n']/(dconxr['m']*dconxr['m']))**(1/4)))
    dconxr=dconxr.assign(chi_parallel_no_ne_no_w=0.0*dconxr['Re_DeltaPrime']+replica_data_arraychi_parallel_no_w)
    dconxr=dconxr.assign(chi_parallel_no_ne_no_w_no_n=dconxr['chi_parallel_no_ne_no_w']*dconxr['n'])

    dconxr=dconxr.assign(WcR_no_w=0.0*dconxr['Re_DeltaPrime']+replica_WcR_bar_data_array)
    dconxr=dconxr.assign(Wc_geom_fac=0.0*dconxr['Re_DeltaPrime']+replica_data_arrayWc_geom_fac)
    dconxr=dconxr.assign(Wc_at_marg_wbar=0.0*dconxr['Re_DeltaPrime']+replica_data_arrayWc_at_marg_wbar)
    dconxr=dconxr.assign(Wc_at_X0=dconxr['Wc_no_w']*(dconxr['X0']**(1/4)))
    dconxr=dconxr.assign(WcR_at_marg_wbarR=0.0*dconxr['Re_DeltaPrime']+replica_data_arrayWcR_at_wbarR)
    dconxr=dconxr.assign(Wc_at_marg_wbarR=0.0*dconxr['Re_DeltaPrime']+replica_data_arrayWc_at_wbarR)

    #Mode stability and ranking
    dconxr=dconxr.assign(marg_wbar=0.0*dconxr['Re_DeltaPrime']+replica_marg_wbar_data_array)
    dconxr=dconxr.assign(marg_wbarR=0.0*dconxr['Re_DeltaPrime']+replica_marg_wbarR_data_array)
    dconxr=dconxr.assign(marg_wbar_size_rank=0.0*dconxr['Re_DeltaPrime']+replica_marg_wbar_order_array)
    dconxr=dconxr.assign(marg_wbarR_size_rank=0.0*dconxr['Re_DeltaPrime']+replica_marg_wbarR_order_array)

    #Checking validity of nonlinear analysis
    dconxr=dconxr.assign(linOnWmarg=0.0*dconxr['Re_DeltaPrime']+replica_data_array_linOnWmarg)  
    dconxr=dconxr.assign(linOnWmargR=0.0*dconxr['Re_DeltaPrime']+replica_data_array_linOnWmargR)
    dconxr=dconxr.assign(linOnWc=0.0*dconxr['Re_DeltaPrime']+replica_data_array_linOnWc)
    dconxr=dconxr.assign(linOnWcR=0.0*dconxr['Re_DeltaPrime']+replica_data_array_linOnWcR)

    #LaHaye terms
    dconxr=dconxr.assign(Btot=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Btot)
    dconxr=dconxr.assign(Btor=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Btor)
    dconxr=dconxr.assign(Bpol=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Bpol)
    dconxr=dconxr.assign(r=0.0*dconxr['Re_DeltaPrime']+replica_data_array_r)
    dconxr=dconxr.assign(R=0.0*dconxr['Re_DeltaPrime']+replica_data_array_R)
    dconxr=dconxr.assign(one_on_R=0.0*dconxr['Re_DeltaPrime']+replica_data_array_one_on_R)
    dconxr=dconxr.assign(JdotBpol=0.0*dconxr['Re_DeltaPrime']+replica_data_array_JdotBpols)
    dconxr=dconxr.assign(JdotBtor=0.0*dconxr['Re_DeltaPrime']+replica_data_array_JdotBtors)
    dconxr=dconxr.assign(Jpara=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Jparas)
    dconxr=dconxr.assign(Jpol=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Jpols)
    dconxr=dconxr.assign(Jtor=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Jtors)
    dconxr=dconxr.assign(Jboot_dot_B=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Jboot_dot_Bs)
    dconxr=dconxr.assign(Avg_Btot=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Avg_Btot)
    dconxr=dconxr.assign(Dnc_simp=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Dnc_simps)
    dconxr=dconxr.assign(mu0_p=0.0*dconxr['Re_DeltaPrime']+replica_data_array_mu0_p)
    dconxr=dconxr.assign(p1_on_p=0.0*dconxr['Re_DeltaPrime']+replica_data_array_p1_on_p)
    dconxr=dconxr.assign(Te_Kev=0.0*dconxr['Re_DeltaPrime']+replica_data_array_T_e_Kev)
    dconxr=dconxr.assign(Ti_Kev=0.0*dconxr['Re_DeltaPrime']+replica_data_array_T_i_Kev)
    dconxr=dconxr.assign(v_te=0.0*dconxr['Re_DeltaPrime']+replica_data_array_v_te)
    dconxr=dconxr.assign(v_ti=0.0*dconxr['Re_DeltaPrime']+replica_data_array_v_ti)
    dconxr=dconxr.assign(ni=0.0*dconxr['Re_DeltaPrime']+replica_data_array_ni)
    dconxr=dconxr.assign(ne=0.0*dconxr['Re_DeltaPrime']+replica_data_array_ne)
    dconxr=dconxr.assign(dp_dr=0.0*dconxr['Re_DeltaPrime']+replica_data_array_dp_drs)
    dconxr=dconxr.assign(dp_dr2=0.0*dconxr['Re_DeltaPrime']+replica_data_array_dp_drs2)
    dconxr=dconxr.assign(dq_dr=0.0*dconxr['Re_DeltaPrime']+replica_data_array_dq_drs)
    dconxr=dconxr.assign(dq_dr2=0.0*dconxr['Re_DeltaPrime']+replica_data_array_dq_drs2)
    dconxr=dconxr.assign(q=0.0*dconxr['Re_DeltaPrime']+replica_data_array_q)
    #Important values
    e=1.60217663e-19
    me=9.1091e-31
    dconxr=dconxr.assign(p=dconxr['mu0_p']/mu0)
    dconxr=dconxr.assign(Lp=dconxr['p']/dconxr['dp_dr'])
    dconxr=dconxr.assign(Lp_alt=dconxr['p']/dconxr['dp_dr2'])
    dconxr=dconxr.assign(Lp_flux=1.0/(dconxr['p1_on_p']*2.0*dconxr['psifac']))
    dconxr=dconxr.assign(Lq=dconxr['q']/dconxr['dq_dr'])
    dconxr=dconxr.assign(Lq_alt=dconxr['q']/dconxr['dq_dr2'])
    dconxr=dconxr.assign(eps_local=dconxr['r']/dconxr['R']) 
    dconxr=dconxr.assign(pe=dconxr['Te_Kev']*1e3*dconxr['ne']*e)
    dconxr=dconxr.assign(Beta_pe=2*mu0*dconxr['pe']/(dconxr['Bpol']**2))
    dconxr=dconxr.assign(Beta=2*dconxr['mu0_p']/(dconxr['Btot']**2))
    mi=tok_xr.average_ion_mass[0].values*1.66054e-27
    dconxr=dconxr.assign(rho_i_theta=mi*dconxr['v_ti']/(dconxr['Bpol']*e)) #rho_i_theta = v_ti/(Bpol*one_on_R*T_i_Kev)
    dconxr=dconxr.assign(rho_e_theta=me*dconxr['v_te']/(dconxr['Bpol']*e)) #rho_e_theta = v_te/(Bpol*one_on_R*T_e_Kev)
    dconxr=dconxr.assign(Hayef1=np.abs(dconxr['Beta_pe']*dconxr['r']/dconxr['Lp']))
    dconxr=dconxr.assign(Hayef1_alt=np.abs(dconxr['Beta_pe']*np.sqrt(dconxr['psifac'])*tok_xr.minor_radius[0].values/dconxr['Lp_alt']))
    dconxr=dconxr.assign(Hayef1_alt2=np.abs(dconxr['Beta_pe']/dconxr['Lp_flux']))
    dconxr=dconxr.assign(Hayef2=dconxr['rho_i_theta']/dconxr['r'])
    dconxr=dconxr.assign(Hayef2_alt=dconxr['rho_i_theta']/(np.sqrt(dconxr['psifac'])*tok_xr.minor_radius[0].values))

    #Min cases etc...
    dconxr=dconxr.assign(min_marg_wbar=dconxr.marg_wbar.min().values)
    dconxr=dconxr.assign(min_marg_wbarR=dconxr.marg_wbarR.min().values)
    dconxr=dconxr.assign(max_linOnWmarg=dconxr.linOnWmarg.max().values)
    dconxr=dconxr.assign(max_linOnWmargR=dconxr.linOnWmargR.max().values)
    dconxr=dconxr.assign(max_linOnWc=dconxr.linOnWc.max().values) 
    dconxr=dconxr.assign(max_linOnWcR=dconxr.linOnWcR.max().values)

    #Linear tearing local values
    #dconxr=dconxr.assign(dJt_dr=0.0*dconxr['Re_DeltaPrime']+replica_data_array_dJt_drs)
    #dconxr=dconxr.assign(dJt2_dr=0.0*dconxr['Re_DeltaPrime']+replica_data_array_dJt2_drs) #This the output (legit one)
    dconxr=dconxr.assign(Jbs=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Jbss)
    dconxr=dconxr.assign(Jbs_on_Jt2=0.0*dconxr['Re_DeltaPrime']+replica_data_array_Jbs_on_Jt2s)
    #Linear tearing well/hill values
    #dconxr=dconxr.assign(approx_minor_radius=0.0*dconxr['Re_DeltaPrime']+replica_data_array_approx_minor_radius)
    #dconxr=dconxr.assign(wellhill_approx_minor_radius_location=0.0*dconxr['Re_DeltaPrime']+replica_data_array_wellhill_approx_minor_radius_location)
    #dconxr=dconxr.assign(wellhill_approx_minor_radius_closeness=0.0*dconxr['Re_DeltaPrime']+replica_data_array_wellhill_approx_minor_radius_closeness)
    #dconxr=dconxr.assign(wellhill_steepness=0.0*dconxr['Re_DeltaPrime']+replica_data_array_wellhill_steepness)
    #dconxr=dconxr.assign(iswell=0.0*dconxr['Re_DeltaPrime']+replica_data_array_iswell)

    #Terms at w marginal
    dconxr=dconxr.assign(Delta_star_at_wmarg=deltaPrime_bar(dconxr['marg_wbar'],dconxr['Re_DeltaPrime'],dconxr['Di'],dconxr['alpha_l_alt']))
    dconxr=dconxr.assign(Delta_nc_at_wmarg=Dnc_term(dconxr['marg_wbar'],dconxr['Wc_at_marg_wbar'],dconxr['k1'],dconxr['Dnc']))
    dconxr=dconxr.assign(Delta_GGJ_at_wmarg=Dh_term(dconxr['marg_wbar'],dconxr['Wc_at_marg_wbar'],dconxr['k1'],dconxr['Dh_alt'],dconxr['alpha_s_alt']))
    #Balance at w marginal
    dconxr=dconxr.assign(dwdt_at_wmarg=dconxr['Delta_nc_at_wmarg']+dconxr['Delta_GGJ_at_wmarg']+dconxr['Delta_star_at_wmarg'])
    dconxr=dconxr.assign(dwdt_scalin=np.sqrt(dconxr['Delta_nc_at_wmarg']*dconxr['Delta_nc_at_wmarg']+dconxr['Delta_GGJ_at_wmarg']*dconxr['Delta_GGJ_at_wmarg']+dconxr['Delta_star_at_wmarg']*dconxr['Delta_star_at_wmarg']))
    dconxr=dconxr.assign(dwdt_at_wmarg_rescaled=dconxr['dwdt_at_wmarg']/dconxr['dwdt_scalin'])
    #Relative size of terms
    dconxr=dconxr.assign(Delta_star_on_Delta_nc_at_wmarg=dconxr['Delta_star_at_wmarg']/dconxr['Delta_nc_at_wmarg'])
    dconxr=dconxr.assign(Delta_GGJ_on_Delta_nc_at_wmarg=dconxr['Delta_GGJ_at_wmarg']/dconxr['Delta_nc_at_wmarg'])
    #Terms at d_r ~ X0
    dconxr=dconxr.assign(Delta_star_at_X0=deltaPrime_bar(dconxr['X0'],dconxr['Re_DeltaPrime'],dconxr['Di'],dconxr['alpha_l_alt']))
    dconxr=dconxr.assign(Delta_nc_at_X0=Dnc_term(dconxr['X0'],dconxr['Wc_at_X0'],dconxr['k1'],dconxr['Dnc']))
    dconxr=dconxr.assign(Delta_GGJ_at_X0=Dh_term(dconxr['X0'],dconxr['Wc_at_X0'],dconxr['k1'],dconxr['Dh_alt'],dconxr['alpha_s_alt']))

    #DeltaPrimeTerms
    dconxr=dconxr.assign(Delta_star_w_scaling_at_wmarg=((1/2)*dconxr['marg_wbar'])**(-2*dconxr['alpha_l_alt']))
    dconxr=dconxr.assign(Delta_star_w_scaling_at_X0=((1/2)*dconxr['X0'])**(-2*dconxr['alpha_l_alt']))
    dconxr=dconxr.assign(Delta_star_Di_scaling=np.sqrt(-4*dconxr['Di']))
    dconxr=dconxr.assign(Delta_star_scaling_at_wmarg=0.0*dconxr['Re_DeltaPrime']+replica_data_array_DeltaPrime_pscaling)
    dconxr=dconxr.assign(Delta_star_Rscaling_at_wmargR=0.0*dconxr['Re_DeltaPrime']+replica_data_array_DeltaPrime_pscalingR)
    dconxr=dconxr.assign(Delta_star_alphapower=-2*dconxr['alpha_l_alt'])

    #Curv effects/extra
    dconxr=dconxr.assign(Dr_minus_Di=dconxr['Dr']-dconxr['Di'])
    dconxr=dconxr.assign(Delta_GGJ_small_scaling=1.0+dconxr['alpha_s_alt'])

    #Averages over modes
    dconxr=dconxr.assign(avg_Re_DeltaPrime=dconxr.Re_DeltaPrime.mean().values)
    dconxr=dconxr.assign(avg_Delta_star_Di_scaling=dconxr.Delta_star_Di_scaling.mean().values)
    dconxr=dconxr.assign(avg_Delta_star_alphapower=dconxr.Delta_star_alphapower.mean().values)
    dconxr=dconxr.assign(avg_Delta_star_at_wmarg=dconxr.Delta_star_at_wmarg.mean().values)

    dconxr=dconxr.assign(avg_Dnc=dconxr.Dnc.mean().values)
    dconxr=dconxr.assign(avg_Dr=dconxr.Dr.mean().values)
    dconxr=dconxr.assign(avg_Dr_minus_Di=dconxr.Dr_minus_Di.mean().values)

    dconxr=dconxr.assign(avg_Drbs=dconxr.Drbs.mean().values)
    dconxr=dconxr.assign(avg_DrbsH_alt=dconxr.DrbsH_alt.mean().values)

    dconxr=dconxr.assign(avg_q1_on_q=dconxr.q1_on_q.mean().values)
    #dconxr=dconxr.assign(avg_dJt2_dr=dconxr.dJt2_dr.mean().values)
    #dconxr=dconxr.assign(avg_mu0_p1_on_Bsq=dconxr.mu0_p1_on_Bsq.mean().values)
    #dconxr=dconxr.assign(avg_Dnc_geom_fac=dconxr.Dnc_geom_fac.mean().values)
    dconxr=dconxr.assign(avg_mufrac=dconxr.mufrac.mean().values)

    dconxr=dconxr.assign(avg_Wc_no_w_no_modes=dconxr.Wc_no_w_no_modes.mean().values)
    dconxr=dconxr.assign(avg_chi_parallel_no_ne_no_w_no_n=dconxr.chi_parallel_no_ne_no_w_no_n.mean().values)
    dconxr=dconxr.assign(avg_Wc_geom_fac=dconxr.Wc_geom_fac.mean().values)

    dconxr=dconxr.assign(avg_Jbs=dconxr.Jbs.mean().values)
    dconxr=dconxr.assign(avg_Jbs_on_Jt2=dconxr.Jbs_on_Jt2.mean().values)
    
    #print(dconxr.psifacs.values)
    #print(dconxr.Dnc.values)
    #print(dconxr.Dncs.values)
    #print(dconxr.Dh_alt.values)
    #print(dconxr.Re_DeltaPrime.values)
    #print(dconxr.Wc_no_w.values)
    #print(dconxr.WcR_no_w.values)
    if verbose:
        print("dW_total")
        print(dconxr.dW_total.values)
        print("ns_rdcon_ran")
        print(dconxr.ns_rdcon_ran.values)
        print("psifac")
        print(dconxr.psifac.values)
        print("Re_DeltaPrime")
        print(dconxr.Re_DeltaPrime.values)
        print("DeltaPrimeCrit")
        print(dconxr.DeltaPrimeCrit.values)
        print("DeltaPrimeCrit_cylinder")
        print(dconxr.DeltaPrimeCrit_cylinder.values)
        print("Dr")
        print(dconxr.Dr.values)
        print("Drbss")
        print(dconxr.Drbss.values)
        print("X0")
        print(dconxr.X0.values)
        print("Dnc")
        print(dconxr.Dnc.values)
        print("Wc_no_w")
        print(dconxr.Wc_no_w.values)
        print("WcR_no_w")
        print(dconxr.WcR_no_w.values)
        print("q1_on_q")
        print(dconxr.q1_on_q.values)
        print("mu0_p1_on_Bsq")
        print(dconxr.mu0_p1_on_Bsq.values)
        print("Dnc_geom_fac")
        print(dconxr.Dnc_geom_fac.values)
        print("marg_wbar")
        print(dconxr.marg_wbar.values)
        print("marg_wbarR")
        print(dconxr.marg_wbarR.values)
        print("Wc_at_marg_wbar")
        print(dconxr.Wc_at_marg_wbar.values)
        print("WcR_at_marg_wbarR")
        print(dconxr.WcR_at_marg_wbarR.values)
        print("marg_wbar_size_rank")
        print(dconxr.marg_wbar_size_rank.values)
        print("marg_wbarR_size_rank")
        print(dconxr.marg_wbarR_size_rank.values)
        print("linOnWmarg")
        print(dconxr.linOnWmarg.values)
        print("linOnWmargR")
        print(dconxr.linOnWmargR.values)
        print("linOnWc")
        print(dconxr.linOnWc.values)
        print("linOnWcR")
        print(dconxr.linOnWcR.values)
        print("DeltaPrime_pscaling")
        print(dconxr.DeltaPrime_pscaling.values)
        print("DeltaPrimeCombined")
        print(dconxr.DeltaPrimeCombined.values)
        print("DeltaPrimeCombined_cylinder")
        print(dconxr.DeltaPrimeCombined_cylinder.values)
        print("debuggle")
        #raise


    return dconxr

def wd_alt(dconxr,tok_xr,verbose=False,Zeff=1.5,temp_chi_perp=0.0,reducedm=0,reducedn=0):
    #Fitz uses 16
    #Wesson uses
    me=9.1091e-31

    psio=dconxr.psio[0].values

    chi_perp=dconxr.chi_perp_no_ne[0].values
    if temp_chi_perp !=0.0:
        chi_perp=temp_chi_perp

    dconxr=dconxr.assign(lnLamb_ei = 15.2-0.5*np.log(dconxr['ne']/1e20)+np.log(dconxr['Te_Kev']))
    dconxr=dconxr.assign(taue_ei=1.09*(10**16)*(dconxr['Te_Kev']**(3/2))/(dconxr['ne']*dconxr['lnLamb_ei'])) #in seconds
    dconxr=dconxr.assign(chi_para_smfp=1.581*dconxr['taue_ei']*dconxr['v_te']**2/(1+0.2535*tok_xr['z_effective'][0]))
    #2in denom, 2 in numerator,(rosenburg 33, Fitz 14.206)
    R0=tok_xr['major_radius'][0].values
    dconxr=dconxr.assign(chi_para_lmfp_no_wbar=2*R0*dconxr['v_te']*dconxr['psifac']/(np.sqrt(np.pi)*dconxr['n']*dconxr['flux_MRE_shear_s'])) #Divide by wbar to get chi_parallel_lmfp
    
    Wc_prefacs_spln=CubicSpline(dconxr.psifacs[0].values,dconxr.Wc_prefacs[0].values) #Mult by chifrac, div by m^2 to get thing inside ()**(1/4)

    temp_data_array=dconxr.Re_DeltaPrime.copy(deep=True)
    temp_data_array2=dconxr.Re_DeltaPrime.copy(deep=True)
    for n in dconxr.n.values:
        for m in dconxr.m.values:
            if reducedm!=0 and m!=reducedm:
                continue
            if reducedn!=0 and n!=reducedn:
                continue
            dconxr_m_n=dconxr.sel(n=n,m=m)
            chi_para_smfp=dconxr_m_n.chi_para_smfp.values
            chi_para_lmfp_no_wbar=dconxr_m_n.chi_para_lmfp_no_wbar.values

            Wc_prefac=Wc_prefacs_spln(dconxr_m_n.psifac.values)
            wd_bar_to_power_of_4_no_chifrac_full=(1/psio)**4*Wc_prefac/(dconxr_m_n.m.values**2)
            wd_bar_to_power_of_4_no_chifrac_fitz=(2*np.sqrt(8))**4*(1/(dconxr_m_n.eps_local.values*2.0*dconxr_m_n.flux_MRE_shear_s.values*dconxr_m_n.n.values)**(2))
            wd_bar_to_power_of_4_no_chifrac_fitz_all_radial=(np.sqrt(8))**4*(1/(dconxr_m_n.eps_local.values*(dconxr_m_n.r.values/dconxr_m_n.Lq)*dconxr_m_n.n.values)**(2))
            wd_bar_start=dconxr_m_n.X0.values**4
            wd_bar4_1=wd_bar_start #wd_bar_1: full geometry,               full chi_parallel
            wd_bar4_2=wd_bar_start #wd_bar_2: Fitz geometry (flux space),  full chi_parallel
            wd_bar4_3=wd_bar_start #wd_bar_3: full geometry,               simple chi_parallel
            wd_bar4_4=wd_bar_start #wd_bar_4: Fitz geometry (flux space),  simple chi_parallel
            wd_bar4_5=wd_bar_start #wd_bar_5: Fitz geometry (meters),      full chi_parallel
            print(f"m,n = {m},{n} wd_bar_start: {wd_bar_start**(1/4):.3e} wd_bar_to_power_of_4_no_chifrac_full: {wd_bar_to_power_of_4_no_chifrac_full:.3e} wd_bar_to_power_of_4_no_chifrac_fitz: {wd_bar_to_power_of_4_no_chifrac_fitz:.3e}")
            for i in range(10):
                wd_bar4_1 = wd_bar_to_power_of_4_no_chifrac_full*(chi_perp/(chi_para_smfp*chi_para_lmfp_no_wbar/(chi_para_lmfp_no_wbar+chi_para_smfp*wd_bar4_1**(1/4))))
                wd_bar4_2 = wd_bar_to_power_of_4_no_chifrac_fitz*(chi_perp/(chi_para_smfp*chi_para_lmfp_no_wbar/(chi_para_lmfp_no_wbar+chi_para_smfp*wd_bar4_2**(1/4))))
                wd_bar4_3 = wd_bar_to_power_of_4_no_chifrac_full*(chi_perp/(chi_para_lmfp_no_wbar/wd_bar4_3**(1/4)))
                wd_bar4_4 = wd_bar_to_power_of_4_no_chifrac_fitz*(chi_perp/(chi_para_lmfp_no_wbar/wd_bar4_4**(1/4)))
                wd_bar4_5 = wd_bar_to_power_of_4_no_chifrac_fitz_all_radial*(chi_perp/(chi_para_smfp*chi_para_lmfp_no_wbar/(chi_para_lmfp_no_wbar+chi_para_smfp*wd_bar4_2**(1/4))))
                if True:
                    if i==0:
                        print("m,n = ",m,n," ",i,f" wd_bar_1: {wd_bar_start**(1/4):.3e} wd_bar_2: {wd_bar_start**(1/4):.3e} wd_bar_3: {wd_bar_start**(1/4):.3e} wd_bar_4: {wd_bar_start**(1/4):.3e} wd_bar_5: {wd_bar_start**(1/4):.3e} chi_para_lmfp: {chi_para_lmfp_no_wbar/(wd_bar4_1**(1/4)):.3e} chi_para_smfp: {chi_para_smfp:.3e}")
                    if i ==9:
                        print("           ",i,f" wd_bar_1: {wd_bar4_1**(1/4):.3e} wd_bar_2: {wd_bar4_2**(1/4):.3e} wd_bar_3: {wd_bar4_3**(1/4):.3e} wd_bar_4: {wd_bar4_4**(1/4):.3e} wd_bar_5: {wd_bar4_5**(1/4):.3e} chi_para_lmfp: {chi_para_lmfp_no_wbar/(wd_bar4_1**(1/4)):.3e} chi_para_smfp: {chi_para_smfp:.3e}, chi_para: {(chi_para_smfp*chi_para_lmfp_no_wbar/(chi_para_lmfp_no_wbar+chi_para_smfp*wd_bar4_1**(1/4))):3e}")
                    else:
                        print("           ",i,f" wd_bar_1: {wd_bar4_1**(1/4):.3e} wd_bar_2: {wd_bar4_2**(1/4):.3e} wd_bar_3: {wd_bar4_3**(1/4):.3e} wd_bar_4: {wd_bar4_4**(1/4):.3e} wd_bar_5: {wd_bar4_5**(1/4):.3e} chi_para_lmfp: {chi_para_lmfp_no_wbar/(wd_bar4_1**(1/4)):.3e} chi_para_smfp: {chi_para_smfp:.3e}")
            print(f"Old, exact soln = {dconxr_m_n.Wc_self_consistent.values:.3e}, Old, exact cylindrical solution = {dconxr_m_n.WcR_no_w.values**(4/3):.3e}")

    return np.nan

#def get_wd_self_consistent(dconxr,verbose=False):
#    dconxr.Wc_no_w #these are Wc_bar without w_mar
#    w_C_Bar=Wc_Bar_prefac*(w_mar**(1/4))   

#    replica_data_array=dconxr.Wc_no_w.copy(deep=True)

#    for n in dconxr.n.values:
#        for m in dconxr.m.values: 
#            Wc_no_w=dconxr.Wc_no_w.sel(n=n,m=m)
#            wC_selfconsistent=wC_selfconsistent_solver(Wc_no_w)

#def wC_selfconsistent_solver(Wc_no_w,startpoint=1e-10,rhs=1.0,tol=1e-12):
#    while d_shift<tol:
#        wC_selfconsistent=startpoint
#        d_shift=1.0
#        while d_shift > 1e-8:
#            wC_selfconsistent_new=wC_selfconsistent-(Wc_no_w/wC_selfconsistent**(1/4))
#            d_shift=np.abs(wC_selfconsistent_new-wC_selfconsistent)
#            wC_selfconsistent=wC_selfconsistent_new


def neo_surf_term_scalars_local(dconxr,drop_ends=False):
    #Dnc_spline=CubicSpline(dconxr.psifacs.values,dconxr.Dncs.values)
    #Hbs_spline=CubicSpline(dconxr.psifacs.values,dconxr.Hbss.values)
    #Drbs_spline=CubicSpline(dconxr.psifacs.values,dconxr.Drbss.values)

    if drop_ends:
        dconxr=dconxr.where(dconxr.qs> 1.2, drop=True)
        dconxr=dconxr.where(dconxr.qs< 2.1, drop=True)
        dconxr=dconxr.where(dconxr.psifacs < 0.9, drop=True)# and dconxr.qs[0]< 2.1 and dconxr.psifacs < 0.9)
        surf_xr=dconxr
    else:
        surf_xr=dconxr.where(dconxr.qs> 1.2, drop=True)
        surf_xr=surf_xr.where(surf_xr.qs< 2.1, drop=True)
        surf_xr=surf_xr.where(surf_xr.psifacs < 0.9, drop=True)# and dconxr.qs[0]< 2.1 and dconxr.psifacs < 0.9)

    #dconxr=dconxr.assign(Drbs_max=surf_xr.Drbss.max())
    #argmax=surf_xr.Drbss.argmax().values
    #dconxr=dconxr.assign(Drbs_max_psi=surf_xr.psifacs[0][argmax].values)
    #dconxr=dconxr.assign(Drbs_max_q=surf_xr.qs[0][argmax].values)

    #dconxr=dconxr.assign(DrbsRS_max=surf_xr.DrbsRSs.max())
    #argmax=surf_xr.DrbsRSs.argmax().values
    #dconxr=dconxr.assign(DrbsRS_max_psi=surf_xr.psifacs[0][argmax].values)
    #dconxr=dconxr.assign(DrbsRS_max_q=surf_xr.qs[0][argmax].values)

    dconxr=dconxr.assign(DrbsH_max=surf_xr.DrbsHs.max())
    argmax=surf_xr.DrbsHs.argmax().values
    dconxr=dconxr.assign(DrbsH_max_psi=surf_xr.psifacs[0][argmax].values)
    dconxr=dconxr.assign(DrbsH_max_q=surf_xr.qs[0][argmax].values)

    dconxr=dconxr.assign(DrbsH_alt_max=surf_xr.DrbsH_alts.max())
    argmax=surf_xr.DrbsH_alts.argmax().values
    dconxr=dconxr.assign(DrbsH_alt_max_psi=surf_xr.psifacs[0][argmax].values)
    dconxr=dconxr.assign(DrbsH_alt_max_q=surf_xr.qs[0][argmax].values)

    dconxr=dconxr.assign(qshear_max=surf_xr.q1_on_qs.max())
    argmax=surf_xr.q1_on_qs.argmax().values
    dconxr=dconxr.assign(qshear_max_psi=surf_xr.psifacs[0][argmax].values)
    dconxr=dconxr.assign(qshear_max_q=surf_xr.qs[0][argmax].values)

    #dconxr=dconxr.assign(pshear_max=surf_xr.mu0_p1_on_Bsqs.max())
    #argmax=surf_xr.mu0_p1_on_Bsqs.argmax().values
    #dconxr=dconxr.assign(pshear_max_psi=surf_xr.psifacs[0][argmax].values)
    #dconxr=dconxr.assign(pshear_max_q=surf_xr.qs[0][argmax].values)

    #dconxr=dconxr.assign(Dnc_geom_fac_max=surf_xr.Dnc_geom_facs.max())
    #argmax=surf_xr.Dnc_geom_facs.argmax().values
    #dconxr=dconxr.assign(Dnc_geom_fac_max_psi=surf_xr.psifacs[0][argmax].values)
    #dconxr=dconxr.assign(Dnc_geom_fac_max_q=surf_xr.qs[0][argmax].values)

    #dconxr=dconxr.assign(Drbs_mean=surf_xr.Drbss.mean())
    dconxr=dconxr.assign(DrbsH_mean=surf_xr.DrbsHs.mean())
    dconxr=dconxr.assign(DrbsH_alt_mean=surf_xr.DrbsH_alts.mean())
    #dconxr=dconxr.assign(DrbsRSs_mean=surf_xr.DrbsRSs.mean())
    #dconxr=dconxr.assign(Drbs_rel_Diffs_mean=surf_xr.Drbs_rel_Diffs.mean())
    dconxr=dconxr.assign(qshear_mean=surf_xr.q1_on_qs.mean())
    #dconxr=dconxr.assign(pshear_mean=surf_xr.mu0_p1_on_Bsqs.mean())
    #dconxr=dconxr.assign(Dnc_geom_fac_mean=surf_xr.Dnc_geom_facs.mean())
    dconxr=dconxr.assign(Jbs_frac_psiavg=surf_xr.Jbs_on_Jt2s.mean())
    dconxr=dconxr.assign(Jbs_psiavg=surf_xr.Jbss.mean())

    #print("Jbs_on_Jt2s_mean=",surf_xr.Jbs_on_Jt2s.mean())
    #print("Jbss_mean=",surf_xr.Jbss.mean())

    return dconxr


#Calculates fraction of current that is bootstrap vs total current
def neo_surf_term_scalars2_local(dconxr,dcon_mcind,tok_xr,verbose=False):
    local_tok_xr=tok_xr
    local_tok_xr=local_tok_xr.assign(minor_radius_dim_rho=np.sqrt(local_tok_xr['psi_N'])*local_tok_xr['minor_radius'][0]*np.sqrt(local_tok_xr['areal_elongation'][0]))
    #print(local_tok_xr.point_type.values)
    #raise Exception("EY0")
    bs_frac2,bs_tot,Itot2 = get_bs_frac(local_tok_xr)
    bs_frac_simple=local_tok_xr.bootstrap_fraction[0].values
    #if verbose:
    #    print(f"bs_frac={bs_frac},bs_frac2={bs_frac2},bs_frac_simple={bs_frac_simple}")
    #dconxr=dconxr.assign(bs_frac=bs_frac)
    dconxr=dconxr.assign(bs_frac2=bs_frac2)
    dconxr=dconxr.assign(bs_frac_simple=bs_frac_simple)
    dconxr=dconxr.assign(Ibs_tot=bs_tot)
    dconxr=dconxr.assign(Itot2=Itot2)
    if False:
        print("bs_frac=",bs_frac)
        print("bs_frac2=",bs_frac2)
        print("bs_frac_simple=",bs_frac_simple)
        print("bs_tot=",bs_tot)
        print("Itot2=",Itot2)
    return dconxr


def MRE_haye_2017(dcon_xr):
    # This broken: dcon_xr=dcon_xr.assign(JdotB=dcon_xr['JdotBtor']+dcon_xr['JdotBpol']) #JdotB is the total JdotB, not just toroidal
    #Lq/rs = (q/(r*dq/dr))_rs = (2(psi dq/dpsi/q)^(-1)
    #dcon_xr=dcon_xr.assign(HAYE_MRE_shear_s=2.0*dcon_xr['psifac']*dcon_xr['q1_on_q'])
    dcon_xr=dcon_xr.assign(flux_MRE_shear_s=dcon_xr['psifac']*dcon_xr['q1_on_q'])
    #dcon_xr=dcon_xr.assign(HAYE_MRE_c1=3*(dcon_xr['Jboot_dot_B']/dcon_xr['JdotB'])/dcon_xr['HAYE_MRE_shear_s'])
    dcon_xr=dcon_xr.assign(HAYE_MRE_c1=3*(np.abs(0.4*(1000)*dcon_xr['Jboot_dot_B']/(dcon_xr['Avg_Btot']*dcon_xr['Jpara'])))/dcon_xr['flux_MRE_shear_s'])
    #Term 1: 1/w_normalised (flux space)
    #Term 2: 1/w_normalised^3 (flux space)
    #Term 3: 1/w_normalised (flux space)
    dcon_xr=dcon_xr.assign(HAYE_MRE_term2=-3.0*(2.0**2)*(np.sqrt(dcon_xr['eps_local'])*dcon_xr['rho_i_theta']/dcon_xr['r'])**2)
    return dcon_xr

def dWdtau_MRE_haye_2017(w_mar,DeltaPrimeDimless,Di,alpha_l_alt,HAYE_MRE_c1,HAYE_MRE_term2,HAYE_MRE_term3=0.0):
    return (deltaPrime_bar(w_mar,DeltaPrimeDimless,Di,alpha_l_alt)+HAYE_MRE_c1/w_mar+HAYE_MRE_c1*HAYE_MRE_term2/(w_mar**3)+HAYE_MRE_c1*HAYE_MRE_term3/w_mar)

def solve_marg_width_haye_2017(dconxr,CDK1=0.5,verbose=False,mchoose=0):
    dconxr=MRE_haye_2017(dconxr)

    replica_data_array1=dconxr.Re_DeltaPrime.copy()
    replica_data_array2=dconxr.Re_DeltaPrime.copy()
    replica_data_array3=dconxr.Re_DeltaPrime.copy()
    replica_data_array4=dconxr.Re_DeltaPrime.copy()
    replica_data_array5=dconxr.Re_DeltaPrime.copy()
    replica_data_array6=dconxr.Re_DeltaPrime.copy()

    wc_bar_vals=np.logspace(-8,0,num=1000)
    for n in dconxr.n.values:
        for m in dconxr.m.values:
            dconxr_n_m=dconxr.sel(n=n,m=m)

            replica_data_array1.loc[dict(n=n,m=m)]=np.nan #w_marg_haye_2017
            replica_data_array2.loc[dict(n=n,m=m)]=np.nan #haye_2017_max_location
            replica_data_array3.loc[dict(n=n,m=m)]=np.nan #haye_2017_max
            replica_data_array4.loc[dict(n=n,m=m)]=np.nan #jcd_on_jbs
            replica_data_array5.loc[dict(n=n,m=m)]=np.nan #jcd_on_jparallel
            replica_data_array6.loc[dict(n=n,m=m)]=np.nan #jcd

            if np.isnan(dconxr_n_m.Re_DeltaPrime.values) or np.isnan(dconxr_n_m.Di.values) or np.isnan(dconxr_n_m.alpha_l_alt.values) or np.isnan(dconxr_n_m.HAYE_MRE_c1.values) or np.isnan(dconxr_n_m.HAYE_MRE_term2.values):
                continue

            if dconxr_n_m.Re_DeltaPrime.values>0:
                continue

            if dconxr_n_m.Di.values>0 or dconxr_n_m.Dr.values>0:
                continue

            dwdtau_loc = lambda w_mar: dWdtau_MRE_haye_2017(w_mar,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Di.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.HAYE_MRE_c1.values,dconxr_n_m.HAYE_MRE_term2.values)
            dwdtau_vals=dwdtau_loc(wc_bar_vals)

            if max(dwdtau_vals)<0:
                replica_data_array1.loc[dict(n=n,m=m)]=1.0
                if verbose:
                    if mchoose!=0 and m==mchoose:
                        xs,ys,nancheck=solve_marg_width(dconxr_n_m.Wc_no_w.values,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Dh_alt.values,dconxr_n_m.Di.values,dconxr_n_m.Dnc.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.alpha_s_alt.values,plot=True)
                        if not np.isnan(nancheck):
                            plt.plot(np.log10(xs),ys,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}, Dnc={dconxr_n_m.Dnc.values}, Wd")
                        xs,ys,nancheck2=solve_marg_width(dconxr_n_m.Wc_no_w.values,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Dh_alt.values,dconxr_n_m.Di.values,dconxr_n_m.Dnc.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.alpha_s_alt.values,plot=True,min_C_Bar_opt=True)
                        if not np.isnan(nancheck2):
                            plt.plot(np.log10(xs),ys,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}, minWd")
                            plt.plot(np.log10(xs),xs/(xs**2+dconxr_n_m.Wc_self_consistent.values**2*1.7/0.5),label="prefac")
                            plt.plot(np.log10(xs),1.7*dconxr_n_m.Dnc.values*xs/(xs**2+dconxr_n_m.Wc_self_consistent.values**2*1.7/0.5),label="prefacComb")
                            dhs=[Dh_term(xi,dconxr_n_m.Wc_self_consistent.values,1.7,dconxr_n_m.Dh_alt.values,dconxr_n_m.alpha_s_alt.values) for xi in xs]
                            plt.plot(np.log10(xs),dhs,label="Dh_term")
                            plt.vlines([np.log10(dconxr_n_m.Wc_self_consistent),np.log10(dconxr_n_m.X0)],-300,100)
                        plt.plot(np.log10(wc_bar_vals),dwdtau_vals,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}")
                        plt.ylim(-300,20)
                        plt.legend()
                continue
            
            dwdtau_spln=CubicSpline(wc_bar_vals,dwdtau_vals,extrapolate=False) 
            dwdtau_deriv_spln=CubicSpline(wc_bar_vals,dwdtau_spln(wc_bar_vals,1),extrapolate=False) 

            if dwdtau_vals[0]>0:
                replica_data_array1.loc[dict(n=n,m=m)]=np.nan
            else:
                replica_data_array1.loc[dict(n=n,m=m)]==dwdtau_spln.roots(extrapolate=False)[0]

            #Solving for CD stabilisation:
            try:
                haye_2017_max_location=dwdtau_deriv_spln.roots(extrapolate=False)[0]
                if verbose: 
                    print(f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values},Di={dconxr_n_m.Di.values},alpha_l_alt={dconxr_n_m.alpha_l_alt.values},HAYE_MRE_c1={dconxr_n_m.HAYE_MRE_c1.values},HAYE_MRE_term2={dconxr_n_m.HAYE_MRE_term2.values},MRE_shear_s={dconxr_n_m.flux_MRE_shear_s.values},Jboot_dot_B={dconxr_n_m.Jboot_dot_B.values}")
                    if mchoose!=0 and m==mchoose:
                        xs,ys,nancheck=solve_marg_width(dconxr_n_m.Wc_no_w.values,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Dh_alt.values,dconxr_n_m.Di.values,dconxr_n_m.Dnc.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.alpha_s_alt.values,plot=True)
                        if not np.isnan(nancheck):
                            plt.plot(np.log10(xs),ys,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}, Dnc={dconxr_n_m.Dnc.values}, Wd")
                        xs,ys,nancheck2=solve_marg_width(dconxr_n_m.Wc_no_w.values,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Dh_alt.values,dconxr_n_m.Di.values,dconxr_n_m.Dnc.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.alpha_s_alt.values,plot=True,min_C_Bar_opt=True)
                        if not np.isnan(nancheck2):
                            plt.plot(np.log10(xs),ys,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}, minWd")
                            plt.plot(np.log10(xs),xs/(xs**2+dconxr_n_m.Wc_self_consistent.values**2*1.7/0.5),label="prefac")
                            plt.plot(np.log10(xs),1.7*dconxr_n_m.Dnc.values*xs/(xs**2+dconxr_n_m.Wc_self_consistent.values**2*1.7/0.5),label="prefacComb")
                            dhs=[Dh_term(xi,dconxr_n_m.Wc_self_consistent.values,1.7,dconxr_n_m.Dh_alt.values,dconxr_n_m.alpha_s_alt.values) for xi in xs]
                            plt.plot(np.log10(xs),dhs,label="Dh_term")
                            plt.vlines([np.log10(dconxr_n_m.Wc_self_consistent),np.log10(dconxr_n_m.X0)],-300,100)
                        plt.plot(np.log10(wc_bar_vals),dwdtau_vals,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}")
                        plt.ylim(-300,20)
                        plt.legend()
            except:
                print('cat')
                print(f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values},Di={dconxr_n_m.Di.values},alpha_l_alt={dconxr_n_m.alpha_l_alt.values},HAYE_MRE_c1={dconxr_n_m.HAYE_MRE_c1.values},HAYE_MRE_term2={dconxr_n_m.HAYE_MRE_term2.values},MRE_shear_s={dconxr_n_m.flux_MRE_shear_s.values},Jboot_dot_B={dconxr_n_m.Jboot_dot_B.values}")
                plt.plot(np.log10(wc_bar_vals),dwdtau_vals,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}")
                plt.ylim(-300,20)
                plt.legend()
                haye_2017_max_location=dwdtau_deriv_spln.roots(extrapolate=False)[0]
                if verbose: 
                    print(f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values},Di={dconxr_n_m.Di.values},alpha_l_alt={dconxr_n_m.alpha_l_alt.values},HAYE_MRE_c1={dconxr_n_m.HAYE_MRE_c1.values},HAYE_MRE_term2={dconxr_n_m.HAYE_MRE_term2.values},MRE_shear_s={dconxr_n_m.flux_MRE_shear_s.values},Jboot_dot_B={dconxr_n_m.Jboot_dot_B.values}")
                    if mchoose!=0 and m==mchoose:
                        xs,ys,nancheck=solve_marg_width(dconxr_n_m.Wc_no_w.values,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Dh_alt.values,dconxr_n_m.Di.values,dconxr_n_m.Dnc.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.alpha_s_alt.values,plot=True)
                        if not np.isnan(nancheck):
                            plt.plot(np.log10(xs),ys,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}, Dnc={dconxr_n_m.Dnc.values}, Wd")
                        xs,ys,nancheck2=solve_marg_width(dconxr_n_m.Wc_no_w.values,dconxr_n_m.Re_DeltaPrime.values,dconxr_n_m.Dh_alt.values,dconxr_n_m.Di.values,dconxr_n_m.Dnc.values,dconxr_n_m.alpha_l_alt.values,dconxr_n_m.alpha_s_alt.values,plot=True,min_C_Bar_opt=True)
                        if not np.isnan(nancheck2):
                            plt.plot(np.log10(xs),ys,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}, minWd")
                            plt.plot(np.log10(xs),xs/(xs**2+dconxr_n_m.Wc_self_consistent.values**2*1.7/0.5),label="prefac")
                            plt.plot(np.log10(xs),1.7*dconxr_n_m.Dnc.values*xs/(xs**2+dconxr_n_m.Wc_self_consistent.values**2*1.7/0.5),label="prefacComb")
                            dhs=[Dh_term(xi,dconxr_n_m.Wc_self_consistent.values,1.7,dconxr_n_m.Dh_alt.values,dconxr_n_m.alpha_s_alt.values) for xi in xs]
                            plt.plot(np.log10(xs),dhs,label="Dh_term")
                            plt.vlines([np.log10(dconxr_n_m.Wc_self_consistent),np.log10(dconxr_n_m.X0)],-300,100)
                        plt.plot(np.log10(wc_bar_vals),dwdtau_vals,label=f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values}")
                        plt.ylim(-300,20)
                        plt.legend()
                
            haye_2017_max=dwdtau_loc(haye_2017_max_location)
            replica_data_array2.loc[dict(n=n,m=m)]=haye_2017_max_location
            replica_data_array3.loc[dict(n=n,m=m)]=haye_2017_max

            if haye_2017_max>0: #rare case that its not
                if verbose: print(f"n={n},m={m},Dp={dconxr_n_m.Re_DeltaPrime.values},Di={dconxr_n_m.Di.values},alpha_l_alt={dconxr_n_m.alpha_l_alt.values},HAYE_MRE_c1={dconxr_n_m.HAYE_MRE_c1.values},HAYE_MRE_term2={dconxr_n_m.HAYE_MRE_term2.values},MRE_shear_s={dconxr_n_m.flux_MRE_shear_s.values},Jboot_dot_B={dconxr_n_m.Jboot_dot_B.values}")
                jcd_on_jbs=haye_2017_max*haye_2017_max_location/(CDK1*dconxr_n_m.HAYE_MRE_c1.values)
                jcd_on_jparallel=jcd_on_jbs*(np.abs(dconxr_n_m['Jboot_dot_B']/(dconxr_n_m['Avg_Btot']*dconxr_n_m['Jpara'])))
                jcd=jcd_on_jparallel*np.abs(dconxr_n_m.Jpara.values)
                replica_data_array4.loc[dict(n=n,m=m)]=jcd_on_jbs
                replica_data_array5.loc[dict(n=n,m=m)]=jcd_on_jparallel
                replica_data_array6.loc[dict(n=n,m=m)]=jcd

            
    dconxr=dconxr.assign(w_marg_haye_2017=0.0*dconxr['Re_DeltaPrime']+replica_data_array1)
    dconxr=dconxr.assign(haye_2017_max_location=0.0*dconxr['Re_DeltaPrime']+replica_data_array2)
    dconxr=dconxr.assign(haye_2017_max=0.0*dconxr['Re_DeltaPrime']+replica_data_array3)
    dconxr=dconxr.assign(jcd_on_jbs=0.0*dconxr['Re_DeltaPrime']+replica_data_array4)
    dconxr=dconxr.assign(jcd_on_jparallel=0.0*dconxr['Re_DeltaPrime']+replica_data_array5)
    dconxr=dconxr.assign(jcd=0.0*dconxr['Re_DeltaPrime']+replica_data_array6)

    return dconxr





def get_bs_frac(local_tok_xr):
    #print(local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo)
    #raise   Exception("EY0")
    dynamo_Jbss=local_tok_xr.Jbs_GSinput_on_linspace_psi_dynamo.values
    #dynamo_Jtots=local_tok_xr.Jtor_GSinput_on_linspace_psi_dynamo.values
    dynamo_Jtots2=local_tok_xr.Jtor_GSoutput_on_linspace_psi_dynamo.values

    #jind = dynamo_Jtots-dynamo_Jbss
    #jind2 = dynamo_Jtots2-dynamo_Jbss

    Jbs_tot=cross_section_surf_int(local_tok_xr,dynamo_Jbss)
    #Jtot=cross_section_surf_int(local_tok_xr,dynamo_Jtots)
    Jtot2=cross_section_surf_int(local_tok_xr,dynamo_Jtots2)

    return (Jbs_tot/Jtot2),Jbs_tot,Jtot2

def cross_section_surf_int(local_tok_xr,vals_to_int): #Not very reliable... to be more accurate would rely on <r> outputted from Tokamaker.
    minor_radius_values=local_tok_xr.minor_radius_dim_rho.values
    inductive_integration_spline=CubicSpline(minor_radius_values,minor_radius_values*vals_to_int)
    cross_section_surf_int_of_vals=2*np.pi*inductive_integration_spline.integrate(min(minor_radius_values), max(minor_radius_values), extrapolate=False)
    return cross_section_surf_int_of_vals

#w_mar is dimless here!!...
def solve_marg_width(Wc_Bar_prefac,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k0=0.8227,k1=1.7,plot=False,min_C_Bar_opt=False):
    #wc_bar_vals=np.linspace(1e-8,0.4,1000)
    wc_bar_vals=np.logspace(-8,0,num=1000)
    if min_C_Bar_opt:
        dwdtau_loc = lambda w_mar: dWdtau_alt(w_mar,Wc_Bar_prefac,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k0,k1)
    else:
        dwdtau_loc = lambda w_mar: dWdtau(w_mar,Wc_Bar_prefac,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k0,k1)
    dwdtau_vals=dwdtau_loc(wc_bar_vals)

    if dwdtau_vals[0]>0:
        return np.nan,np.nan,np.nan
    if max(dwdtau_vals)<0:
        return 1.0,np.nan,np.nan

    #raise
    dwdtau_spln=CubicSpline(wc_bar_vals,dwdtau_vals,extrapolate=False) 
    w_mar=dwdtau_spln.roots(extrapolate=False)
    if plot:
        return wc_bar_vals,dwdtau_vals,1.0

    if len(w_mar)>0:
        return w_mar[0],(w_mar[0]**(1/4)*Wc_Bar_prefac),((w_mar[0]/2)**(-2*alpha_l_alt))*np.sqrt(-4*Di)
    
    return np.nan,np.nan,np.nan

def dWdtau(w_mar,Wc_Bar_prefac,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k0,k1):
    w_C_Bar=Wc_Bar_prefac*(w_mar**(1/4))
    return (deltaPrime_bar(w_mar,DeltaPrimeDimless,Di,alpha_l_alt)+Dh_term(w_mar,w_C_Bar,k1,Dh_alt,alpha_s_alt)+Dnc_term(w_mar,w_C_Bar,k1,Dnc))

def dWdtau_alt(w_mar,Wc_Bar_prefac,DeltaPrimeDimless,Dh_alt,Di,Dnc,alpha_l_alt,alpha_s_alt,k0,k1):
    #w_C_Bar=Wc_Bar_prefac*(w_mar**(1/4))
    w_C_Bar=Wc_Bar_prefac**(4/3)
    return (deltaPrime_bar(w_mar,DeltaPrimeDimless,Di,alpha_l_alt)+Dh_term(w_mar,w_C_Bar,k1,Dh_alt,alpha_s_alt)+Dnc_term(w_mar,w_C_Bar,k1,Dnc))

#Whole thing w^-1
def deltaPrime_bar(w_mar,DeltaPrimeDimless,Di,alpha_l_alt):
    return DeltaPrimeDimless*(w_mar/2)**(-2*alpha_l_alt)*np.sqrt(-4*Di)

#Read as DHegna
#Should be k1, not k0, error is present in Schlutt and Hegna 2012
#Whole thing w^-1
def Dh_term(w_mar,w_C_Bar,k1,Dh_alt,alpha_s_alt):
    return k1*Dh_alt/(w_mar+w_C_Bar*k1/(0.3*(1+alpha_s_alt)))

 #Whole thing w^-1
def Dnc_term(w_mar,w_C_Bar,k1,Dnc):
    return k1*Dnc*w_mar/(w_mar**2+(w_C_Bar**2)*k1/0.5)

def neo_terms(dcon_xr,dcon_mcind,tok_xr):
    #Key surface terms, added to modes for convenience
    dcon_xr=neo_terms_on_psifacs1(dcon_xr,dcon_mcind,tok_xr)
    dcon_xr=neo_terms_on_psifacs1b(dcon_xr,dcon_mcind,tok_xr,verbose=False) #shear and Dnc breakdown
    dcon_xr=lin_terms_on_psifacs(dcon_xr,dcon_mcind,tok_xr) #dJt_on_dr, well and hill info, Jt, Jbs

    dcon_xr=neo_surf_term_scalars(dcon_xr)
    dcon_xr=neo_terms_on_psifacs2(dcon_xr,dcon_mcind,tok_xr)
    dcon_xr=neo_terms_on_psifacs2b(dcon_xr,dcon_mcind,tok_xr) #X0, Qcrit, Delta'_crit
    dcon_xr=neo_surf_term_scalars2(dcon_xr,dcon_mcind,tok_xr) #bootstrap fraction ONLY
    dcon_xr=neo_terms_on_modes(dcon_xr)

    #raise
    #dcon_xr=neo_terms_on_modes(dcon_xr)
    #   !!!Need Dnc, Wc, 
    #   Got Dh, Dh_alt
    #   Add Drbs, Drbs_RS? Not yet...

    #print(dcon_xr.Drbs_rel_Diffs_mean.values,dcon_xr.DrbsRS_max.values,dcon_xr.DrbsH_alt_max.values)

    #Advanced terms related to marginally stable island width, needed only if key surface terms are positive or Deltaprimes are postive
    #dcon_xr=neo_terms_on_psifacs2(dcon_xr,dcon_mcind,tok_xr)
    #dcon_xr=neo_terms_on_modes2(dcon_xr,dcon_mcind,tok_xr)
    return dcon_xr

#Untested
def compile_dconxrs(dcon_dir,tok_xr,debug=False):
    dcon_unique_mcinds=[]
    dcon_xrs=[]
    dcon_file_list=[i for i in os.listdir(dcon_dir) if not ('xrF' in i)]
    #LOOP FOR DCON XARRAYS 
    #   GATHERING INFO AND XRs
    for dcon_file in dcon_file_list:
        dcon_filespl=dcon_file.split('_')
        dcon_mcind=dcon_filespl[2][3:]
        dcon_Jfrac=float(dcon_filespl[3][5:].replace('p','.'))
        #New: neo_terms_on_psifacs
        if not debug:
            try:
                dcon_xr=xr.open_dataset(dcon_dir+'/'+dcon_file).assign_coords(plasma_current_fraction=dcon_Jfrac)
                dcon_xr=neo_terms(dcon_xr,int(dcon_mcind),tok_xr)
                dcon_xrs.append([dcon_xr,dcon_mcind])
                if not(dcon_mcind in dcon_unique_mcinds):
                    dcon_unique_mcinds.append(dcon_mcind) 
            except:
                print('FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                print('FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                print('\n\n\n\n'+'DCON DIR:'+dcon_dir)
                print('DCON FILE:'+dcon_dir+'/'+dcon_file)
                print('DCON MCIND:'+dcon_mcind)
                continue
        else:
            dcon_xr=xr.open_dataset(dcon_dir+'/'+dcon_file).assign_coords(plasma_current_fraction=dcon_Jfrac)
            dcon_xr=neo_terms(dcon_xr,int(dcon_mcind),tok_xr)
            dcon_xrs.append([dcon_xr,dcon_mcind])
            if not(dcon_mcind in dcon_unique_mcinds):
                dcon_unique_mcinds.append(dcon_mcind)

    #Constuction of whole DCON xr from double-loop
    whole_dcon_xr_vec=[]
    for MCind in dcon_unique_mcinds:
        mcind_xr_vec=[]
        currents=[]
        for dxr in dcon_xrs:
            if dxr[1]==MCind:
                mcind_xr_vec.append(dxr[0])
                currents.append(dxr[0].plasma_current_fraction.values)
        min_current_frac=min(currents)
        whole_dcon_xr_vec.append(xr.concat(mcind_xr_vec,dim='plasma_current_fraction').assign(min_current_frac=min_current_frac).assign_coords(MC_index=int(MCind)))
    return whole_dcon_xr_vec

#Untested
def compile_run_dir(runs_dir,run_dir_short,debug=False):
    run_dir=runs_dir+'/'+run_dir_short
    in_rundir=os.listdir(run_dir)
    #SHALLOW NAMES (ALL CORRESPONDING W EACH OTHER)
    xr_filenames=[i for i in in_rundir if "tokamak_xr_" in i]
    eqdsk_dirs=['eqdsks_b'+j[11:] for j in xr_filenames]
    dcon_dirs=['dcon_xrs_b'+j[11:] for j in xr_filenames]
    #LOOP FOR DCON DIRS
    batchxrs=[]
    for b_x in range(len(xr_filenames)):
        if debug:
            print(run_dir+'/'+xr_filenames[b_x])
            print(run_dir+'/'+dcon_dirs[b_x])
        if not os.path.isdir(run_dir+'/'+dcon_dirs[b_x]):
            continue
        tok_xr=xr.open_dataset(run_dir+'/'+xr_filenames[b_x]).drop_sel(point_type=['max_fusion','max_Q']).squeeze(dim='point_type')
        if not debug:
            try:
                tok_xr=prepare_neo_terms(tok_xr)
                dconxr_vec=compile_dconxrs(run_dir+'/'+dcon_dirs[b_x],tok_xr,debug=debug)
                if len(dconxr_vec)>0:
                    print('concatting dconxr_vec:::::::',run_dir+'/'+xr_filenames[b_x],dcon_dirs[b_x])
                    dconxr=xr.concat(dconxr_vec,dim='MC_index')
                    print('merging dconxr_vec:::::::')
                    tok_xr=exact_plasma_current_fraction(tok_xr)
                    dconxr=exact_plasma_current_fraction(dconxr)
                    comb_xr=xr.merge([tok_xr,dconxr])
                    print(comb_xr.plasma_current_fraction.values)
                    batchxrs.append(comb_xr)  #This is the step where I want to 
                    print('merging sucessful')
            except:
                print('merging failed on tokdir ',run_dir+'/'+xr_filenames[b_x],dcon_dirs[b_x])
                print('^^^FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                print('^^^FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
        else:
            tok_xr=prepare_neo_terms(tok_xr)
            dconxr_vec=compile_dconxrs(run_dir+'/'+dcon_dirs[b_x],tok_xr,debug=debug)
            if len(dconxr_vec)>0:
                print('concatting dconxr_vec:::::::',run_dir+'/'+xr_filenames[b_x],dcon_dirs[b_x])
                dconxr=xr.concat(dconxr_vec,dim='MC_index')
                print('merging dconxr_vec:::::::')
                tok_xr=exact_plasma_current_fraction(tok_xr)
                dconxr=exact_plasma_current_fraction(dconxr)
                comb_xr=xr.merge([tok_xr,dconxr])
                print(comb_xr.plasma_current_fraction.values)
                batchxrs.append(comb_xr)  #This is the step where I want to 
                print('merging sucessful')
    return batchxrs

def exact_plasma_current_fraction(dset):
    pfc_set=np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])
    pcf=np.zeros(len(dset.plasma_current_fraction.values))
    for i in range(len(dset.plasma_current_fraction.values)):
        pfc_set_diffs=np.abs(pfc_set-dset.plasma_current_fraction.values[i])
        pcf[i]=pfc_set[np.argmin(pfc_set_diffs)]
    dset=dset.assign(pcf_alt=dset['plasma_current_fraction']*0+pcf)
    dset=dset.set_index(plasma_current_fraction='pcf_alt')
    return dset

#Untested
#Complete
def prepare_neo_terms_local(tok_xr):
    #Dnc_mufrac, interpolate onto psifacs then multiply with Dnc_prefac to get Dnc
    zeff=tok_xr.z_effective.values[0]
    tok_xr=tok_xr.assign(KB_s00 = lambda tok_xr: (1.0+0.533/zeff))
    #tok_xr=tok_xr.assign(fc_GSoutput_on_linspace_psi_dynamo=1.0-tok_xr['ftr_GSoutput_on_linspace_psi_dynamo'])
    tok_xr=tok_xr.assign(Dnc_mufrac_on_linspace_psi_dynamo = lambda tok_xr: tok_xr.ftr_GSoutput_on_linspace_psi_dynamo*tok_xr.KB_s00/((1.0-tok_xr.ftr_GSoutput_on_linspace_psi_dynamo)+tok_xr.ftr_GSoutput_on_linspace_psi_dynamo*tok_xr.KB_s00))
    #This is KB_s00*f_tr/(f_c+KB_s00*f_tr)
    #Hbs, interpolate onto psifacs, multiply with mu0*Hbs_prefac to get Hbs
    tok_xr=tok_xr.assign(Jbs_GSinput_on_linspace_psi_dynamo=tok_xr.j_bootstrap)
    tok_xr=tok_xr.assign(Jtor_GSoutput_on_linspace_psi_dynamo=tok_xr.j_tor)
    return tok_xr

#Untested
#Loops through run dirs:
#   Merges and saves big batchvec of compile_results_  
def compile_results_(debug=True,compiled_xr_name='compiled_xr',just_precompiled=False, merge_as_go=False, return_big_batchvec=True, debug_int=5):
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[i for i in os.listdir(runs_dir) if "_ran" in i]
    #LOOP FOR RUN DIRS
    xrtot=None
    big_batchvec=[]
    failed_dirs=[]
    if debug:
        run_dirs=run_dirs[0:debug_int]
    case_merged=False
    macro_xr=None
    rund=0
    for run_dir_short in run_dirs:
        rund+=1
        print("Compiling from ",run_dir_short, f" ({rund} of {len(run_dirs)})\n\n\n")
        if os.path.isfile(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name) and (not debug):
            print("  - reading pre-existing compiled xr in ",run_dir_short)
            tok_xr=xr.open_dataset(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
            if return_big_batchvec:
                big_batchvec.append(tok_xr)
            if merge_as_go:
                if not case_merged:
                    macro_xr=tok_xr
                    case_merged=True
                else:
                    macro_xr=xr.merge([macro_xr,tok_xr])
        elif not just_precompiled:
            if debug:
                print("debugging")
                tot_xr_vec=compile_run_dir(runs_dir,run_dir_short,debug=debug)
                if len(tot_xr_vec)>0:
                    tok_xr=xr.merge(tot_xr_vec)
                    tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
                    tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
                    tok_xr.to_netcdf(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
                    print("  - saving compiled xr for ",run_dir_short)
                    if return_big_batchvec:
                        big_batchvec.append(tok_xr)
            else:
                try:
                    tot_xr_vec=compile_run_dir(runs_dir,run_dir_short)
                    if len(tot_xr_vec)>0:
                        try:
                            tok_xr=xr.merge(tot_xr_vec)
                            tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
                            tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
                            tok_xr.to_netcdf(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
                            print("  - saving compiled xr for ",run_dir_short)
                            if return_big_batchvec:
                                big_batchvec.append(tok_xr)
                            continue
                        except:
                            assert  len(tot_xr_vec)>0 
                            excluded_inds=[]
                            tot_xr_vecCOPY=copy.deepcopy(tot_xr_vec)
                            iii=0
                            while iii <= len(tot_xr_vecCOPY):
                                try:
                                    xr.merge(tot_xr_vecCOPY[0:iii])
                                    iii+=1
                                except:
                                    if iii<=1:
                                        print("Problem is in first two!")
                                        if (len(tot_xr_vec)-1)>0: #Can delete one keep trying
                                            try: #Testing removing first element
                                                tok_xr=xr.merge(tot_xr_vecCOPY[1:])
                                            except: #Testing removing first two elements
                                                if (len(tot_xr_vec)-2)>0: #Can delete two and keep trying
                                                    try: 
                                                        tok_xr=xr.merge(tot_xr_vecCOPY[2:])
                                                    except:
                                                        print("It was in first tok_xrs but didn't work when they were removed")
                                                        print(" could be MORE THAN ONE FAIL PRESENT")
                                                        print('^^FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                                                        print('^^FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                                                        break
                                                else:
                                                    print("It was in first tok_xr but only two tok_xrs are present!")
                                                    print('^^FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                                                    break
                                            tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
                                            tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
                                            tok_xr.to_netcdf(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
                                            print("  - saving compiled xr for ",run_dir_short)
                                            break
                                        else:
                                            print("It was in first tok_xr but one tok_xr is present!")
                                            print('^^FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL:::::::::::::::::::::::::FAILFAILFAIL')
                                            break
                                    else:
                                        excluded_inds.append(iii)
                                        del tot_xr_vecCOPY[iii-1]
                                        try:
                                            tok_xr=xr.merge(tot_xr_vecCOPY)
                                            tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
                                            tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
                                            tok_xr.to_netcdf(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
                                            print("  - saving compiled xr for ",run_dir_short)
                                            break
                                        except:
                                            print("MORE THAN ONE FAIL PRESENT")
                except:
                    print("Skipping ",run_dir_short," :: couldn't even get through compile_run_dir.")
                    failed_dirs.append(run_dir_short)
                #tok_xr=tot_xr.assign_coords(tokID=[run_dir_short+'_MCi'+str(i) for i in tot_xr["MC_index"].values],dims='MC_index').set_index(tokID='MC_index')
                #).set_index(tokID=MC_index).reset_coords('MC_index',drop=True)

    print(failed_dirs)
    return big_batchvec,failed_dirs,macro_xr

#run_dir_short='v1001_run1__proc22'
#tok_xr=big_batchvec[1]
#tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
#tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
#test1=test1.assign_coords(tokID=run_dir_short+'_MCi'+10000+test1["MC_index"].values).drop_indexes('tokID').reset_coords('dirID').set_index(tokID='MC_index')
#).set_index(tokID=MC_index).reset_coords('MC_index',drop=True)

#Untested
#One step function:
#   Merges and saves big batchvec of compile_results_
def compile_results(filename='surf_xr1',just_precompiled=False,debug=True,merge_as_go=False,compiled_xr_name='compiled_surf_xr1',debug_int=5,return_big_batchvec=True):
    #if len(filename)>0:
    #    debug=False
    big_batchvec,failed_dirs,macro_xr=compile_results_(debug_int=debug_int,debug=debug,merge_as_go=merge_as_go,compiled_xr_name=compiled_xr_name,return_big_batchvec=return_big_batchvec,just_precompiled=just_precompiled)
    if len(big_batchvec)>0 or merge_as_go:
        try:
            if not merge_as_go:
                print("Merging big batch:")
                xrtot = xr.merge(big_batchvec) #.stack(tokID=("dirID","MC_index"))
            else:
                xrtot=macro_xr
            if len(filename)>0:
                os.chdir('/home/sbenjamin/TearingMacroFolder/results')
                xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename)
            return xrtot,big_batchvec,failed_dirs
        except:
            return None,big_batchvec,failed_dirs
    return

def compile_resultsb(array_id,filename='surf_xr1',merge_as_go=True,compiled_xr_name='compiled_surf_xr1'):
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)

    run_dirs=[i for i in os.listdir(runs_dir) if "_ran" in i]
    run_dirs=run_dirs[(array_id-1)*67:array_id*67]
    filename=filename+'_batch_'+str(array_id)+'_of_8'

    if merge_as_go:
        print("Merging as we go.")
    
    big_batchvec=[]
    rund=0
    case_merged=False
    for run_dir_short in run_dirs:
        rund+=1
        print("Compiling from ",run_dir_short, f" ({rund} of {len(run_dirs)})\n\n\n")
        if os.path.isfile(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name):
            print("  - reading pre-existing compiled xr in ",run_dir_short)
            tok_xr=xr.open_dataset(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
            big_batchvec.append(tok_xr)
            if merge_as_go:
                if not case_merged:
                    macro_xr=tok_xr
                    case_merged=True
                else:
                    macro_xr=xr.merge([macro_xr,tok_xr])

    if len(big_batchvec)>0:
        print("Merging big batch of ",len(big_batchvec)," xarrays.")
        if not merge_as_go:
            xrtot = xr.merge(big_batchvec)
        else:
            xrtot=macro_xr
        print("Saving big batch.")
        if len(filename)>0:
            os.chdir('/home/sbenjamin/TearingMacroFolder/results')
            xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename)

    return xrtot,big_batchvec

def compile_resultsc(filename='surf_xr1',compiled_xr_name='compiled_surf_xr1'):
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)

    run_dirs=[i for i in os.listdir(runs_dir) if "_ran" in i]

    print("Merging as we go.")
    oneqrt=len(run_dirs)//4
    
    big_batchvec=[]
    rund=0
    case_merged=False
    for run_dir_short in run_dirs:
        rund+=1
        print("Compiling from ",run_dir_short, f" ({rund} of {len(run_dirs)})\n\n\n")
        if os.path.isfile(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name):
            print("  - reading pre-existing compiled xr in ",run_dir_short)
            tok_xr=xr.open_dataset(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
            big_batchvec.append(tok_xr)
            if not case_merged:
                macro_xr=tok_xr
                case_merged=True
            else:
                macro_xr=xr.merge([macro_xr,tok_xr])

        if rund==5:
            print("First Test Print")
            xrtot=macro_xr
            xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename+"_test_print_1")
        if rund==10:
            print("Second Test Print")
            xrtot=macro_xr
            xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename+"_test_print_2")
        if rund==oneqrt:
            xrtot=macro_xr
            xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename+"_one_qrtr")
        if rund==2*oneqrt:
            xrtot=macro_xr
            xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename+"_two_qrtr")
        if rund==3*oneqrt:
            xrtot=macro_xr
            xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename+"_three_qrtr")

    xrtot=macro_xr
    xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename+"_four_qrtr")

    return 

#exec(open("/home/sbenjamin/TearingMacroFolder/local_scripts/compilation.py").read())

#Big batch compile:
#xrtot,big_batchvec,failed_dirs=compile_results(filename='proper_compile_macro_xr1',compiled_xr_name='proper_compile_xr1',just_precompiled=False,debug=False,merge_as_go=False,debug_int=5,return_big_batchvec=False)
#xrtot,big_batchvec,failed_dirs=compile_results(filename='proper_compile_macro_xr1_fixedDh',compiled_xr_name='proper_compile_xr1_fixedDh',just_precompiled=False,debug=False,merge_as_go=False,debug_int=5,return_big_batchvec=False)
#xrtot,big_batchvec,failed_dirs=compile_results(filename='proper_compile_macro_xr3',compiled_xr_name='proper_compile_xr3',just_precompiled=False,debug=False,merge_as_go=False,debug_int=2,return_big_batchvec=False)
#xrtot,big_batchvec,failed_dirs=compile_results(filename='proper_compile_macro_xr4',compiled_xr_name='proper_compile_xr4',just_precompiled=False,debug=False,merge_as_go=False,debug_int=2,return_big_batchvec=False)
#compile_resultsc(filename='proper_compile_macro_xr1_fixedDh',compiled_xr_name='proper_compile_xr1_fixedDh')
#compile_resultsc(filename='proper_compile_macro_xr',compiled_xr_name='proper_compile_xr3')
#compile_resultsc(filename='proper_compile_macro_xr4',compiled_xr_name='proper_compile_xr4')

#BBBBB
#array_id=int(sys.argv[1])
#compile_resultsb(array_id,filename='proper_compile_macro_xr1',compiled_xr_name='proper_compile_xr1')
#print("Reading in big batch 1:" + 'proper_compile_macro_xr1_batch_'+str(((array_id-1)*2+1))+'_of_8')
#toxr1=xr.open_dataset('/home/sbenjamin/TearingMacroFolder/results/'+'proper_compile_macro_xr1_batch_'+str(((array_id-1)*2+1))+'_of_8')
#print("Reading in big batch 2:" + 'proper_compile_macro_xr1_batch_'+str(((array_id-1)*2+2))+'_of_8')
#toxr2=xr.open_dataset('/home/sbenjamin/TearingMacroFolder/results/'+'proper_compile_macro_xr1_batch_'+str(((array_id-1)*2+2))+'_of_8')
#print("Merging")
#xrtot = xr.merge([toxr1,toxr2])
#print("Saving big batch:"+ 'proper_compile_macro_xr1_batch_B'+str(array_id)+'_of_4')
#os.chdir('/home/sbenjamin/TearingMacroFolder/results')
#xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+'proper_compile_macro_xr1_batch_B'+str(array_id)+'_of_4')

#xrtot,big_batchvec,failed_dirs=compile_results(filename='',debug=True,compiled_xr_name='margTest1')
#xrtot,big_batchvec,failed_dirs=compile_results(filename='',debug=True,compiled_xr_name='gooftest')
#

#xrtot,big_batchvec,failed_dirs=compile_results(filename='',debug=True,debug_int=2,merge_as_go=False,just_precompiled=False,compiled_xr_name='compiled_surf_ran_xrT2',return_big_batchvec=False)
"""
time.sleep(30*60)
xrtot,big_batchvec,failed_dirs=compile_results(filename='marg_xr1_miniprint2',debug=False,merge_as_go=True,just_precompiled=True,compiled_xr_name='compiled_marg_xr1')
time.sleep(30*60)
num_compileds=count_prog()
xrtot,big_batchvec,failed_dirs=compile_results(filename='marg_xr1_miniprint3',debug=False,merge_as_go=True,just_precompiled=True,compiled_xr_name='compiled_marg_xr1')
if num_compileds<300:
    num_compileds2=count_prog()
    if num_compileds2 > 300:
        xrtot,big_batchvec,failed_dirs=compile_results(filename='marg_xr1_bigprint',debug=False,merge_as_go=True,just_precompiled=True,compiled_xr_name='compiled_marg_xr1')
    else:
        xrtot,big_batchvec,failed_dirs=compile_results(filename='marg_xr1_miniprint4',debug=False,merge_as_go=True,just_precompiled=True,compiled_xr_name='compiled_marg_xr1')
xrtot,big_batchvec,failed_dirs=compile_results(filename='marg_xr1_bigprint',debug=False,merge_as_go=True,just_precompiled=True,compiled_xr_name='compiled_marg_xr1')
"""
#xrtot,big_batchvec,failed_dirs=compile_results(filename='marg_xr1',debug=False,merge_as_go=False,just_precompiled=False,compiled_xr_name='compiled_marg_xr1')

#marg_stable.sel(m=2,n=1).Re_DeltaPrime.count()
#marg_stable=marg_mp4.where(marg_mp4.dcon_nzero==0)
#marg_stable=marg_mp4.where(marg_mp4.dcon_nzero==0)
#marg_stable2=marg_stable.where(marg_stable.dW_total>0)
#marg_stable3=marg_stable2.dropna(dim='ns_rdcon_ran')
#marg_stable4=marg_stable3.dropna(dim='ns_dcon_ran')

#what I want:
    #throw out cases of (MC_index,plasma_current_fraction) if... 
    #   not (all dW_totals are real and positive)
    #   not (all dcon_nzeros exist and are 0)
    #       dcon_nzeros, dW_totals have coordidnate ns_dcon_ran


#make a variable: all_dWtotals positive = min


"""
marg_mp4.where(marg_mp4.dcon_nzero==0)

marg_mp4.where(marg_mp4.dcon_nzero==0)

marg_mp42=marg_mp4.copy()
marg_mp42.assign(no_nan_dWs )


marg_stable=marg_mp4.where(marg_mp4.dW_total>0.0,drop=True)
marg_R_cases=marg_mp4.where(marg_mp4.min_marg_wbarR>0.1,drop=True)
marg_cases=marg_mp4.where(marg_mp4.min_marg_wbar>0.1,drop=True)

marg_stable


#^these working

marg_cases_no_nonsense=marg_R_cases.where(marg_R_cases.>0.1)
"""

import os
def count_prog(compiled_xr_name='proper_compile_xr4'):
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[i for i in os.listdir(runs_dir) if "_ran" in i]
    print(run_dirs)
    num_compileds=0
    for run_dir_short in run_dirs:
        if os.path.isfile(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name):
            num_compileds+=1
        else:
            print(runs_dir+'/'+run_dir_short+"/"+compiled_xr_name)
    print(num_compileds," out of ",len(run_dirs)," compiled.")
    return num_compileds

#xrtot,big_batchvec,failed_dirs=compile_results()
#def buildDnc(xrtot):
    #xrtot=xrtot.drop

    #numpy.linspace(1.E-4,1.0-(1.E-4),len(xrtot['ftr_GSoutput_on_linspace_psi_dynamo'].dim_rho),dtype=numpy.float64)



#WORKS TO GET Dnc_mufrac_on_linspace_psi_dynamo:
#xrtot=xrtot.assign(KB_s00 = lambda xrtot: (1.0+0.533/xrtot.z_effective))
#xrtot=xrtot.assign(linspace_psi_dynamo = xrtot['ftr_GSoutput_on_linspace_psi_dynamo']*0.0+np.linspace(1.E-4,1.0-(1.E-4),len(xrtot['ftr_GSoutput_on_linspace_psi_dynamo'].dim_rho),dtype=np.float64))
#xrtot=xrtot.assign(fc_GSoutput_on_linspace_psi_dynamo=1.0-xrtot['ftr_GSoutput_on_linspace_psi_dynamo'])
#xrtot=xrtot.assign(Dnc_mufrac_on_linspace_psi_dynamo = lambda xrtot: xrtot.ftr_GSoutput_on_linspace_psi_dynamo*xrtot.KB_s00/(xrtot.fc_GSoutput_on_linspace_psi_dynamo+xrtot.ftr_GSoutput_on_linspace_psi_dynamo*xrtot.KB_s00))

#FOR A SINGLE POINT, CAN WE GET SOMETHING NEW:
#xrtot[dict(MC_index=2,plasma_current_fraction=13)]
#xrtot=xrtot.assign(Dnc_mufrac_on_psifacs=0.0*xrtot['psifacs'])

#xrtot[dict(MC_index=2)].Dnc_mufrac_on_psifacs

#xrtot[dict(MC_index=2,plasma_current_fraction=13)]['Dnc_mufrac_on_psifacs'] = 3.0

#.Dnc_mufrac_on_psifacs[plasma_current_fraction=16]
#xrtot[dict(MC_index=2)].Dnc_mufrac_on_linspace_psi_dynamo

#Dnc_mufrac_on_linspace_psi_dynamo_xr=xrtot.Dnc_mufrac_on_linspace_psi_dynamo.reset_coords('psi_normalised',drop=True)
#psifacs=xrtot.psifacs.drop_indexes('ipsis').reset_coords('ipsis',drop=True)
#Dnc_mufrac_on_psifacs=Dnc_mufrac_on_linspace_psi_dynamo_xr.interp(dim_rho=psifacs)
#len_dim_rho=len(xrtot['ftr_GSoutput_on_linspace_psi_dynamo'].dim_rho)
#xrtot.inter(x=)
#xrtot=xrtot.assign(Dnc_on_psifacs = 0.0*xrtot['psifacs']+xrtot['Dnc_mufrac_on_linspace_psi_dynamo'].interp(dim_rho=xrtot['psifacs']*(len_dim_rho-1)))
#xrtot['Dnc_prefacs']*
#xrtot.assign(Dnc_mufrac = lambda xrtot: xrtot.Dnc_prefacs*(xrtot.
#dnc_mufracs=
#xrtot.assign(dnc_mufrac=

"""          
DCON surface values:
"ipsis,psifacs,fs,mu0_ps,dvdpsis,qs,",
"Dis,Drs,Hs,Dnc_prefacs,Wc_prefacs,",
"Hbs_prefacs,taua_prefacs,taur_prefacs"


Recipe to complete:
    Dnc_prefacs:
        mu_e needs collis. frac term, equ B14 from https://cptc.wisc.edu/wp-content/uploads/sites/327/2017/09/UW-CPTC_09-6_rev-1.pdf
            tau_ss given in equ A3
            mu_s given above equ 5, also A9
            K parameters in TABLE I
            Use banana regime
                ? check dimensions (no tau_ee?)
                - Z, Zeff to built K^B_s00
                - trapped fraction/passing fraction

    Dh: 
        ? alpha_s big or small?

    taua: 
        - local mass density line 128 mercier.f
            ? rho check units
            ? forgot charge of ion?
        - toroidal mode number n

    taur: 
        - local resistivity, line 128 mercier.f
            - ne
            - te
            - me, mi

    Hbs:
        - mu0 prefactor 
        - trapped fraction
        - complicated function of H
        - X0=(taur/taua)^(-1/3)

    Wc_prefacs:
        - needs perp/parallel ratio, see rosenbweg/Fitz 1994
            - use energy_confinement_time (straight from popcons) for perp
            - need perp. wavelength got parallel
                - use n*s_s*W(m)/R0 = n*s_s*sqrt(w_psinorm)*(sqrt(kappa)*<a>)/R0
                                    = n*2*(psi)*(q1/qs)*sqrt(w_psinorm)*(sqrt(kappa)*<a>)/R0
                    - s_s = r(dq/dr)/qs = 2 psi q1/qs (check...)
            - wd in 
        - poloidal mode number m

    wsat
        - Dh
        - k0
        - k1
        - Dnc
        - Wc

"""
#def useful_operations(xrtot,filename='',debug=True):




"""
def define_min_point(xrtot_stacked):
    xrtot_stacked.sel(m=2,n=1)

def analyse_deltaPrime(xrtot_stacked;m=2,n=1):
xrtot_mn=xrtot_stacked.sel(m=m,n=n)
num_toks=len(xrtot_mn)
assert len(xrtot_mn)==len(xrtot_mn.tokID)
num_plasma_current_fracs=len(xrtot_mn.plasma_current_fraction)
assert xrtot_mn.plasma_current_fraction[0]<xrtot_mn.plasma_current_fraction[-1]

min_current_int=np.ones(num_toks, dtype=int)
min_current_int=min_current_int*num_plasma_current_fracs-1
xrtot_mn=xrtot_mn.assign(min_current_int=min_current_int)

for i in range(num_toks):
    for j in range(num_plasma_current_fracs):
        if not np.isnan(xrtot_mn.Re_DeltaPrime.isel(tokID=i,plasma_current_fraction=j)):
            xrtot_mn['xrtot_mn'][i]=j
            break

xrtot_mn=xrtot_mn.assign(min_current_int=min_current_int).set_index(min_current_int='tokID')

    return 

def vals_of_interest(xtot):
    stabinf=xtot.drop_coords 

#What do I want:
    #2/1, 3/2, 4/3, 5/4 Delta primes on 4D space

#2/1 vs []

#Two fluff variables:
#   MC_index and dirID

#Neo terms on modes deprecated
def neo_terms_on_modes1_dep(dconxr):
    #Already got Dh, Dh_alt
    #Just need Dnc, Hbs, Drbs
    print(dconxr.psifacs.values)
    print(dconxr.Dncs.values)
    assert len(dconxr.psifacs.values)==len(dconxr.Dncs.values)

    Dnc_spline=CubicSpline(dconxr.psifacs.values,dconxr.Dncs.values)
    Hbs_spline=CubicSpline(dconxr.psifacs.values,dconxr.Hbss.values)
    Drbs_spline=CubicSpline(dconxr.psifacs.values,dconxr.Drbss.values)

    #dnc_on_nodes=Dnc_spline(dconxr['psifac'].values)[0]

    #dnc_on_nodes=dconxr.map(Dnc_spline)

    print(Dnc_spline(0.12345))

    #dconxr=dconxr.assign(Dnc = lambda dconxr: Dnc_spline(dconxr.psifac.values))


    dconxr=dconxr.assign(Dnc =  0.0*dconxr['psifac'])#+dnc_on_nodes)

    for n in dconxr.psifac.n.values:
        for m in dconxr.psifac.m.values:
            cheeckval=dconxr.psifac.sel(m=m,n=n).values
            print("cheeckval = ",cheeckval)
            Dncval=Dnc_spline(0.786)
            print(Dncval)
            print("debuggle")
            raise
            print(dconxr['Dnc'].loc[dict(m=m, n=n)])
            dconxr['Dnc'].loc[dict(m=m, n=n)]=Dncval#Dnc_spline(dconxr.psifac.sel(m=m,n=n).values)

    print("debuggle")
    raise
    #psifac_ds=dconxr.get('psifac')
    #dnc_da=dconxr.Dnc
    #print(psifac_ds)
    #print("debuggle")
    #raise
    psifac_ds=xr.Dataset({"psifac": dconxr.psifac})
    print(psifac_ds)
    psifac_ds.map(lambda x: Dnc_spline(x))
    print(psifac_ds)
    #dnc_da=Dnc_spline(psifac_da)
    print("debuggle")
    raise

    print(dnc_da)
    print(dconxr['Dnc'])

    dconxr['Dnc'] = dnc_da

    #Dnc_vals = 
    #print(dconxr.Dnc.m)
    #print(dconxr.Dnc.n)


    dconxr=dconxr.assign(DrbsH = lambda dconxr: dconxr.Dnc+dconxr.Dh)
    dconxr=dconxr.assign(DrbsH_alt = lambda dconxr: dconxr.Dnc+dconxr.Dh_alt)
    dconxr=dconxr.assign(Hbs = lambda dconxr: Hbs_spline(dconxr.psifac))
    dconxr=dconxr.assign(Drbs = lambda dconxr: Drbs_spline(dconxr.psifac.values))

    return dconxr

"""
#Strat:
#   Gather MCi, Jfrac from name
#   Assign each dconxr a current frac, concat along that first
#   Assign each concatted xr an MCi, concat along that
#   Merge w tokamaker/cfspopcon thingo? For that batch?
#   Finally combine across different batches to get total Mci for that outfile
#   Combining across outfiles... idk how to do that
#   To create unique labels, replace each MCi with a filecode (directory+MCi)
#   Then do what...

#Useful things to check:
#   xrtot.sel(m=2,n=1).Re_DeltaPrime.count() [basically ideal MHD stable nn=1 cases]
#   xrtot.sel(m=2,n=1).P_fusion.count()   [total cases]

#First:
    #Can we just merge the two xarrays....
    #One 

    #Drop all but min pressure? No
    #   b1_xr=b1_tokxr.drop_sel(point_type=['max_fusion','max_Q'])
