import os
import numpy as np
import xarray as xr
from OpenFUSIONToolkit.util import mu0
from scipy.interpolate import CubicSpline

#Untested
def compile_dconxrs(dcon_dir,tok_xr):
    dcon_unique_mcinds=[]
    dcon_xrs=[]
    dcon_file_list=[i for i in os.listdir(dcon_dir) if not ('xrF' in i)]
    #LOOP FOR DCON XARRAYS 
    #   GATHERING INFO AND XRs
    for dcon_file in dcon_file_list:
        dcon_filespl=dcon_file.split('_')
        dcon_mcind=dcon_filespl[2][3:]
        dcon_Jfrac=float(dcon_filespl[3][5:].replace('p','.'))
        if not(dcon_mcind in dcon_unique_mcinds):
            dcon_unique_mcinds.append(dcon_mcind)
        dcon_xr=xr.open_dataset(dcon_dir+'/'+dcon_file).assign_coords(plasma_current_fraction=dcon_Jfrac)

        dcon_xr=neo_terms_on_psifacs(dcon_xr,int(dcon_mcind),tok_xr)

        dcon_xrs.append([dcon_xr,dcon_mcind])
    #Constuction of whole DCON xr from double-loop
    whole_dcon_xr_vec=[]
    for MCind in dcon_unique_mcinds:
        mcind_xr_vec=[]
        for dxr in dcon_xrs:
            if dxr[1]==MCind:
                mcind_xr_vec.append(dxr[0])
        whole_dcon_xr_vec.append(xr.concat(mcind_xr_vec,dim='plasma_current_fraction').assign_coords(MC_index=int(MCind)))
    return whole_dcon_xr_vec

#Untested
def neo_terms_on_psifacs(dconxr,dcon_mcind,tok_xr):
    local_tok_xr=tok_xr.sel(MC_index=dcon_mcind,plasma_current_fraction=dconxr_vec.plasma_current_fraction.values)

    interp_dim_rho_vals=dconxr.psifacs.values*(len(local_tok_xr.dim_rho.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)
    
    Dnc_mufrac_on_psifacs=local_tok_xr.Dnc_mufrac_on_linspace_psi_dynamo.interp(dim_rho=interp_dim_rho_vals, method="cubic")
    avg_BJbs_on_psifacs=local_tok_xr.avg_BJbs_on_linspace_psi_dynamo.interp(dim_rho=interp_dim_rho_vals, method="cubic")

    dconxr=dconxr.assign(Dncs=dconxr['Dnc_prefacs']*Dnc_mufrac_on_psifacs)
    dconxr=dconxr.assign(Hbss=dconxr['Hbs_prefacs']*avg_BJbs_on_psifacs*mu0)
    dconxr=dconxr.assign(Drbss=dconxr['Drs']+dconxr['Hbs']*(dconxr['Hs']-0.5))
    dconxr=dconxr.assign(Dhs=dconxr['Drs']/(-0.5+np.sqrt(-dconxr['Dis'])-dconxr['Hs']))
    dconxr=dconxr.assign(Dh_alts=dconxr['Drs']/(0.5+np.sqrt(-dconxr['Dis'])-dconxr['Hs']))

    ni_on_psifacs=
    ne_on_psifacs=
    Ti_on_psifacs=
    Te_on_psifacs=
    mi
    #rho = (ne*(2.5*1.672614e-27)) kg/m^3
    #lambda=24-.5*LOG(ne)+LOG(te)
    #taue=3.44e5*te**1.5/(ne*lambda)
    #ne*mi*1e6
    """
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
    """
    #Wc =

    local_tok_xr.

    taua_no_n=
    taur=
    sfac=

    dconxr=neo_terms_on_modes(dconxr)
    #dconxr=dconxr.assign(N1=dconxr['Drs']+dconxr['Hbs']*(dconxr['Hs']-0.5))

    #for i in range(len(dconxr_vec)):
    #    MC_ind_dcon=dconxr_vec[i].MC_index.values
    #    for DCONJfrac in dconxr_vec[i].plasma_current_fraction.values:
    #        local_tok_xr=tok_xr.sel(MC_index=MC_ind_dcon,plasma_current_fraction=DCONJfrac)
    #
    #        interp_dim_rho_vals=dconxr_vec[i].psifacs.values*(len(local_tok_xr.dim_rho.values)-1) #this is off by 2E-4 relative to psi=0,...,1 (see definition of Tokamaker_get_Jtorr_and_f_tr in random_start.py)
    #        Dnc_mufrac_on_psifacs=local_tok_xr.Dnc_mufrac_on_linspace_psi_dynamo.interp(dim_rho=interp_dim_rho_vals, method="cubic")
    return dconxr

#Untested
def neo_terms_on_modes(dconxr):
    Dnc_spln=CubicSpline(dconxr.psifacs.values,dconxr.Dncs.values)
    Hbss_spln=CubicSpline(dconxr.psifacs.values,dconxr.Hbss.values)

    dconxr=dconxr.assign(Dnc = lambda dconxr: Dnc_spln(dconxr.psifac.values))
    dconxr=dconxr.assign(Hbs = lambda dconxr: Hbss_spln(dconxr.psifac.values))




#Untested
def compile_run_dir(runs_dir,run_dir_short):
    run_dir=runs_dir+'/'+run_dir_short
    in_rundir=os.listdir(run_dir)
    #SHALLOW NAMES (ALL CORRESPONDING W EACH OTHER)
    xr_filenames=[i for i in in_rundir if "tokamak_xr_" in i]
    eqdsk_dirs=['eqdsks_b'+j[11:] for j in xr_filenames]
    dcon_dirs=['dcon_xrs_b'+j[11:] for j in xr_filenames]
    #LOOP FOR DCON DIRS
    batchxrs=[]
    for b_x in range(len(xr_filenames)):
        if not os.path.isdir(run_dir+'/'+dcon_dirs[b_x]):
            continue
        tok_xr=xr.open_dataset(run_dir+'/'+xr_filenames[b_x]).drop_sel(point_type=['max_fusion','max_Q']).squeeze(dim='point_type')
        tok_xr=prepare_neo_terms(tok_xr)
        dconxr_vec=compile_dconxrs(run_dir+'/'+dcon_dirs[b_x],tok_xr)
        if len(dconxr_vec)>0:
            dconxr_vec=neo_terms_on_psifacs(dconxr_vec,tok_xr)
            dconxr=xr.concat(dconxr_vec,dim='MC_index')
            batchxrs.append(xr.merge([tok_xr,dconxr]))  #This is the step where I want to 
    return batchxrs

#Untested
def prepare_neo_terms(tok_xr):
    #tok_xr=tok_xr.assign(linspace_psi_dynamo = tok_xr['ftr_GSoutput_on_linspace_psi_dynamo']*0.0+np.linspace(1.E-4,1.0-(1.E-4),len(tok_xr['ftr_GSoutput_on_linspace_psi_dynamo'].dim_rho),dtype=np.float64))

    #Dnc_mufrac, interpolate onto psifacs then multiply with Dnc_prefac to get Dnc
    tok_xr=tok_xr.assign(KB_s00 = lambda tok_xr: (1.0+0.533/tok_xr.z_effective))
    tok_xr=tok_xr.assign(fc_GSoutput_on_linspace_psi_dynamo=1.0-tok_xr['ftr_GSoutput_on_linspace_psi_dynamo'])
    tok_xr=tok_xr.assign(Dnc_mufrac_on_linspace_psi_dynamo = lambda tok_xr: tok_xr.ftr_GSoutput_on_linspace_psi_dynamo*tok_xr.KB_s00/(tok_xr.fc_GSoutput_on_linspace_psi_dynamo+tok_xr.ftr_GSoutput_on_linspace_psi_dynamo*tok_xr.KB_s00))
    tok_xr
    
    #Hbs, interpolate onto psifacs, multiply with mu0*Hbs_prefac to get Hbs
    #tok_xr.avg_BJbs_on_linspace_psi_dynamo

    return tok_xr

#Untested
def compile_results_(debug=True):
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[i for i in os.listdir(runs_dir) if "_run" in i]
    #LOOP FOR RUN DIRS
    xrtot=None
    big_batchvec=[]
    failed_dirs=[]
    if debug:
        run_dirs=run_dirs[0:5]
    for run_dir_short in run_dirs:
        print("Compiling from ",run_dir_short)
    try:
        tot_xr_vec=compile_run_dir(runs_dir,run_dir_short)
        if len(tot_xr_vec)>0:
            tok_xr=xr.merge(tot_xr_vec)
            tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
            tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
            big_batchvec.append(tok_xr)
    except:
        print("Skipping ",run_dir_short," after encountering an error.")
        failed_dirs.append(run_dir_short)
        #tok_xr=tot_xr.assign_coords(tokID=[run_dir_short+'_MCi'+str(i) for i in tot_xr["MC_index"].values],dims='MC_index').set_index(tokID='MC_index')
        #).set_index(tokID=MC_index).reset_coords('MC_index',drop=True)
    return big_batchvec,failed_dirs

#run_dir_short='v1001_run1__proc22'
#tok_xr=big_batchvec[1]
#tok_xr=tok_xr.assign_coords(MCindstr=tok_xr["MC_index"].astype(str)).set_index(MC_index='MCindstr')
#tok_xr=tok_xr.assign_coords(MCindstr=run_dir_short+'_MCi'+tok_xr["MC_index"]).set_index(MC_index='MCindstr')
#test1=test1.assign_coords(tokID=run_dir_short+'_MCi'+10000+test1["MC_index"].values).drop_indexes('tokID').reset_coords('dirID').set_index(tokID='MC_index')
#).set_index(tokID=MC_index).reset_coords('MC_index',drop=True)

#Untested
def compile_results(filename='',debug=True):
    if len(filename)>0:
        debug=False
    big_batchvec,failed_dirs=compile_results_(debug=debug)
    if len(big_batchvec)>0:
        try:    
            xrtot = xr.merge(big_batchvec) #.stack(tokID=("dirID","MC_index"))
            if len(filename)>0:
                os.chdir('/home/sbenjamin/TearingMacroFolder/results')
                xrtot.to_netcdf('/home/sbenjamin/TearingMacroFolder/results/'+filename)
            return xrtot,big_batchvec,failed_dirs
        except:
            return None,big_batchvec,failed_dirs
    return 

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
                - use n*s_s*W(m)/R0 = n*s_s*sqrt(w_psi,norm)*<a>/R0
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
