#%reload_ext autoreload
#%autoreload 2

import random
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import h5py

#import run_local_RDCON
#from run_local_RDCON import run_DCON_on_equilibrium2,write_equil_in,write_rdcon_inputs,write_dcon_inputs
exec(open("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/run_local_RDCON.py").read())
from scipy.interpolate import CubicSpline, make_interp_spline
import os
import errno
import sys
import math
import shutil
import time
import inspect
import matplotlib.pyplot as plt
from omfit_classes.utils_fusion import sauter_bootstrap, Hmode_profiles
from pathlib import Path

rootpath=tokamaker_python_path
sys.path.append(os.path.join(rootpath,'python'))
import OpenFUSIONToolkit
from OpenFUSIONToolkit.TokaMaker import TokaMaker, solve_with_bootstrap,basic_dynamo_w_bootstrap
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, read_eqdsk, eval_green, create_power_flux_fun
from OpenFUSIONToolkit.util import mu0, eC

plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
#%matplotlib inline
#%config InlineBackend.figure_format = "retina"

#cfs_python_path = os.getenv('/Users/sbenjamin/Desktop/TEARING_WIDGET/cfspopcon')
import cfspopcon
from cfspopcon.unit_handling import ureg

# Change to the top-level directory. Required to find radas_dir in its default location.
#%cd {Path(cfspopcon.__file__).parents[1]}
# As a sanity check, print out the current working directory
#print(f"Running in {Path('').absolute()}")

mu_0 = 1.25663706127e-6
elongation_ratio_areal_to_psi95=1.025

#workflow:
#create input.yaml, put it in a starting directory
#create function that makes list of random inputs
#create two checks on my random inputs
#create function that takes list of confirmed random inputs, 
#   calls read_case (updating inputs)
#   runs popcon
#   finds key point(s)
#   filters cases (eliminating failed ones with .count() on mask)
#   combines cases (via dim=monte-carlo dimension, shot-type, Ifrac(?))
#   calculates FF basic ()how to add 1D data...
#   saves the whole new xarray

#density pedestal heigh is fine:
    #https://iopscience.iop.org/article/10.1088/0029-5515/48/7/075005/pdf
    #https://arxiv.org/pdf/2404.17040
    #https://www-pub.iaea.org/mtcd/publications/pdf/csp_008c/pdf/iterp_03.pdf
    #https://iopscience.iop.org/article/10.1088/0741-3335/42/5A/302
    #Fig 9 from this: https://w3.pppl.gov/~hammett/gyrofluid/papers/2002/PHP05018.pdf

#Pablo explanation of pedestal heights & his profiles:
    #Basically there's no physics behind the temp pedestal height
    #It's backsolved to get the correct <Te> AFTER a/Lt, core peaking have been worked out...
        #Going forward, we will apply PRF's eped neural net. This is our first pass
        #In mean time, look for historically accurate heights!!! Can papers suggest its legit?
    #a/LT choose 1.5 if sad, 2.0 if average...

#TESTED UNDER RIGHT CONDITIONS...
#Point must be of type min_pressure...
#setting rescale_ind_Jtor_by_ftr to False since even the largest tokamaks are q0~0.6 with rescale_ind_Jtor_by_ftr=False, which is already
# unrealistic in the tokamak (sawtoothing will prevent this) situation...
def run_tokamaker_case_on_point_macro(point, mygs, current_backdown, tot_num_cases,out_path,batch_num,total_batches,nr,nz,
                                      smooth_dynamo_inputs=True, high_def_plasma_dx=0.002, max_high_def_iterations=6,rescale_ind_Jtor_by_ftr=False,
                                      save_equil=False, run_phys_suite=True, verbose=False,lcfs_pad=1e-6,ttol=1e-14, plot_iteration_internal=False, 
                                      just_dynamo=False, **kwargs):
    kwargs_dict=dict(kwargs)
    flag1=10011
    flag2=10011
    flag3=10011

    Jfrac=point.plasma_current_fraction.values
    g_file1_name=None #redefined if run_phys_suite or save_equil
    g_file3_name=None
    run_tokamaker_case_on_point_args=list(inspect.signature(run_tokamaker_case_on_point).parameters)
    basic_dynamo_w_bootstrap_args=list(inspect.signature(basic_dynamo_w_bootstrap).parameters)
    run_tokamaker_case_on_point_args_dict={k: kwargs.pop(k) for k in kwargs_dict if k in run_tokamaker_case_on_point_args}
    basic_dynamo_w_bootstrap_dict={k: kwargs.pop(k) for k in kwargs_dict if k in basic_dynamo_w_bootstrap_args}

    #initial low res run for bootstrap current, no q>1 solving:
    try:
        mygs, flag1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, ne, Te, ni, Ti, jtor_noBS1, zeffs, pp_prof1, ffp_prof1, pax1 = run_tokamaker_case_on_point(
                                    point, mygs, plasma_dx=0.01, coil_dx=0.005, vac_dx=0.06, coil_set=2, include_bs_solver=True, 
                                    max_iterations=max_high_def_iterations, Z0=0.0, smooth_inputs=True, rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,
                                    out_all=True,Iptarget=True,initialize_eq=False, plot_iteration=plot_iteration_internal, **run_tokamaker_case_on_point_args_dict)
        if flag1!=0: #reattempt using initialize_eq=True
            print("TokaMaker init with pedestal failed, trying init with no pedestal")
            mygs.reset()
            mygs, flag1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, ne, Te, ni, Ti, jtor_noBS1, zeffs, pp_prof1, ffp_prof1, pax1 = run_tokamaker_case_on_point(point, 
                                    mygs, plasma_dx=0.01, coil_dx=0.005, vac_dx=0.06, coil_set=2, include_bs_solver=True, 
                                    max_iterations=max_high_def_iterations, Z0=0.0, smooth_inputs=True, rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,
                                    out_all=True,Iptarget=True,initialize_eq=True, plot_iteration=plot_iteration_internal, **run_tokamaker_case_on_point_args_dict)
    except:
        if not (flag1!=10011 and flag1!=0):
            flag1=10012
    
    if flag1==0 and just_dynamo:
        J_tot_true1,f_tr1=Tokamaker_get_Jtorr_and_f_tr(mygs,len(j_BS1))
        _,qvals1,_,_,_,_ = mygs.get_q(npsi=len(j_BS1))   
        if qvals1[0]>1.02: #we print THIS equilibrium as the dynamo.. gotta skip step 2 (dynamo calculation) and go straigh to printing high res (step 3)
            flag2==0
            flag_end2=True
            pax2=pax1
            ffp_prof2=ffp_prof1
            pp_prof2=pp_prof1
            j_BS2=j_BS1 
            jtor_total2=jtor_total1 
            flux_surf_avg_of_B_timesj_BS2=flux_surf_avg_of_B_timesj_BS1 
            jtor_noBS2=jtor_noBS1 
            pp_prof2=pp_prof1 
            ffp_prof2=ffp_prof1 
            qvals2=qvals1 
    #make high res grid, save high res before moving on to dynamo:   
    elif flag1==0 and not just_dynamo:
        #initialise high-res grid
        mygs.reset()
        mygs=run_tokamaker_case_on_point(point, mygs, plasma_dx=high_def_plasma_dx, just_init=True,
                                coil_dx=0.005, vac_dx=0.06, coil_set=2,
                                Z0=0.0, xpt_Roffset=0.25,xpt_Zoffset=0.1, Iptarget=True,
                                initialize_eq=False,plot_machine=False,plot_mesh=False,plot_coils=False)
        #resolve with iterated profiles in higher resolution
        mygs.solve()
        mygs.set_targets(pax=pax1,retain_previous=True)
        mygs.set_profiles(ffp_prof=ffp_prof1,pp_prof=pp_prof1)
        flag1 = mygs.solve()

        J_tot_true1,f_tr1=Tokamaker_get_Jtorr_and_f_tr(mygs,len(j_BS1))
        _,qvals1,_,_,_,_ = mygs.get_q(npsi=len(j_BS1))

        if plot_iteration_internal:
            print("Step one complete, printing high-res equilibrium")
            plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250, rho=False)
            mygs.print_info()
        if save_equil:
            if current_backdown:
                g_file1_name = save_gfile(mygs,tot_num_cases,out_path,batch_num,total_batches,gname='g_MCi',Jfrac=Jfrac,verbose=verbose,lcfs_pad=lcfs_pad,nr=nr,nz=nz)
            else:
                g_file1_name = save_gfile(mygs,tot_num_cases,out_path,batch_num,total_batches,gname='g_MCi',verbose=verbose,lcfs_pad=lcfs_pad,nr=nr,nz=nz)
        elif run_phys_suite:
            TokaMaker.save_eqdsk(mygs,out_path+"/g_eqdsk_temp",nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=1000000,ttol=ttol)
            g_file1_name = out_path+"/g_eqdsk_temp"   

    #steps 2 and 3: calculation of dynamo case
    if flag1==0 and qvals1[0]<1.02: 
        #Step two: q-profile flattening with low res grid (plasma_dx=0.01)
        #reset mygs
        mygs.reset() 
        #initialise new grid
        mygs=run_tokamaker_case_on_point(point, mygs, plasma_dx=0.01, just_init=True, 
                                    coil_dx=0.005, vac_dx=0.06, coil_set=2, 
                                    Z0=0.0, xpt_Roffset=0.25,xpt_Zoffset=0.1, Iptarget=True,
                                    initialize_eq=False)
        #run dynamo iterator
        mygs.solve()
        if plot_iteration_internal:
            fig2,ax2=plt.subplots(3,1,figsize=(4,6),constrained_layout=True)
        else:
            fig2=ax2=None

        try:
            mygs, flag2, j_BS2, jtor_total2, flux_surf_avg_of_B_timesj_BS2, jtor_noBS2, pp_prof2, ffp_prof2, q0_vals, q1_psi_surfs, qvals2, pax2, flag_end2 = basic_dynamo_w_bootstrap(mygs,ne,Te,ni,Ti,jtor_noBS1,zeffs,
                                    smooth_inputs=smooth_dynamo_inputs,max_iterations=50,initialize_eq=False,fig=fig2,ax=ax2, rescale_ind_Jtor_by_ftr=False, plot_iteration=plot_iteration_internal, **basic_dynamo_w_bootstrap_dict)
        except:
            if not (flag2!=10011 and flag2!=0):
                flag2=10013
                flag_end2=False
        
    #Gen high res dynamo
    if flag2==0 and flag_end2: 
        print("q profile converged, success!")
        if plot_iteration_internal:
            mygs.print_info()
            plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250, rho=False)
        #Step three: recalculate flattened q-profile case with high res grid (plasma_dx=0.002)
        #reset mygs
        mygs.reset()
        #initialise new grid
        mygs=run_tokamaker_case_on_point(point, mygs, plasma_dx=high_def_plasma_dx, just_init=True,
                                coil_dx=0.005, vac_dx=0.06, coil_set=2,
                                Z0=0.0, xpt_Roffset=0.25,xpt_Zoffset=0.1, Iptarget=True,
                                initialize_eq=False,plot_machine=False,plot_mesh=False,plot_coils=False)
        #resolve with iterated profiles in higher resolution
        mygs.solve()
        mygs.set_targets(pax=pax2,retain_previous=True)
        mygs.set_profiles(ffp_prof=ffp_prof2,pp_prof=pp_prof2)
        flag3 = mygs.solve()
        #if smooth_dynamo_inputs and flag3==0: #option to smooth profiles, causes one additional solve:
        #    #Note max_iterations=0 gives us a single iteration since its while n <= max iterations in solve_with_bootstrap
        #    flag3, j_BS3, jtor_total3, flux_surf_avg_of_B_timesj_BS3, jtor_noBS3, pp_prof3, ffp_prof3=solve_with_bootstrap(mygs,ne,Te,ni,Ti,jtor_noBS2,zeffs,smooth_inputs=True,max_iterations=0,rescale_ind_Jtor_by_ftr=False,initialize_eq=False,fig=None,ax=None,plot_iteration=False)
        #    _,qvals3,_,_,_,_ = mygs.get_q(npsi=len(j_BS3))
        #    flag_end3=qvals3[0]>1.02
        if flag3==0: #Roll on variables
            j_BS3=j_BS2
            jtor_total3=jtor_total2
            flux_surf_avg_of_B_timesj_BS3=flux_surf_avg_of_B_timesj_BS2
            jtor_noBS3=jtor_noBS2
            pp_prof3=pp_prof2
            ffp_prof3=ffp_prof2
            qvals3=qvals2
            flag_end3=flag_end2

            #If basic_dynamo_w_bootstrap made a hollow q-profile, shift everything up slightly in DCON to avoid the 1/1
            q0new3=0
            if min(qvals3)<1.02:
                q_increase_fac = 1.02/min(qvals3)
                q0new3=qvals3[0]*q_increase_fac #DCON input q0new3
                if q_increase_fac>1.2 or q0new3>1.24: #Need to stay below 5/4 = 1.25 for my m,n analysis
                    print("basic_dynamo_w_bootstrap produced too hollow a current profile, case aborted.")
                    flag3==1
    elif flag2!=0:
        flag3=flag2
    else:
        flag3=-100 #flag_end2 failed

    #save dynamo high res run:   
    if flag3==0:
        if plot_iteration_internal:
            print("Step three complete, printing high-res dynamo equilibrium")
            plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250, rho=False)
            mygs.print_info()
        J_tot_true3,f_tr3=Tokamaker_get_Jtorr_and_f_tr(mygs,len(j_BS3))
        if save_equil:
            if current_backdown:
                g_file3_name = save_gfile(mygs, tot_num_cases,out_path,batch_num,total_batches,gname="g_dyn_MCi",Jfrac=Jfrac,verbose=verbose,lcfs_pad=lcfs_pad,nr=nr,nz=nz)
            else:
                g_file3_name = save_gfile(mygs, tot_num_cases,out_path,batch_num,total_batches,gname="g_dyn_MCi",verbose=verbose,lcfs_pad=lcfs_pad,nr=nr,nz=nz)
        elif run_phys_suite:
            TokaMaker.save_eqdsk(mygs,out_path+"/g_dyn_eqdsk_temp",nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=1000000,ttol=ttol)
            g_file3_name = out_path+"/g_dyn_eqdsk_temp" 

    #return flag1, g_file1_name, J_tot_true1, f_tr1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1
    #return flag1, None,         None,        None,  None,  None,        None,               
    #return flag3, g_file3_name, J_tot_true3, f_tr3, j_BS3, jtor_total3, flux_surf_avg_of_B_timesj_BS3, qvals3, q0_vals, q1_psi_surfs, q0new
    #return flag3, None,         None,        None,  None,  None,        None,                          None,   None,    None,       , Nonew
    #TO RETURN:
    #                    MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jbs_GSinput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_BS)
    #                    MC_case_datasetout[1]=MC_case_datasetout[1].assign(avg_BJbs_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+avg_BJbs)
    #                    MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSinput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot)
    #                    MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSoutput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot_true)
    #                    MC_case_datasetout[1]=MC_case_datasetout[1].assign(ftr_GSoutput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+f_tr)

    mygs.reset()

    if flag1!=0:
        return flag1, None, None, None, None, None, None, None, flag3, None, None, None, None, None, None, None, None, None, None  
    elif flag1==0 and flag3!=0:
        return flag1, g_file1_name, J_tot_true1, f_tr1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, qvals1, flag3, None, None, None, None, None, None, None, None, None, None  
    else: #both cases ran... 
        print("WHERE MY RESULTS")
        return flag1, g_file1_name, J_tot_true1, f_tr1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, qvals1, flag3, g_file3_name, J_tot_true3, f_tr3, j_BS3, jtor_total3, flux_surf_avg_of_B_timesj_BS3, qvals3, q0_vals, q1_psi_surfs, q0new3

#plasma_dx=0.002 for good error performance, plasma_dx=0.01 for fast and big errors
#tested under right conditions
def run_tokamaker_case_on_point(point, mygs, plasma_dx=0.002, coil_dx=0.005, vac_dx=0.06, coil_set=2, include_bs_solver=True, plot_mesh=False,plot_coils=False,
                                max_iterations=5, Z0=0.0, xpt_Roffset=0.25,xpt_Zoffset=0.1, 
                                out_all=False,Iptarget=True,plot_iteration=True,initialize_eq=False,plot_machine=False,just_init=False, **kwargs):
    a=point.minor_radius.values
    R0=point.major_radius.values
    kappa=point.elongation_psi95.values
    delta=point.triangularity_psi95.values
    B0=point.magnetic_field_on_axis.values
    F_0 = B0*R0
    
    print("Tokamaker running: 1/12")
    #Rescaling scale factors based off device size
    rscle=0.42
    aold=0.15
    kappaold=1.3
    rscle=(R0+a)/(rscle+aold)
    zscle=a*kappa*(1.0+xpt_Zoffset)/(aold*kappaold)

    #Setting tokamaker mesh parameters
    plasma_dx = plasma_dx*rscle
    vac_dx = vac_dx*rscle
    coils=coil_sets(rscle,zscle,coil_set=coil_set)
    print("                  2/12")

    #Creating mesh objects
    gs_mesh = gs_Domain()
    gs_mesh.define_region('air',vac_dx,'boundary')
    gs_mesh.define_region('plasma',plasma_dx,'plasma')
    for i, coil_set in enumerate(coils):
        for j, coil in enumerate(coil_set):
            gs_mesh.define_region('PF_{0}_{1}'.format(i,j),coil_dx,'coil')
    print("                  3/12")

    #Defining mesh regions
    LCFS_contour = create_isoflux(80,R0,Z0,a,kappa,delta)
    maxes = LCFS_contour.max(axis=0)
    mins = LCFS_contour.min(axis=0)
    gs_mesh.add_rectangle((maxes[0]+mins[0])/2.0-0.025,(maxes[1]+mins[1])/2.0,(maxes[0]-mins[0])*1.2,kappa*a*(1.0+xpt_Zoffset)*2.3,'plasma',parent_name='air')
    # Define PF coils
    for i, coil_set in enumerate(coils):
        for j, coil in enumerate(coil_set):
            gs_mesh.add_rectangle(coil[0],coil[1],0.01,0.01,'PF_{0}_{1}'.format(i,j),parent_name='air')
    print("                  4/12")

    gs_mesh.zextents[0] = 1.2*gs_mesh.zmin
    gs_mesh.zextents[1] = 1.2*gs_mesh.zmax

    #plot mesh option
    if plot_coils:
        fig, ax = plt.subplots(1,1,figsize=(4,6),constrained_layout=True)
        gs_mesh.plot_topology(fig,ax)
    #build mesh
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    coil_dict = gs_mesh.get_coils()
    cond_dict = gs_mesh.get_conductors()
    print("                  5/12")


    if plot_mesh:
        fig, ax = plt.subplots(1,2,figsize=(8,6),constrained_layout=True)
        gs_mesh.plot_mesh(fig,ax,show_legends=False)

    #set mesh into tokamaker
    #mygs.reset()
    mygs.setup_mesh(mesh_pts,mesh_lc,mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
    mygs.settings.free_boundary = True
    mygs.settings.maxits=300
    mygs.setup(order=2,F0=F_0) #This is where B0 is specified....
    #print(mygs.lim_contour)
    print("                  6/12")

    #setting first set of PF coils as vertical stability coils...
    vsc_signs = np.zeros((mygs.ncoils,), dtype=np.float64)
    vsc_signs[[mygs.coil_sets['PF_5_0']['id'],mygs.coil_sets['PF_5_1']['id']]] = [1.0,-1.0]
    mygs.set_coil_vsc(vsc_signs)
    print("                  7/12")

    #Solving equilibrium:
    Ip_target=point.plasma_current.pint.to(ureg.A).values
    P0_target=point.peak_pressure.pint.to(ureg.Pa).values
    if Iptarget:
        mygs.set_targets(Ip=Ip_target, pax=P0_target)
    else:
        mygs.set_targets(pax=P0_target)
    #Alternative options... mygs.set_targets(Ip=Ip_target,Ip_ratio=(1.0/Beta_target - 1.0),R0=R0_target,V0=Z0)
    print("                  8/12")

    #Putting in shape contraints:
    x_point = np.array([[R0-delta*(1.0+xpt_Roffset)*a, kappa*a*(1.0+xpt_Zoffset)],[R0-delta*(1.0+xpt_Roffset), -kappa*a*(1.0+xpt_Zoffset)]])
    isoflux_pts = np.array([LCFS_contour[0],LCFS_contour[5],LCFS_contour[10],LCFS_contour[30],LCFS_contour[35],
                        LCFS_contour[40],LCFS_contour[45],LCFS_contour[50],LCFS_contour[70],LCFS_contour[75]])
    mygs.set_isoflux(np.vstack((isoflux_pts,x_point)))
    mygs.set_saddles(x_point)
    print("                  9/12")

    #Generating coil regularization matrix: 
    coil_reg_mat = np.eye(mygs.ncoils+1, dtype=np.float64)
    coil_reg_weights = np.ones((mygs.ncoils+1,))
    coil_reg_targets = np.zeros((mygs.ncoils+1,))
    # Set regularization weights
    for key, coil in mygs.coil_sets.items():
        if key.startswith('CS'):
            if key.startswith('CS1'):
                coil_reg_weights[coil['id']] = 1.E-2
            else:
                coil_reg_weights[coil['id']] = 1.E-2
        elif key.startswith('PF'):
            coil_reg_weights[coil['id']] = 1.E-2
        elif key.startswith('VS'):
            coil_reg_weights[coil['id']] = 1.E-2
    # Set weight for VSC virtual coil
    coil_reg_weights[-1] = 1.E2
    # Pass regularization terms to TokaMaker
    mygs.set_coil_reg(coil_reg_mat, reg_weights=coil_reg_weights, reg_targets=coil_reg_targets)
    print("                  10/12")

    if just_init:
        err_flag = mygs.init_psi(R0, Z0, a, kappa, delta)
        mygs.set_targets(pax=P0_target,Ip=Ip_target)
        return mygs

    #Solve equilibrium:
    if include_bs_solver:
        err_flag = mygs.init_psi(R0, Z0, a, kappa, delta)
        err_flag = mygs.solve()

        Te=np.array(point.electron_temp_profile.pint.to(ureg.eV).values)
        Ti=np.array(point.ion_temp_profile.pint.to(ureg.eV).values)
        ne=np.array(point.electron_density_profile.pint.to(ureg.m**-3).values)
        ni=np.array(point.fuel_ion_density_profile.pint.to(ureg.m**-3).values)
        if max(Te)<100: Te=Te*1000
        if max(Ti)<100: Ti=Ti*1000
        if max(ne)<1000: ne=ne*1e-19
        if max(ni)<1000: ne=ne*1e-19

        jtor_noBS=np.array(point.normalised_inductive_plasma_current_profile.values*point.inductive_current_profile_scale_fac.values) #A/m^2

        Te=on_uniform_psi_grid(Te,point.psi_normalised.values)
        Ti=on_uniform_psi_grid(Ti,point.psi_normalised.values)
        ne=on_uniform_psi_grid(ne,point.psi_normalised.values)
        ni=on_uniform_psi_grid(ni,point.psi_normalised.values)
        jtor_noBS=on_uniform_psi_grid(jtor_noBS,point.psi_normalised.values)

        if not len(Te)==len(Ti)==len(ne)==len(ni)==len(jtor_noBS)==len(point.psi_normalised.values):
            Warning("error in spine input lengths")

        mygs.set_targets(pax=P0_target,Ip=Ip_target)
        print("                  11/12")

        if plot_iteration:
            fig,ax=plt.subplots(2,1,sharex=True)
            mygs, err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, jtor_noBS, pp_prof, ffp_prof, pax = solve_with_bootstrap(mygs,ne,Te,ni,Ti,jtor_noBS,point.z_effective.values,max_iterations=max_iterations,initialize_eq=initialize_eq,fig=fig,ax=ax,plot_iteration=plot_iteration,**kwargs)
            #if qmin>0.01: 
            #    fig2,ax2=plt.subplots(4,1,sharex=True)
            #    err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS = solve_with_flat_qmin(mygs,qmin,ne,Te,ni,Ti,point.z_effective.values,iterate_psi_step=iterate_psi_step,fig=fig2,ax=ax2,plot_iteration=plot_iteration)
        else:
            mygs, err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, jtor_noBS, pp_prof, ffp_prof, pax = solve_with_bootstrap(mygs,ne,Te,ni,Ti,jtor_noBS,point.z_effective.values,max_iterations=max_iterations,initialize_eq=initialize_eq,**kwargs)
            #if qmin>0.01: 
            #    err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS = solve_with_flat_qmin(mygs,qmin,ne,Te,ni,Ti,point.z_effective.values,iterate_psi_step=iterate_psi_step)
        print("                  12/12")

        if plot_machine:
            plotmachine(mygs)

        if out_all:
            return mygs, err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, ne, Te, ni, Ti, jtor_noBS, point.z_effective.values, pp_prof, ffp_prof, pax

        return mygs, err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS
    else: 
        psi_sample=np.array(point.psi_normalised.values)
        pprime=np.array(get_pprime_popcon(point,normalised=False))
        pprime=pprime/pprime[0] #they seem to do this (negative pressure gradients cos psi_norm=1 on axis)
        ffprime=np.array(point.est_FFprime_profile.values) #mu_0*A/m
        ffprime=ffprime/ffprime[0] #useful due to disrepancy between FFprime and tokamaker internal ffp

        if psi_sample[-1] != 1.0:
            np.append(psi_sample,1.0)
            np.append(pprime,0.0)
            np.append(ffprime,0.0)
        else:
            pprime[-1]=0.0
            ffprime[-1]=0.0

        if psi_sample[0] != 0.0:
            Warning("need psi_grid (and hence rho grid in popcons) to start at zero.")

        ffp_prof= {
            'type': 'linterp',
            'x': psi_sample,
            'y': ffprime
        }
        pp_prof= {
            'type': 'linterp',
            'x': psi_sample,
            'y': pprime
        }
        print("                  11/12")

        mygs.set_profiles(ffp_prof=ffp_prof,pp_prof=pp_prof)
        err_flag = mygs.init_psi(R0, Z0, a, kappa, delta)
        err_flag = mygs.solve()
        print("                  12/12")

        if plot_machine:
            plotmachine(mygs)
    
    return mygs, err_flag

#tested
def on_uniform_psi_grid(vals,psi_vals):
    assert (psi_vals[-1]==1.0 and psi_vals[0]==0.0)
    spln=CubicSpline(psi_vals,vals)

    outvec=np.zeros(len(vals))
    psi_norm = np.linspace(0.,1.,len(vals))
    for i in range(len(vals)):
        outvec[i]=spln(psi_norm[i])

    return outvec

#tested
def plotmachine(mygs):
    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
    mygs.plot_psi(fig,ax,xpoint_color=None,vacuum_nlevels=4)
    mygs.plot_constraints(fig,ax,isoflux_color='tab:red',isoflux_marker='o')
    return fig, ax

#tested
def plot_flux_surface(mygs,LCFS_contour,kappa,a,R0):
    mygs.print_info()
    # Plot flux surfaces and source LCFS
    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
    mygs.plot_psi(fig,ax,xpoint_color=None,plasma_nlevels=10,vacuum_nlevels=4)
    ax.plot(LCFS_contour[:,0],LCFS_contour[:,1],'ro',fillstyle='none')
    mygs.plot_constraints(fig,ax,isoflux_color='tab:red',isoflux_marker='o')
    ax.set_ylim(-1.2*kappa*a,1.2*kappa*a)
    _ = ax.set_xlim(R0-1.2*a,R0+1.2*a)

    return fig,ax

#tested, works
def plot_passing_frac(mygs, psi_pad=1.E-4, npsi=500):
    psi,fc,r_avgs,_ = mygs.sauter_fc(npsi=npsi,psi_pad=psi_pad)
    return psi,fc,r_avgs[2] #final output is minor radius (in m??? check TokaMaker)

#tested
def plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=500, rho=False):
    psi,f,fp,p,pp = mygs.get_profiles(psi_pad=psi_pad,npsi=npsi)
    #fp=fp 
        #^this half isn't in the GS equation (ideal MHD Freidberg), 
        #  but it's needed to get the correct value from tokamaker's output f, fp since
        #  the f*fp tokamaker outputs is equal to 2 times the FF' from ideal MHD Freidberg
    psi_q,qvals,ravgs,dl,rbounds,zbounds = mygs.get_q(psi_pad=psi_pad,npsi=npsi)
    if rho:
        psi_q=np.sqrt(psi_q)
        psi=np.sqrt(psi)

    fig, ax = plt.subplots(6,1,sharex=True)
    # Plot F*F'
    ax[0].scatter(psi,f*fp,label='TokaMaker')
    ax[0].set_ylim(bottom=-10)
    ax[0].set_ylabel("FF'")
    ax[0].legend()
    FFp_spl=CubicSpline(psi,f*fp)
    # Plot P
    ax[1].scatter(psi,p)
    ax[1].set_ylabel("P")
    # Plot P'
    ax[2].scatter(psi,pp/max(pp))
    ax[2].set_ylabel("P' norm.")
    pp_spl=CubicSpline(psi,pp)
    # Plot q
    ax[3].scatter(psi_q,qvals)
    ax[3].set_ylim(bottom=0.0,top=6)
    ax[3].axhline([1.0],color='black',linestyle='dotted')
    ax[3].set_ylabel("q")
    # Plot < J_phi >
    avg_Jphi=pp_spl(psi_q)*ravgs[0,:]+FFp_spl(psi_q)*ravgs[1,:]/mu_0                       
    ax[4].scatter(psi_q,avg_Jphi)
    ax[4].set_ylabel(r"< $J_{\phi}$ >")
    _ = ax[-1].set_xlabel(r"$\hat{\psi}$")
    # Plot F
    ax[5].scatter(psi,f,label='TokaMaker')
    ax[5].set_ylabel("F")
    ax[5].legend()

    return fig, ax

#tested
def plot_Jbs(mygs,J_BS,point,psi_pad=1.E-6):
    n_sample=len(J_BS)
    psi,f,fp,p,pp = mygs.get_profiles(npsi=n_sample,psi_pad=psi_pad)
    fp=fp 
        #^this half isn't in the GS equation (ideal MHD Freidberg), 
        #  but it's needed to get the correct value from tokamaker's output f, fp since
        #  the f*fp tokamaker outputs is equal to 2 times the FF' from ideal MHD Freidberg
    J_ind=np.array(point.normalised_inductive_plasma_current_profile.values*point.inductive_current_profile_scale_fac.values)
    J_ind_psi_spln = CubicSpline(np.array(point.psi_normalised.values),J_ind)
    J_ind_psi = np.zeros(len(psi))
    for i in range(len(psi)):
        J_ind_psi[i]=J_ind_psi_spln(psi[i])
    fig, ax = plt.subplots(1,1,sharex=True)
    ### Calculate output j_tor and j_BS from output profiles using Grad-Shafranov equation
    _,_,ravgs,_,_,_ = mygs.get_q(npsi=n_sample,psi_pad=psi_pad) # get flux averaged R and 1/R from equilibrium solution
    R_avg = ravgs[0] # R
    one_over_R_avg = ravgs[1] # 1/R
    mu0 = np.pi*4.E-7
    tkmkr_jtor = R_avg * pp + one_over_R_avg*(f*fp) / mu0 # Jtor = R*P' + FF' / (mu0*R)
    # Plot j_total
    ax.scatter(psi,tkmkr_jtor/1e+6,label='Output $j_{tor}$') # in MA/m^2
    # Plot j_bootstrap
    ax.scatter(psi,J_BS/1e+6,label='$j_{bootstrap}$') # in MA/m^2
    ax.scatter(psi,J_ind_psi/1e+6,label='Popcon $j_{ind}$') # in MA/m^2
    ax.scatter(psi,(tkmkr_jtor-J_BS)/1e+6,label='Tokamaker $j_{ind}$') # in MA/m^2
    ax.set_ylabel("j [MA/m$^2$]")
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.set_title("H-mode $j_{tor}$")
    _ = ax.legend()
    return fig, ax

#untested but simple
def Tokamaker_get_Jtorr_and_f_tr(mygs,npsi):
    _,f,fp,_,pp = mygs.get_profiles(npsi=npsi,psi_pad=1.E-4)
    _,_,ravgs,_,_,_ = mygs.get_q(npsi=npsi,psi_pad=1.E-4) # get flux averaged R and 1/R from equilibrium solution
    _,fc,_,_ = mygs.sauter_fc(npsi=npsi,psi_pad=1.E-4) #fc is passing fraction

    R_avg = ravgs[0] # R
    one_over_R_avg = ravgs[1] # 1/R
    #fp=0.5*fp 
        #^this half isn't in the GS equation (ideal MHD Freidberg), 
        #  but it's needed to get the correct value from tokamaker's output f, fp since
        #  the f*fp tokamaker outputs is equal to 2 times the FF' from ideal MHD Freidberg

    tkmkr_jtor = R_avg * pp + one_over_R_avg*(f*fp) / mu0 # Jtor = R*P' + FF' / (mu0*R)
    return tkmkr_jtor,(1-fc) #second output is trapped fraction

#tested
def coil_sets(rscle,zscle,coil_set=3):
    # Define coil positions
    coils1 = [
        np.asarray([[rscle*0.67287,zscle*0.49371],[rscle*0.67287,-zscle*0.49371]]),
        np.asarray([[rscle*0.40958,zscle*0.80565],[rscle*0.40958,-zscle*0.80565]]),
        np.asarray([[rscle*0.71120,zscle*0.80170],[rscle*0.71120,-zscle*0.80170]]),
        np.asarray([[rscle*0.08900,zscle*0.49000],[rscle*0.08900,-zscle*0.49000]]),
        np.asarray([[rscle*0.86177,zscle*0.40640],[rscle*0.86177,-zscle*0.40640]]),
        np.asarray([[rscle*0.63106,zscle*0.36671],[rscle*0.63106,-zscle*0.36671]]),
        np.asarray([[rscle*0.07465,zscle*0.3],[rscle*0.07465,zscle*0.2],[rscle*0.07465,zscle*0.1],[rscle*0.07465,zscle*0.0],[rscle*0.07465,-zscle*0.1],[rscle*0.07465,-zscle*0.2],[rscle*0.07465,-zscle*0.3]])
    ]
    coils2 = [
        np.asarray([[rscle*0.67287,zscle*0.49371],[rscle*0.67287,-zscle*0.49371]]),
        np.asarray([[rscle*0.40958,zscle*0.80565],[rscle*0.40958,-zscle*0.80565]]),
        np.asarray([[rscle*0.71120,zscle*0.80170],[rscle*0.71120,-zscle*0.80170]]),
        np.asarray([[rscle*0.08900,zscle*0.49000],[rscle*0.08900,-zscle*0.49000]]),
        np.asarray([[rscle*0.86177,zscle*0.40640],[rscle*0.86177,-zscle*0.40640]]),
        np.asarray([[rscle*0.86177,zscle*0.8],[rscle*0.86177,-zscle*0.8]]),
        np.asarray([[rscle*0.86177,zscle*0.7],[rscle*0.86177,-zscle*0.7]]),
        np.asarray([[rscle*0.86177,zscle*0.6],[rscle*0.86177,-zscle*0.6]]),
        np.asarray([[rscle*0.86177,zscle*0.5],[rscle*0.86177,-zscle*0.5]]),

        np.asarray([[rscle*0.86177,zscle*0.2],[rscle*0.86177,-zscle*0.2]]),
        np.asarray([[rscle*0.63106,zscle*0.36671],[rscle*0.63106,-zscle*0.36671]]),
        np.asarray([[rscle*0.75,zscle*0.36671],[rscle*0.75,-zscle*0.36671]]),
        np.asarray([[rscle*0.7,zscle*0.36671],[rscle*0.7,-zscle*0.36671]]),
        np.asarray([[rscle*0.6,zscle*0.36671],[rscle*0.6,-zscle*0.36671]]),
        np.asarray([[rscle*0.55,zscle*0.36671],[rscle*0.55,-zscle*0.36671]]),
        np.asarray([[rscle*0.5,zscle*0.36671],[rscle*0.5,-zscle*0.36671]]),
        np.asarray([[rscle*0.45,zscle*0.36671],[rscle*0.45,-zscle*0.36671]]),
        np.asarray([[rscle*0.4,zscle*0.36671],[rscle*0.4,-zscle*0.36671]]),
        np.asarray([[rscle*0.35,zscle*0.36671],[rscle*0.35,-zscle*0.36671]]),
        np.asarray([[rscle*0.3,zscle*0.36671],[rscle*0.3,-zscle*0.36671]]),
        np.asarray([[rscle*0.25,zscle*0.36671],[rscle*0.25,-zscle*0.36671]]),
        np.asarray([[rscle*0.2,zscle*0.36671],[rscle*0.2,-zscle*0.36671]]),
        np.asarray([[rscle*0.15,zscle*0.36671],[rscle*0.15,-zscle*0.36671]]),
        np.asarray([[rscle*0.1,zscle*0.36671],[rscle*0.1,-zscle*0.36671]]),
        np.asarray([[rscle*0.07465,zscle*0.3],[rscle*0.07465,zscle*0.2],[rscle*0.07465,zscle*0.1],[rscle*0.07465,zscle*0.0],[rscle*0.07465,-zscle*0.1],[rscle*0.07465,-zscle*0.2],[rscle*0.07465,-zscle*0.3]])
    ]
    coils3 = [
        np.asarray([[rscle*0.67287,zscle*0.49371],[rscle*0.67287,-zscle*0.49371]]),
        np.asarray([[rscle*0.55,zscle*0.36671],[rscle*0.55,-zscle*0.36671]]),
        np.asarray([[rscle*0.45,zscle*0.36671],[rscle*0.45,-zscle*0.36671]]),
        np.asarray([[rscle*0.4,zscle*0.36671],[rscle*0.4,-zscle*0.36671]]),
        np.asarray([[rscle*0.35,zscle*0.36671],[rscle*0.35,-zscle*0.36671]]),
        np.asarray([[rscle*0.3,zscle*0.36671],[rscle*0.3,-zscle*0.36671]]),
        np.asarray([[rscle*0.25,zscle*0.36671],[rscle*0.25,-zscle*0.36671]]),
        np.asarray([[rscle*0.2,zscle*0.36671],[rscle*0.2,-zscle*0.36671]]),
        np.asarray([[rscle*0.15,zscle*0.36671],[rscle*0.15,-zscle*0.36671]]),
        np.asarray([[rscle*0.07465,zscle*0.3],[rscle*0.07465,zscle*0.2],[rscle*0.07465,zscle*0.1],[rscle*0.07465,zscle*0.0],[rscle*0.07465,-zscle*0.1],[rscle*0.07465,-zscle*0.2],[rscle*0.07465,-zscle*0.3]])
    ]
    if coil_set==1: return coils1
    if coil_set==2: return coils2
    if coil_set==3: return coils3
    else: Warning("Choose a valid coil set")
    return

#untested with run_tokamaker, run_physics_suite
def gen_n_successful_cases(n,out_path="dset",return_total=True,save=True,save_equil=True,cases_in_batch=100,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",
                           verbose=True,back_down_current=True,run_tokamaker=True,plot_coils=False,plot_iteration=False,radas_dir_path='/Users/sbenjamin/Desktop/TEARING_WIDGET/cfspopcon/radas_dir',just_dynamo=True,
                           nr=450,nz=450,maxsteps=1000000,lcfs_pad=1e-6,ttol=1e-14,plot_machine=False,req_mhd_stable=False,plot_popcons=False,plot_mesh=False,run_phys_suite=True,working_dir='',**kwargs):
    if run_phys_suite:
        run_tokamaker=True
    if back_down_current:
        return _gen_n_successful_cases(n,out_path=out_path,return_total=return_total,save_equil=save_equil,plot_coils=plot_coils,plot_iteration=plot_iteration,save=save,cases_in_batch=cases_in_batch,inputpath=inputpath,
                           verbose=verbose,run_tokamaker=run_tokamaker,req_mhd_stable=req_mhd_stable,radas_dir_path=radas_dir_path,working_dir_internal=working_dir,just_dynamo=just_dynamo,
                           nr=nr,nz=nz,maxsteps=maxsteps,lcfs_pad=lcfs_pad,ttol=ttol,plot_machine=plot_machine,plot_mesh=plot_mesh,plot_popcons=plot_popcons,run_phys_suite=run_phys_suite,**kwargs)

    #Separating out kwargs into those to send to each function (there's a lot after all)
    run_DCON_on_equilibrium_args=list(inspect.signature(run_DCON_on_equilibrium2).parameters)+list(inspect.signature(write_equil_in).parameters)+list(inspect.signature(write_dcon_inputs).parameters)+list(inspect.signature(write_rdcon_inputs).parameters)
    run_DCON_on_equilibrium_args=list(set(run_DCON_on_equilibrium_args))
    gen_random_args=list(inspect.signature(gen_random_inputs_current_backdown).parameters)
    run_DCON_on_equilibrium_args_dict={k: kwargs.pop(k) for k in dict(kwargs) if k in run_DCON_on_equilibrium_args}
    gen_random_args_dict={k: kwargs.pop(k) for k in dict(kwargs) if k in gen_random_args}

    rand_length_var=3
    total_batches=math.ceil(n/cases_in_batch)
    gen_batches = save and (total_batches>1) 
    if n>cases_in_batch and (not save): Warning("might want to save your popcons as batches as you compute them")
    if (not save) and (not return_total): Warning("gen_n_successful_cases won't do anything since save=False and return_total=False")
    if (not save) and (run_tokamaker): Warning("geqdsk files won't be saved")

    #save will either save total or save batches depending on whether we are generating batches or not

    batch_num=1
    tot_num_cases=0
    if return_total: 
        tot_dataset_vec=[]

    if save:
        try:
            shutil.copy(inputpath+"/input.yaml", out_path+"_input.yaml")
            print("input.yaml copied successfully.")
        except:
            print("input.yaml copy failed, already present in destination...")
            shutil.copy(inputpath+"/input.yaml", out_path+"_input.yaml")
    
    while tot_num_cases<n:
        batch_num_cases=0
        batch_dataset_vec=[]
        
        while batch_num_cases<min(cases_in_batch,n): #loop generating random inputs
            a=gen_random_inputs(rand_length_var*min(cases_in_batch,n),**gen_random_args_dict)
            if a.shape[0]<max(3,min(cases_in_batch,n)/10):
                rand_length_var=rand_length_var*2
                continue
            for i in range(a.shape[0]):
                if verbose:
                    print(f"Rand loop var {i} of {a.shape[0]-1} in batch {batch_num} of {total_batches}")
                    print(f"num total cases found = {tot_num_cases}")
                inputs=a[i,:]
                MC_case_datasetout = run_input_case(inputs,tot_num_cases,inputpath=inputpath,verbose=False,req_min_P=True,save_whole_grid=False)

                #Initialising tokamaker, dcon run indicators as positive in case run_tokamaker=False,run_phys_suite=False
                flag1=flag3=-123
                dcon_ran=False
                if not run_tokamaker:
                    flag1=flag3=0
                if not run_phys_suite:
                    dcon_ran=True
                if run_tokamaker and MC_case_datasetout[0]: 
                    flag1, g_file1_name, J_tot_true1, f_tr1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, qvals1, flag3, g_file3_name, J_tot_true3, f_tr3, j_BS3, jtor_total3, flux_surf_avg_of_B_timesj_BS3, qvals3, q0_vals, q1_psi_surfs, q0new3 = run_tokamaker_case_on_point_macro(
                            MC_case_datasetout[1].sel(point_type='min_pressure'), mygs, back_down_current, 
                            tot_num_cases,out_path,batch_num,total_batches,nr,nz,
                            smooth_dynamo_inputs=True, 
                            save_equil=save_equil, run_phys_suite=run_phys_suite, 
                            verbose=False,lcfs_pad=1e-6,ttol=1e-14, **kwargs)
                    mygs.reset()
                    #what conditions matter....
                    #flag 3 is dependent on flag1
                    if run_phys_suite:
                        dcon_ran1=False
                        rdcon_ran3=False
                        ideal_MHD_stable=False
                        ideal_MHD_stable1=False
                        ideal_MHD_stable3=False
                        if flag1==0: 
                            dcon_ran1,rdcon_ran1,MRE_xr1=run_DCON_on_equilibrium2(g_file1_name,newq0=0,**run_DCON_on_equilibrium_args_dict)
                            MRE_xr1=MRE_xr1.assign_coords(equil_type='peaked')
                            if req_mhd_stable:
                                ideal_MHD_stable1=((not any(MRE_xr1.dcon_nzero.values!=0)) and MRE_xr1.dW_total>0)
                            else:
                                ideal_MHD_stable1=True
                        if flag3==0:
                            dcon_ran3,rdcon_ran3,MRE_xr3=run_DCON_on_equilibrium2(g_file3_name,newq0=q0new3,**run_DCON_on_equilibrium_args_dict) #focus is on getting MRE data for this one...
                            MRE_xr3=MRE_xr3.assign_coords(equil_type='dynamo')
                            if req_mhd_stable:
                                ideal_MHD_stable3=((not any(MRE_xr3.dcon_nzero.values!=0)) and MRE_xr3.dW_total>0)
                            else:
                                ideal_MHD_stable3=True
                            MRE_xr1=xr.concat([MRE_xr1,MRE_xr3],dim="equil_type")
                        if req_mhd_stable:
                            ideal_MHD_stable = ideal_MHD_stable1 or ideal_MHD_stable3
                        else:
                            ideal_MHD_stable = True
                        dcon_ran = dcon_ran1 and rdcon_ran3 #valid surface values for peaked, valid rdcon values for dynamo
                        dcon_ran = dcon_ran and ideal_MHD_stable #extra requirement of stability... optional, maybe set req_mhd_stable=False since we want all surface values...
                    else:
                        dcon_ran=True

                #Putting tokamaker info and dcon info into our results_store
                if MC_case_datasetout[0] and (flag1==0 and flag3==0) and dcon_ran: #This represents a positive case with dynamo equilibrium generated, qsurf on 
                    if run_tokamaker:
                        if flag1==0:
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jbs_GSinput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+j_BS1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(avg_BJbs_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+flux_surf_avg_of_B_timesj_BS1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSinput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+jtor_total1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSoutput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot_true1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(ftr_GSoutput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+f_tr1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(q_GSoutput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+qvals1)
                        if flag3==0:
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jbs_GSinput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+j_BS3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(avg_BJbs_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+flux_surf_avg_of_B_timesj_BS3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSinput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+jtor_total3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSoutput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot_true3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(ftr_GSoutput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+f_tr3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(q_GSoutput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+qvals3)
                            #MC_case_datasetout[1]=MC_case_datasetout[1].assign(q0_dynamo_iteration=q0_vals,q1_psi_surf_dynamo_iteration=q1_psi_surfs)

                    if run_phys_suite and dcon_ran:
                        if dcon_ran and ideal_MHD_stable: 
                            MC_case_datasetout[1]=MC_case_datasetout[1].merge(MRE_xr1)

                    if plot_popcons:
                        plot_case_from_dataset_point(MC_case_datasetout[1],inputpath=inputpath,plot_style_yaml=inputpath+"/plot_popcon.yaml")
                    batch_dataset_vec.append(MC_case_datasetout[1])
                    if return_total: 
                        tot_dataset_vec.append(MC_case_datasetout[1])
                    tot_num_cases+=1
                    batch_num_cases+=1
                MC_case_datasetout=None
                if batch_num_cases>=min(cases_in_batch,n):
                    if verbose:
                        print(f"Batch filled, breaking loop of random inputs")
                    break
    
        if gen_batches: #'and save:' is implied as per starting logic
            batch_dataset=xr.concat(batch_dataset_vec,dim="MC_index")
            batch_dataset.drop_vars(
                ["atomic_data","density_profile_form","temp_profile_form","radiated_power_method","radas_dir","core_impurity_species","dim_species"]
                ).to_netcdf(path=out_path+f"_popcons_b{batch_num}of{total_batches}",engine="scipy")
                #if run_tokamaker:
                #   save_gfiles(mygs_batch_vec,J_BS_batch_vec,J_tot_batch_vec,MC_ind_batch_vec,out_path,batch_num,total_batches)
            batch_num+=1

    if save and (total_batches==1):
        total_dataset=xr.concat(batch_dataset_vec,dim="MC_index")
        total_dataset.drop_vars(
            ["atomic_data","density_profile_form","temp_profile_form","radiated_power_method","radas_dir","core_impurity_species","dim_species"]
            ).to_netcdf(path=out_path+f"_popcons_b1of1",engine="scipy")
            #if run_tokamaker:
            #   save_gfiles(mygs_batch_vec,J_BS_batch_vec,J_tot_batch_vec,MC_ind_batch_vec,out_path,1,1)
        if return_total: 
            if run_tokamaker:
                return total_dataset
            return total_dataset
    elif return_total: 
        if run_tokamaker:
                return xr.concat(tot_dataset_vec,dim="MC_index")
        return xr.concat(tot_dataset_vec,dim="MC_index")
    return 


#def compile_results(out_path,n=0,cases_in_batch=0,):
    #if not os.path.isfile(out_path+"/run_numbers.csv") and n==0 and cases_in_batch==0:
    #    raise "Need to specify n, cases_in_batch. gen_n_successful_cases didn't write them."
    #elif os.path.isfile(out_path+"/run_numbers.csv"):
    #    run_numbers_DF=pd.read_csv(out_path+"/run_numbers.csv", dtype={'n': 'int', 'cases_in_batch': 'int'})
    #    n=run_numbers_DF.n.values[0]
    #    cases_in_batch=run_numbers_DF.cases_in_batch.values[0]

    #total_batches=math.ceil(n/cases_in_batch)

    #Need list of dcon_ddirectories
    #Luckily main files just in out_path
    #for i in range(1,total_batches+1):

def run_dcon_on_equilibria(out_path,dcon_executable,rdcon_executable,n=0,cases_in_batch=0,**kwargs):
    if not os.path.isfile(out_path+"/run_numbers.csv") and n==0 and cases_in_batch==0:
        raise "Need to specify n, cases_in_batch. gen_n_successful_cases didn't write them."
    elif os.path.isfile(out_path+"/run_numbers.csv"):
        run_numbers_DF=pd.read_csv(out_path+"/run_numbers.csv", dtype={'n': 'int', 'cases_in_batch': 'int'})
        n=run_numbers_DF.n.values[0]
        cases_in_batch=run_numbers_DF.cases_in_batch.values[0]

    if os.path.isfile(out_path+"/doccupied"):
        print(out_path," already occupied, cycling...")
        return
    else:
        f_2=open(out_path+"/doccupied", 'w')
        f_2.write("hi")
        f_2.close()
    
    working_dir=out_path+f"/dcon_working_dir"
    working_dir=working_dir.replace("//", "/")
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    
    total_batches=math.ceil(n/cases_in_batch)

    read_write_dir_vec = []

    for i in range(1,total_batches+1):
        tokamaker_xr_name=out_path+f"/tokamak_xr_{i}of{total_batches}"
        eqdsk_dir_name=out_path+f"/eqdsks_b{i}of{total_batches}"
        dcon_dir_name=out_path+f"/dcon_xrs_b{i}of{total_batches}"
        eqdsk_dir_name=eqdsk_dir_name.replace("//", "/")
        dcon_dir_name=dcon_dir_name.replace("//", "/")

        if os.path.isdir(eqdsk_dir_name) and os.path.isfile(tokamaker_xr_name):
            if not os.path.isdir(dcon_dir_name):
                os.mkdir(dcon_dir_name)
            read_write_dir_vec.append([eqdsk_dir_name,dcon_dir_name])
        else:
            break
            
    for i in read_write_dir_vec:
        equil_names = os.listdir(i[0])
        if len(equil_names)>0:
            for j in equil_names:
                if not (os.path.isfile(i[1]+"/"+j+"_dcon_xr") or os.path.isfile(i[1]+"/"+j+"_dcon_xrF")): # and os.path.isfile(i+"/"+j):
                    #equil_directory.append([i+"/"+j,i]) #filename, directory
                    print("running dcon on equilibrium file: ",i[0]+"/"+j)
                    dcon_ran,rdcon_ran,MRE_xr,ideal_xr,DP_xr=run_DCON_on_equilibrium2(i[0]+"/"+j,
                            working_dir=working_dir,dcon_executable=dcon_executable,
                            rdcon_executable=rdcon_executable,
                            **kwargs)
                    if dcon_ran:
                        MRE_xr.to_netcdf(path=i[1]+"/"+j+"_dcon_xr",engine="scipy")
                        ideal_xr.to_netcdf(path=i[1]+"/"+j+"_ideal_xr",engine="scipy")
                    if rdcon_ran:
                        DP_xr.to_netcdf(path=i[1]+"/"+j+"_DP_xr",engine="scipy")
                    else:
                        f__0=open(i[1]+"/"+j+"_dcon_xrF", 'w')
                        f__0.write('dcon_ran,rdcon_ran\n')
                        f__0.write(f'{str(dcon_ran)},{str(rdcon_ran)}\n')
                        f__0.close()

    #for i in equil_directory:
    print("dcon analysis up to date on ",out_path)
    os.remove(out_path+"/doccupied")
    return
        
#untested with run_tokamaker, run_physics_suite
def _gen_n_successful_cases(n,out_path="dset",return_total=True,save=True,save_equil=True,run_tokamaker=False,cases_in_batch=100,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",radas_dir_path='/Users/sbenjamin/Desktop/TEARING_WIDGET/cfspopcon/radas_dir',
                           verbose=True,plot_iteration=False,plot_coils=False,plot_machine=False,plot_mesh=False,
                           just_dynamo=False,nr=450,nz=450,maxsteps=1000000,lcfs_pad=1e-6,ttol=1e-14,plot_popcons=False,run_phys_suite=True,req_mhd_stable=False,working_dir_internal='',**kwargs):


    out_path=out_path.replace("//", "/")
    OFT_working_dir=out_path+"/OFT_working_dir/"
    OFT_working_dir=OFT_working_dir.replace("//", "/")
    DCON_working_dir=out_path+"/dcon_working_dir" #keep it with no end dash!
    DCON_working_dir=DCON_working_dir.replace("//", "/")

    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if not os.path.isdir(OFT_working_dir):
        os.mkdir(OFT_working_dir)
    os.chdir(OFT_working_dir)
    mygs = TokaMaker()
    if not os.path.isdir(DCON_working_dir):
        os.mkdir(DCON_working_dir)

    if not os.path.isfile(out_path+"/run_numbers.csv"):
        f__1=open(out_path+"/run_numbers.csv", 'w')
        f__1.write('n,cases_in_batch\n')
        f__1.write(f'{str(n)},{str(cases_in_batch)}\n')
        f__1.close()

    if not os.path.isdir(out_path + "/radas_dir"):
        try:
            shutil.copytree(radas_dir_path, out_path+"/radas_dir")
        except OSError as exc: # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(radas_dir_path, out_path+"/radas_dir")
            else: raise

    if not os.path.isdir(OFT_working_dir + "/radas_dir"):
        try:
            shutil.copytree(radas_dir_path, OFT_working_dir+"/radas_dir")
        except OSError as exc: # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(radas_dir_path, OFT_working_dir+"/radas_dir")
            else: raise

    num_failed_dynamos=0
    num_failed_dcons=0
    num_failed_rdcons=0
    failed_cases=[]
    total_batches=math.ceil(n/cases_in_batch)
    gen_batches = save and (total_batches>1)
    if n>cases_in_batch and (not save): Warning("might want to save your popcons as batches as you compute them")
    if (not save) and (not return_total): Warning("gen_n_successful_cases won't do anything since save=False and return_total=False")
    #save will either save total or save batches depending on whether we are generating batches or not

    #Separating out kwargs into those to send to each function (there's a lot after all)
    run_DCON_on_equilibrium_args=list(inspect.signature(run_DCON_on_equilibrium2).parameters)+list(inspect.signature(write_equil_in).parameters)+list(inspect.signature(write_dcon_inputs).parameters)+list(inspect.signature(write_rdcon_inputs).parameters)
    run_DCON_on_equilibrium_args=list(set(run_DCON_on_equilibrium_args))
    gen_random_args=list(inspect.signature(gen_random_inputs_current_backdown).parameters)
    run_DCON_on_equilibrium_args_dict={k: kwargs.pop(k) for k in dict(kwargs) if k in run_DCON_on_equilibrium_args}
    gen_random_args_dict={k: kwargs.pop(k) for k in dict(kwargs) if k in gen_random_args}
    
    random.seed()

    batch_num=1
    tot_num_cases=0
    if return_total: 
        tot_dataset_vec=[]

    #hotstart functionality, wont override previous cases...:
    for i in reversed(range(1,total_batches+1)):
        if os.path.isfile(out_path+f"/tokamak_xr_{i}of{total_batches}") and i<total_batches:
            print("hotstart detected at batch ",i)
            batch_num=i+1
            tot_num_cases=cases_in_batch*i
            break

    if save:
        try:
            shutil.copy(inputpath+"/input.yaml", out_path+"_input.yaml")
            print("input.yaml copied successfully.")
        except:
            print("input.yaml copy failed, already present in destination...")
            shutil.copy(inputpath+"/input.yaml", out_path+"_input.yaml")
    
    while tot_num_cases<n:
        print(f"Batch number {batch_num} of {total_batches} starting.")
        if run_phys_suite:
             print(f"Num failed dynamos = {num_failed_dynamos}.")
        batch_num_cases=0
        batch_dataset_vec=[]
        
        while batch_num_cases<min(cases_in_batch,n): #loop generating random inputs
            os.chdir(out_path)
            a=gen_random_inputs_current_backdown(**gen_random_args_dict)
            found_case=False
            device_dataset_vec=[]
            min_working_curr_frac=a[-1,-2]
            max_working_curr_frac=a[0,-2]

            for i in range(a.shape[0]):
                if verbose:
                    print(f"Rand loop var {i} of {a.shape[0]-1} in batch {batch_num} of {total_batches}")
                    print(f"num total cases found = {tot_num_cases}")
            
                inputs=a[i,:]

                os.chdir(out_path)
                MC_case_datasetout = run_input_case(inputs,tot_num_cases,inputpath=inputpath,verbose=False,req_min_P=True,save_whole_grid=False)
                
                flag1=flag3=-123
                if not run_tokamaker:
                    flag1=flag3=0
                dcon_ran = not run_phys_suite
    
                if run_tokamaker and MC_case_datasetout[0]: 
                    os.chdir(OFT_working_dir)
                    flag1, g_file1_name, J_tot_true1, f_tr1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, qvals1, flag3, g_file3_name, J_tot_true3, f_tr3, j_BS3, jtor_total3, flux_surf_avg_of_B_timesj_BS3, qvals3, q0_vals, q1_psi_surfs, q0new3  = run_tokamaker_case_on_point_macro(
                            MC_case_datasetout[1].sel(point_type='min_pressure'), mygs, True, 
                            tot_num_cases,out_path,batch_num,total_batches,nr,nz,
                            smooth_dynamo_inputs=True,  
                            save_equil=save_equil, run_phys_suite=run_phys_suite, 
                            verbose=False,lcfs_pad=1e-6,ttol=1e-14, 
                            just_dynamo=just_dynamo,**kwargs)
                    if run_phys_suite:
                        dcon_ran1=False
                        rdcon_ran3=False
                        ideal_MHD_stable=False
                        ideal_MHD_stable1=False
                        ideal_MHD_stable3=False
                        #if flag1==0: 
                            #os.chdir(DCON_working_dir)
                            #dcon_ran1,rdcon_ran1,MRE_xr1=run_DCON_on_equilibrium2(g_file1_name,newq0=0,working_dir=working_dir_internal,**run_DCON_on_equilibrium_args_dict)
                            #MRE_xr1=MRE_xr1.assign_coords(equil_type='peaked')
                            #if req_mhd_stable:
                            #    ideal_MHD_stable1=((not any(MRE_xr1.dcon_nzero.values!=0)) and MRE_xr1.dW_total>0)
                            #else:
                            #    ideal_MHD_stable1=True
                        if flag3==0:
                            try:
                                os.chdir(DCON_working_dir)
                                dcon_ran3,rdcon_ran3,MRE_xr3=run_DCON_on_equilibrium2(g_file3_name,newq0=q0new3,working_dir=DCON_working_dir,**run_DCON_on_equilibrium_args_dict) #focus is on getting MRE data for this one...
                            except:
                                raise
                                dcon_ran3=rdcon_ran3=False
                                num_failed_dcons+=1
                                num_failed_rdcons+=1
                                failed_cases.append(MC_case_datasetout[0])

                            if req_mhd_stable:
                                ideal_MHD_stable3=((not any(MRE_xr3.dcon_nzero.values!=0)) and MRE_xr3.dW_total>0)
                            else:
                                ideal_MHD_stable3=True
                            #MRE_xr1=xr.concat([MRE_xr1,MRE_xr3],dim="equil_type")

                            if rdcon_ran3:
                                MRE_xr3=MRE_xr3.assign_coords(equil_type='dynamo')
                                MRE_xr1=xr.concat([MRE_xr3],dim="equil_type")
                                #MRE_xr1=xr.concat([MRE_xr1,MRE_xr3],dim="equil_type")
                        if req_mhd_stable:
                            ideal_MHD_stable = ideal_MHD_stable1 or ideal_MHD_stable3
                        else:
                            ideal_MHD_stable = True
                        dcon_ran = rdcon_ran3 #dcon_ran1 and rdcon_ran3 #valid surface values for peaked, valid rdcon values for dynamo
                        dcon_ran = dcon_ran and ideal_MHD_stable #extra requirement of stability... optional, maybe set req_mhd_stable=False since we want all surface values...
                        if flag1==0 and flag3!=0:
                            num_failed_dynamos+=1
                    else:
                        dcon_ran=True

                #Putting tokamaker info and dcon info into our results_store
                #if MC_case_datasetout[0] and (flag1==0 and flag3==0) and dcon_ran: #This represents a positive case with dynamo equilibrium generated, qsurf working... 
                if MC_case_datasetout[0] and (flag3==0) and dcon_ran: #This represents a positive case with dynamo equilibrium generated, qsurf working... 
                    found_case=True
                    found_case=True
                    min_working_curr_frac=inputs[-2]
                    if run_tokamaker:
                        if flag1==0 and not just_dynamo:
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jbs_GSinput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+j_BS1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(avg_BJbs_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+flux_surf_avg_of_B_timesj_BS1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSinput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+jtor_total1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSoutput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot_true1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(ftr_GSoutput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+f_tr1)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(q_GSoutput_on_linspace_psi_peaked=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+qvals1)
                        if flag3==0:
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jbs_GSinput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+j_BS3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(avg_BJbs_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+flux_surf_avg_of_B_timesj_BS3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSinput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+jtor_total3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSoutput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot_true3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(ftr_GSoutput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+f_tr3)
                            MC_case_datasetout[1]=MC_case_datasetout[1].assign(q_GSoutput_on_linspace_psi_dynamo=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+qvals3)
                            #MC_case_datasetout[1]=MC_case_datasetout[1].assign(q0_dynamo_iteration=q0_vals,q1_psi_surf_dynamo_iteration=q1_psi_surfs)

                    if run_phys_suite and dcon_ran:
                        if dcon_ran and ideal_MHD_stable: 
                            MC_case_datasetout[1]=MC_case_datasetout[1].merge(MRE_xr1)

                    device_dataset_vec.append(MC_case_datasetout[1])

                    if plot_popcons:
                        plot_case_from_dataset_point(MC_case_datasetout[1],inputpath=inputpath,plot_style_yaml=inputpath+"/plot_popcon.yaml")
                
                if not MC_case_datasetout[0]:# or i==1:  #HERE TRUNCATING
                    break #no point lowering current more once no valid popcon device region has been found

            if found_case:
                device_dataset=xr.concat(device_dataset_vec,dim="plasma_current_fraction")
                batch_dataset_vec.append(device_dataset)
                batch_num_cases+=1
                tot_num_cases+=1
                if return_total:
                    tot_dataset_vec.append(device_dataset)
                if verbose:
                    print(f"Device with R0,eps,delta,kappa=({inputs[0],inputs[1],inputs[2],inputs[3]}) found cases down to Ip_frac={min_working_curr_frac}")
            else:
                if verbose:
                    print(f"Device with R0,eps,delta,kappa=({inputs[0],inputs[1],inputs[2],inputs[3]}) had no working scenarios for Ip_frac<={max_working_curr_frac}")
    
        if gen_batches: #'and save:' is implied as per starting logic
            batch_dataset=xr.concat(batch_dataset_vec,dim="MC_index")
            batch_dataset.drop_vars(
                ["atomic_data","density_profile_form","temp_profile_form","radiated_power_method","radas_dir","core_impurity_species","dim_species"]
                ).to_netcdf(path=out_path+f"/tokamak_xr_{batch_num}of{total_batches}",engine="scipy")
            batch_num+=1
            if len(failed_cases)>0:
                xr.concat(failed_cases,dim="MC_index").drop_vars(
                    ["atomic_data","density_profile_form","temp_profile_form","radiated_power_method","radas_dir","core_impurity_species","dim_species"]
                    ).to_netcdf(path=out_path+f"/failed_cases_b{batch_num}of{total_batches}",engine="scipy")

    if save and (total_batches==1):
        total_dataset=xr.concat(batch_dataset_vec,dim="MC_index")
        total_dataset.drop_vars(
            ["atomic_data","density_profile_form","temp_profile_form","radiated_power_method","radas_dir","core_impurity_species","dim_species"]
            ).to_netcdf(path=out_path+f"/tokamak_xr_b1of1",engine="scipy")
        if return_total:
            return total_dataset
    elif return_total: 
        total_dataset=xr.concat(tot_dataset_vec,dim="MC_index")
        if save:
            total_dataset.drop_vars(
                ["atomic_data","density_profile_form","temp_profile_form","radiated_power_method","radas_dir","core_impurity_species","dim_species"]
            ).to_netcdf(path=out_path+f"/tokamak_xr",engine="scipy")
        return total_dataset
    return 

""" Old logic:
mygscopy, err_flag, J_BS, J_tot, avg_BJbs = run_tokamaker_case_on_point(MC_case_datasetout[1].sel(point_type='min_pressure'), mygs, 
                                        plasma_dx=plasma_dx, Iptarget=True,plot_coils=plot_coils, plot_iteration=plot_iteration,plot_machine=plot_machine,plot_mesh=plot_mesh,initialize_eq=False)
if err_flag!=0:
    print("TokaMaker init with pedestal failed, trying init with no pedestal")
    mygs.reset()
    mygscopy, err_flag, J_BS, J_tot, avg_BJbs = run_tokamaker_case_on_point(MC_case_datasetout[1].sel(point_type='min_pressure'), mygs, 
                                        plasma_dx=plasma_dx, Iptarget=True,plot_coils=plot_coils, plot_iteration=plot_iteration,plot_machine=plot_machine,plot_mesh=plot_mesh,initialize_eq=True)
if err_flag==0:
    J_tot_true,f_tr=Tokamaker_get_Jtorr_and_f_tr(mygs,len(J_BS))
    if save_equil:
        save_gfile(tot_num_cases,out_path,batch_num,total_batches,Jfrac=MC_case_datasetout[1].plasma_current_fraction.values,verbose=verbose,lcfs_pad=lcfs_pad,nr=nr,nz=nz)
        g_file_name=out_path+f"/eqdsks_b{batch_num}of{total_batches}/g_MCi{tot_num_cases}_Jfrac{MC_case_datasetout[1].plasma_current_fraction.values}"
    elif run_phys_suite:
        TokaMaker.save_eqdsk(mygs,out_path+"/g_eqdsk_temp",nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=1000000,ttol=ttol)
        g_file_name=out_path+"/g_eqdsk_temp"
    if run_phys_suite:
        dcon_ran,MRE_xr=run_DCON_on_equilibrium2(g_file_name,**run_DCON_on_equilibrium_args_dict)
        if req_mhd_stable:
            ideal_MHD_stable = (not any(MRE_xr.dcon_nzero.values!=0))
        else:
            ideal_MHD_stable=True
else:
    print("TokaMaker failed for this case")
mygs.reset()
if MC_case_datasetout[0] and err_flag==0 and ((not run_phys_suite) or (run_phys_suite and dcon_ran and ideal_MHD_stable)): 
found_case=True
min_working_curr_frac=inputs[-2]
if run_tokamaker:
    MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jbs_GSinput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_BS)
    MC_case_datasetout[1]=MC_case_datasetout[1].assign(avg_BJbs_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+avg_BJbs)
    MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSinput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot)
    MC_case_datasetout[1]=MC_case_datasetout[1].assign(Jtor_GSoutput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+J_tot_true)
    MC_case_datasetout[1]=MC_case_datasetout[1].assign(ftr_GSoutput_on_linspace_psi_from_TokaMaker_min_P_case=0.0*MC_case_datasetout[1].sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"]+f_tr)
    if run_phys_suite:
        if dcon_ran and ideal_MHD_stable: 
            MC_case_datasetout[1]=MC_case_datasetout[1].merge(MRE_xr)
device_dataset_vec.append(MC_case_datasetout[1])
if plot_popcons:
    plot_case_from_dataset_point(MC_case_datasetout[1],inputpath=inputpath,plot_style_yaml=inputpath+"/plot_popcon.yaml")
else: 
break #no point lowering current more once no valid device region has been found
"""
                    
#tested
def run_inputs_in_popcon(a,MCindex_offset=0,req_min_P=True,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",verbose=False,save_whole_grid=False):
    random_length=a.shape[0]
    shot_datasets = []
    if save_whole_grid:
        grid_dataset = []
        dataset_points = []

    for i in range(random_length):
        inputs=a[i,:]

        output = run_input_case(inputs,i+MCindex_offset,inputpath=inputpath,verbose=verbose,req_min_P=req_min_P,save_whole_grid=save_whole_grid)
        if output[0]==False: continue

        if save_whole_grid:
            shot_datasets.append(output[1])
            grid_dataset.append(output[2])
            dataset_points.append(output[3])
        else:
            shot_datasets.append(output[1])

    if shot_datasets == None or len(shot_datasets)==0:
        return 
    
    popcon_dataset=xr.concat(shot_datasets,dim="MC_index")

    if save_whole_grid:
        grid_dataset=xr.concat(grid_dataset,dim="MC_index")
        return [popcon_dataset,grid_dataset,dataset_points]
    
    return popcon_dataset

#tested
def run_input_case(inputs,MC_index,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",req_min_P=True,verbose=False,save_whole_grid=False):
    if verbose: 
        print("run_input_case initiated")

    found_case=False
    type_datasets = []

    input_parameters, algorithm, points, plots  = cfspopcon.read_case(inputpath,dict(
        major_radius=inputs[0],
        inverse_aspect_ratio=inputs[1],
        triangularity_psi95=inputs[2],
        elongation_psi95=inputs[3],
        plasma_current=inputs[4],
        magnetic_field_on_axis=inputs[5]
    ))
    plasma_current_fraction=inputs[6]
    if verbose: 
        print("run_input_case 1/5")

    algorithm.validate_inputs(input_parameters)
    if verbose: 
        print("run_input_case 1.1/5")
    dataset = xr.Dataset(input_parameters)
    if verbose: 
        print("run_input_case 1.2/5")
    dataset = algorithm.update_dataset(dataset)
    if verbose: 
        print("run_input_case 1.3/5")
    dataset = dataset.set_coords(names=["major_radius","inverse_aspect_ratio","triangularity_psi95","elongation_psi95"])
    if verbose: 
        print("run_input_case 1.4/5")
    dataset = dataset.assign_coords(plasma_current_fraction=plasma_current_fraction)

    if verbose: 
        print("run_input_case 2/5")

    minPressure_params = points["MinPressure"]
    maxFusion_params = points["MaxFusion"]
    maxQ_params = points["MaxQ"]

    minPressureMask = cfspopcon.shaping_and_selection.build_mask_from_dict(dataset, minPressure_params)
    maxFusionMask = cfspopcon.shaping_and_selection.build_mask_from_dict(dataset, maxFusion_params)
    maxQMask = cfspopcon.shaping_and_selection.build_mask_from_dict(dataset, maxQ_params)

    if verbose: 
        print("run_input_case 3/5")

    if dataset.Q.where(minPressureMask).count()>0:
        found_case=True
        minPpt=dataset.isel(cfspopcon.shaping_and_selection.find_coords_of_minimum(dataset.peak_pressure, mask=minPressureMask))
        minPpt=minPpt.assign_coords(point_type="min_pressure",MC_index=MC_index)
        minPpt=prepare_input_profiles(minPpt)
        type_datasets.append(minPpt)
        if verbose:
            print("Min pressure point identified")
            print(minPpt.P_fusion)
        
    if dataset.Q.where(maxFusionMask).count()>0:
        if not req_min_P:
            found_case=True
        maxFpt=dataset.isel(cfspopcon.shaping_and_selection.find_coords_of_maximum(dataset.P_fusion, mask=maxFusionMask))
        maxFpt=maxFpt.assign_coords(point_type="max_fusion",MC_index=MC_index)
        maxFpt=prepare_input_profiles(maxFpt)
        type_datasets.append(maxFpt)
        if verbose:
            print("Max Fusion point identified")
            print(maxFpt.P_fusion)

    if dataset.Q.where(maxQMask).count()>0:
        if not req_min_P:
            found_case=True
        maxQpt=dataset.isel(cfspopcon.shaping_and_selection.find_coords_of_maximum(dataset.Q, mask=maxQMask))
        maxQpt=maxQpt.assign_coords(point_type="max_Q",MC_index=MC_index)
        maxQpt=prepare_input_profiles(maxQpt)
        type_datasets.append(maxQpt)
        if verbose:
            print("Max Q point identified")
            print(maxQpt.P_fusion)
            print(maxQpt.Q)

    if verbose: 
        print("run_input_case 4/5")
        print(type_datasets)
        print(found_case)

    if found_case:
        MC_case_dataset_loc=xr.concat(type_datasets,dim="point_type")
        if verbose: 
            print("run_input_case 5/5")
        if save_whole_grid:
            dataset=dataset.assign_coords(MC_index=MC_index)
            return [True,MC_case_dataset_loc,dataset,points]
        return [True,MC_case_dataset_loc]
    return [False,None]
    
#tested
def plot_case_from_dataset_point(dataset_point,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",plot_style_yaml="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/plot_popcon.yaml"):
    output=get_full_grid_from_dataset_point(dataset_point,inputpath=inputpath)
    plot=plot_case_from_grid(output[0],output[1],plot_style_yaml=plot_style_yaml)
    return plot

#tested
def get_full_grid_from_dataset_point(dataset_point,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts"):
    try:
        if dataset_point.dims['point_type']>0:
            try:
                try:
                    dataset_point=dataset_point.sel(point_type='min_pressure')
                    
                except:
                    dataset_point=dataset_point.sel(point_type='max_Q')
            except:
                dataset_point=dataset_point.sel(point_type='max_Fusion')
    except:
        dataset_point=dataset_point


    input_parameters, algorithm, points, plots  = cfspopcon.read_case(inputpath,dict(
        major_radius=dataset_point.major_radius.values,
        inverse_aspect_ratio=dataset_point.inverse_aspect_ratio.values,
        triangularity_psi95=dataset_point.triangularity_psi95.values,
        elongation_psi95=dataset_point.elongation_psi95.values,
        plasma_current=dataset_point.plasma_current.values,
        magnetic_field_on_axis=dataset_point.magnetic_field_on_axis.values,
    ))
    plasma_current_fraction=dataset_point.plasma_current_fraction.values

    algorithm.validate_inputs(input_parameters)
    dataset = xr.Dataset(input_parameters)
    dataset = algorithm.update_dataset(dataset)
    dataset = dataset.set_coords(names=["major_radius","inverse_aspect_ratio","triangularity_psi95","elongation_psi95"])
    dataset = dataset.assign_coords(plasma_current_fraction=plasma_current_fraction)

    return [dataset,points]

#tested
def plot_case_from_grid(grid_dataset,dataset_points,MC_index=None,plot_style_yaml="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/plot_popcon.yaml"):
    plot_style = cfspopcon.read_plot_style(plot_style_yaml)

    if MC_index==None:
        plot=cfspopcon.plotting.make_plot(
            grid_dataset,
            plot_style,
            points=dataset_points,
            title="POPCON example",
            save_name=None
        )
    else:
        plot=cfspopcon.plotting.make_plot(
            grid_dataset.isel(MC_index=MC_index),
            plot_style,
            points=dataset_points[MC_index],
            title="POPCON example",
            save_name=None
        )

    return plot

#tested, but FFprime not confirmed
def prepare_input_profiles(point):
    point=point.assign(pressure_profile=
        point["electron_temp_profile"] * point["electron_density_profile"] +
        point["ion_temp_profile"] * point["fuel_ion_density_profile"])
    
    point=point.assign(normalised_inductive_plasma_current_profile=
           (1.0/((2.8e-8)*(point["electron_temp_profile"]**(-1.5))))/
        max(1.0/((2.8e-8)*(point["electron_temp_profile"]**(-1.5))))
    )

    minor_radius_values=point.rho.values*point.minor_radius.pint.to(ureg.m).values #m

    Jtorr_inductive_integration_spline=CubicSpline(minor_radius_values,minor_radius_values*point.normalised_inductive_plasma_current_profile.values)
    
    cross_section_surf_int=2*np.pi*Jtorr_inductive_integration_spline.integrate(min(minor_radius_values), max(minor_radius_values), extrapolate=False)
    cross_section_surf_int=point.elongation_psi95*cross_section_surf_int #ratio of elliptical cross sectional area to circular cross sectional area, not exactly right
                                                                          #but either way other approximations will overridde during the GS solve 

    inductive_current_profile_scale_fac=point.inductive_plasma_current.pint.to(ureg.A).values/cross_section_surf_int #A/m^2
    total_current_profile_scale_fac=point.plasma_current.pint.to(ureg.A).values/cross_section_surf_int #A/m^2
    point=point.assign(inductive_current_profile_scale_fac=inductive_current_profile_scale_fac,total_current_profile_scale_fac=total_current_profile_scale_fac)

    ########popcons outputs in rho=sqrt(psi) where psi is the normalised poloidal flux  ########
    ########tokamaker takes inputs in normalised poloidal flux                          ########
    point=point.assign_coords(psi_normalised=point["rho"]**2)
    point=point.reset_coords(names="rho", drop=False)

    ###.assign 12 and 13 below need checking...
    total_poloidal_flux=abs(est_total_poloidal_flux(point)) #positive poloidal flux
    point=point.assign(est_total_poloidal_flux=total_poloidal_flux)

    pprime=get_pprime_popcon(point)
    ######## Deprecated below: FFprime_profile=estimate_FFprime(point,pprime) ########
    ######## Deprecated below: FFprime_normalised=FFprime_profile             ########
    point=point.assign(est_FFprime_profile=
        (point["normalised_inductive_plasma_current_profile"]*point["total_current_profile_scale_fac"]
             -point["major_radius"].values*pprime)* (mu_0*point["major_radius"].values)
    ) #first point in brackets: A/m^2, second point in brackets: [m]*[Pa/Wb]=[m]*[kg⋅m−1⋅s−2/(kg⋅m2⋅s−2⋅A−1)]
    #                                                                       =[/(m2⋅A−1)]=A/m^2 good
    # total units: (A/m^2)*mu_0*m = mu_0*A/m

    return point

#tested
def get_pprime_popcon(point,normalised=False):
    #derivative with respect to poloidal flux
    psi_norm=np.array(point.psi_normalised.values)
    pressure_spln=make_interp_spline(psi_norm,np.array(point.pressure_profile.pint.to(ureg.Pa).values),k=3)

    pressure_gradient_points=np.zeros(len(psi_norm))
    for i in range(len(psi_norm)):
        pressure_gradient_points[i]=pressure_spln(psi_norm[i],1)

    if normalised:
        return pressure_gradient_points/max(abs(pressure_gradient_points))
    
    return pressure_gradient_points * point.est_total_poloidal_flux.values #units: Pa/Wb

#tested but not confirmed
def estimate_FFprime(point,pprime):
    #Converts from J_toroidal to FF' using Grad-Shafranov equation (also see OpenFusionToolkit)
    # assuming average flux surface major radius = major radius.
    jtor=np.array(point.normalised_inductive_plasma_current_profile.values*point.total_current_profile_scale_fac.values)

    FFprime_est = (jtor -  point.major_radius.values * (-pprime)) * (mu_0*point.major_radius.values)

    return FFprime_est

#tested
def est_total_poloidal_flux(point):
    #using equ 6.5 in IdealMHD, plus the assumption that |Bp| is uniform in the poloidal plane, such that
    # Bz ~ enclosed_total_toroidal_current/flux_surf_area, and the assumption that the total plasma current
    # density is equal to the toroidal plasma current density
    minor_radius_values=point.rho.values*point.minor_radius.values
    major_radius_values=minor_radius_values+point.major_radius.values
    est_flux_surf_circumference=2 * np.pi * minor_radius_values * (1 + 0.55 * (point.areal_elongation.values - 1))

    enclosed_total_toroidal_current_integration_spline=CubicSpline(minor_radius_values,(2*np.pi*point.elongation_psi95.values)*minor_radius_values*point.total_current_profile_scale_fac.values*point.normalised_inductive_plasma_current_profile.values)

    enclosed_total_toroidal_current=np.zeros(len(minor_radius_values))
    for i in range(len(minor_radius_values)):
        enclosed_total_toroidal_current[i]=enclosed_total_toroidal_current_integration_spline.integrate(min(minor_radius_values), minor_radius_values[i], extrapolate=False)

    est_flux_surf_circumference = np.array(est_flux_surf_circumference)
    est_flux_surf_circumference[0]=1e-10
    psi_pol_int_spline=CubicSpline(major_radius_values,mu_0*major_radius_values*enclosed_total_toroidal_current/est_flux_surf_circumference)

    total_poloidal_flux=psi_pol_int_spline.integrate(min(major_radius_values),max(major_radius_values),extrapolate=False)

    return total_poloidal_flux #Wb::: Int(dR)mu0*[m]*[A]/[m]=Int[dR]mu0*[A]=mu0*[A]*[m]=(kg*m)/(s^2*A^2)*A*m=(kg*m^2)/(s^2*A) = Wb = (kg·m^2)/(s^2·A)

#tested, less accurate... (assumes uniform current density)
def est_total_poloidal_flux2(point):
    #using equ 6.5 in IdealMHD, and the assumptions that the toroidal plasma current density 
    # is uniform in the poloidal plane, and integrated equals the total plasma current current
    minor_radius_values=point.rho*point.minor_radius
    major_radius_values=minor_radius_values+point.major_radius
    est_flux_surf_circumference=2 * np.pi * minor_radius_values * (1 + 0.55 * (point.areal_elongation - 1))

    enclosed_total_toroidal_current=point.psi_normalised*point.plasma_current.pint.to(ureg.A).values

    est_flux_surf_circumference = np.array(est_flux_surf_circumference)
    est_flux_surf_circumference[0]=1e-10
    psi_pol_int_spline=CubicSpline(major_radius_values,mu_0*major_radius_values*enclosed_total_toroidal_current/est_flux_surf_circumference)

    total_poloidal_flux=psi_pol_int_spline.integrate(min(major_radius_values),max(major_radius_values),extrapolate=False)

    return total_poloidal_flux #Wb::: 

#tested
#max_surface_area=350 $5B scaling based off manta analysis, min_surface_area based on SPARC
def gen_random_inputs(random_length,
                    max_surface_area=350, #5$B upfront based off MANTA study
                    min_surface_area=60, #SPARC-like
                    R0_min=1.5,R0_max=5.0, #SPARC-like vs big lol
                    eps_min=0.23,eps_max=0.8,
                    tri_psi95_min=-0.1,tri_psi95_max=0.55, #any more negative and we can be sure H-mode is inaccessible. Top limit based on triangularity of separatrix>triangularity at psi95
                    elongation_min=1.0,
                    current_frac_min=0.3,current_frac_max=1.0):

    random.seed()
    orig_random_length=random_length
    
    a = np.zeros(shape=(random_length,8))
    
    ndel=0
    for i in range(random_length):
        major_radius=random.uniform(R0_min,R0_max)
        inverse_aspect_ratio=random.uniform(eps_min,eps_max)
        triangularity_psi95=random.uniform(tri_psi95_min,tri_psi95_max)
        elongation_psi95=random.uniform(elongation_min,calc_elongation_psi95_from_Song_(triangularity_psi95))
        plasma_current_fraction=random.uniform(current_frac_min,current_frac_max)
        magnetic_field_on_axis = field_on_axis(major_radius,inverse_aspect_ratio)  
        plasma_current=calc_plasma_current_from_marginally_stable_current_fraction(
            plasma_current_fraction,
            magnetic_field_on_axis ,
            major_radius,
            inverse_aspect_ratio, 
            elongation_psi95)
        
        surface_area=calc_plasma_surface_area(major_radius,elongation_psi95,inverse_aspect_ratio)
        if(surface_area>max_surface_area or surface_area<min_surface_area or magnetic_field_on_axis<0.0):
            continue

        a[ndel,:]=np.array([major_radius,inverse_aspect_ratio,triangularity_psi95,elongation_psi95,plasma_current,magnetic_field_on_axis,plasma_current_fraction,surface_area])
        ndel+=1
        
    #display(a)
    a=a[:-(random_length-ndel), :]

    return a

#tested
def gen_random_inputs_current_backdown(max_surface_area=350, #5$B upfront based off MANTA study
                    min_surface_area=60, #SPARC-like
                    R0_min=1.5,R0_max=5.0, #SPARC-like vs big lol
                    eps_min=0.23,eps_max=0.8, 
                    tri_psi95_min=-0.1,tri_psi95_max=0.55, #any more negative and we can be sure H-mode is inaccessible. Top limit based on triangularity of separatrix>triangularity at psi95
                    elongation_min=1.0,
                    current_frac_min=0.29999, current_frac_max=1.0, current_frac_jump=0.1):
    #random.seed() seed random before calling this function...
    val_found=False
    infflag=0

    current_frac_vec=np.arange(current_frac_max,current_frac_min,step=-current_frac_jump)
    num_samples=len(current_frac_vec)
    a = np.zeros(shape=(num_samples,8))

    #initialising variables outside loop
    major_radius=0
    inverse_aspect_ratio=0
    triangularity_psi95=0
    elongation_psi95=0
    magnetic_field_on_axis=0
    surface_area=0

    while val_found==False and infflag<100000:
        major_radius=random.uniform(R0_min,R0_max)
        inverse_aspect_ratio=random.uniform(eps_min,eps_max)
        triangularity_psi95=random.uniform(tri_psi95_min,tri_psi95_max)
        elongation_psi95=random.uniform(elongation_min,calc_elongation_psi95_from_Song_(triangularity_psi95))

        magnetic_field_on_axis=field_on_axis(major_radius,inverse_aspect_ratio)
        surface_area=calc_plasma_surface_area(major_radius,elongation_psi95,inverse_aspect_ratio)

        if(surface_area>max_surface_area or surface_area<min_surface_area or magnetic_field_on_axis<0.0):
            infflag+=1
            continue
        else:
            val_found=True

    if not val_found:
        Warning("gen_random_inputs_current_backdown requirements cant find a single case in 10^5 samples, check if they're valid.")
    
    for i in range(num_samples):
        plasma_current_fraction=current_frac_vec[i]
          
        plasma_current=calc_plasma_current_from_marginally_stable_current_fraction(
            plasma_current_fraction,
            magnetic_field_on_axis ,
            major_radius,
            inverse_aspect_ratio, 
            elongation_psi95)
        
        a[i,:]=np.array([major_radius,inverse_aspect_ratio,triangularity_psi95,elongation_psi95,plasma_current,magnetic_field_on_axis,plasma_current_fraction,surface_area])
        
    return a

#tested
def calc_elongation_psi95_from_Song_(triangularity_psi95):
    #Junhyuk Song et al 2021 Nucl. Fusion 61 096033
    delta_vals=[-0.1,0.1,0.3,0.4,0.5,0.6]
    elongation_psi95_vals=[1.95,2.2,1.97,1.9,1.75,1.65]
    return np.interp(triangularity_psi95,delta_vals,elongation_psi95_vals)

#tested
def calc_plasma_current_from_marginally_stable_current_fraction(
    plasma_current_fraction,
    magnetic_field_on_axis ,
    major_radius,
    inverse_aspect_ratio, 
    elongation_psi95):

    minor_radius=inverse_aspect_ratio*major_radius
    return plasma_current_fraction * inverse_aspect_ratio * (1.0 + elongation_psi95**2.0) * np.pi * minor_radius * magnetic_field_on_axis / (mu_0 * 2.0)

#tested
def field_on_axis(major_radius,inverse_aspect_ratio,magnetic_field_on_coil=23,magnet_to_plasma_distance=0.9,inboard_space=0.6): 
    minor_radius=inverse_aspect_ratio*major_radius
    coil_radius=major_radius-minor_radius-magnet_to_plasma_distance
    magnetic_field_on_axis = magnetic_field_on_coil*(coil_radius/major_radius)

    if coil_radius-inboard_space<0.0:
        return -1.0

    return magnetic_field_on_axis

#tested
def calc_plasma_surface_area(major_radius,elongation_psi95,inverse_aspect_ratio,elongation_ratio_areal_to_psi95=1.025):
    areal_elongation= elongation_psi95 * elongation_ratio_areal_to_psi95
    return 2.0 * np.pi * (major_radius**2.0) * inverse_aspect_ratio * areal_elongation * (np.pi + 2.0 - (np.pi - 2.0) * inverse_aspect_ratio)

#tested
def calc_plasma_surface_area_MANTA(major_radius,elongation_psi95,inverse_aspect_ratio,elongation_ratio_areal_to_psi95=1.025):
    areal_elongation= elongation_psi95 * elongation_ratio_areal_to_psi95
    return 2.0 * np.pi**2 * (major_radius)**2 * inverse_aspect_ratio * (1+areal_elongation)

#major_radius=1.85
#coil_radius=1.04
#12.2*(major_radius)/(coil_radius)

#untested
def save_gfile(mygs, MC_ind,out_path,batch_num,total_batches,gname='g_MCi',verbose=False,Jfrac=0.0,nr=700,nz=700,maxsteps=1000000,lcfs_pad=1e-6,ttol=1e-9,run_simple=True):

    dir_name=out_path+f"/eqdsks_b{batch_num}of{total_batches}/"
    dir_name=dir_name.replace("//", "/")
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    current_dir=os.getcwd()
    os.chdir(dir_name)

    if run_simple: #avoids a mystery seg fault at the expense of throwing an error if a single case fails
        if Jfrac==0.0:
            gname_tot = gname+f"{MC_ind}"
            gname_tot = gname_tot.replace("//", "/")
            TokaMaker.save_eqdsk(mygs,gname_tot
                            ,nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=maxsteps,ttol=ttol)
        else:
            Jfrac=str(np.round(Jfrac, decimals=2)).replace('.', 'p')
            gname_tot = gname+f"{MC_ind}_Jfrac{Jfrac}"
            gname_tot = gname_tot.replace("//", "/")
            TokaMaker.save_eqdsk(mygs,gname_tot
                            ,nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=maxsteps,ttol=ttol)
        os.chdir(current_dir)
        return gname_tot

    success=False
    failflag=0

    while not success and failflag<10:
        #try:
        if verbose:
            print(f"Trying lcfs_pad = {lcfs_pad}:")
        if Jfrac==0.0:
            gname_tot = gname+f"{MC_ind}"
            gname_tot = gname_tot.replace("//", "/")
            success=TokaMaker.save_eqdsk(mygs,gname_tot
                            ,nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=maxsteps,ttol=ttol)
        else:
            Jfrac=str(np.round(Jfrac, decimals=2)).replace('.', 'p')
            gname_tot = gname+f"{MC_ind}_Jfrac{Jfrac}"
            gname_tot = gname_tot.replace("//", "/")
            success=TokaMaker.save_eqdsk(mygs,gname_tot
                            ,nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=maxsteps,ttol=ttol)
        #except:
        lcfs_pad=lcfs_pad*10
        failflag+=1
        continue
        #success=True

    if success:
        if verbose:
            if Jfrac==0.0:
                print(f"saved eqdsk {MC_ind} with lcfs_pad = {lcfs_pad}")
            else:
                print(f"saved eqdsk MC_ind,Jfrac=({MC_ind},{Jfrac}) with lcfs_pad = {lcfs_pad}")
        os.chdir(current_dir)
        return gname_tot
    else:
        if Jfrac==0.0:
            print(f"ERROR SAVING EQDSK::: {MC_ind}")
        else:
            print(f"ERROR SAVING EQDSK::: MC_ind,Jfrac=({MC_ind},{Jfrac})")
        os.chdir(current_dir)
        return None
    os.chdir(current_dir)
    return

"""
def save_gfiles(mygs_batch_vec,J_BS_batch_vec,J_tot_batch_vec,MC_ind_batch_vec,out_path,batch_num,total_batches,nr=450,nz=450,maxsteps=1000000,lcfs_pad=1e-6,ttol=1e-14):
    assert len(mygs_batch_vec)==len(J_BS_batch_vec)
    if not os.path.isdir(out_path+f"/eqdsks_b{batch_num}of{total_batches}"):
        os.mkdir(out_path+f"/eqdsks_b{batch_num}of{total_batches}")

    for i in range(len(mygs_batch_vec)):
        success=False
        failflag=0

        MC_ind=MC_ind_batch_vec[i]
        while not success and failflag<10:
            #try:
            TokaMaker.save_eqdsk(mygs_batch_vec[i],out_path+f"/eqdsks_b{batch_num}of{total_batches}/g_MCi{MC_ind}"
                             ,nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=maxsteps,ttol=ttol)
            #except:
            #    lcfs_pad=lcfs_pad*10
            #    failflag+=1
            #    continue
            success=True

        if not success:
            print("Error while printing gfiles")

    np.savetxt(out_path+f"/eqdsks_b{batch_num}of{total_batches}/Jbs",np.array(J_BS_batch_vec), delimiter=",")
    np.savetxt(out_path+f"/eqdsks_b{batch_num}of{total_batches}/Jtots",np.array(J_tot_batch_vec), delimiter=",")
    np.savetxt(out_path+f"/eqdsks_b{batch_num}of{total_batches}/MCindexes",np.array(MC_ind_batch_vec), fmt='%5i', delimiter=",")

    return

#deprecated
def _save_gfiles(batch_dataset_vec,out_path,batch_num,total_batches,nr=450,nz=450,maxsteps=1000000,lcfs_pad=1e-6,ttol=1e-14):
    if not os.path.isdir(out_path+f"/eqdsks_b{batch_num}of{total_batches}"):
        os.mkdir(out_path+f"/eqdsks_b{batch_num}of{total_batches}")

    num_datasets=len(batch_dataset_vec)

    MC_ind_batch_vec=np.zeros(num_datasets)

    for i in range(num_datasets):
        success=False
        failflag=0

        MC_ind=batch_dataset_vec[i].MC_index.values
        MC_ind_batch_vec[i]=MC_ind
        while not success and failflag<10:
            try:
                TokaMaker.save_eqdsk(batch_dataset_vec[i].mygs.values,out_path+f"/eqdsks_b{batch_num}of{total_batches}/g_MCi{MC_ind}"
                             ,nr=nr,nz=nz,lcfs_pad=lcfs_pad,maxsteps=maxsteps,ttol=ttol)
            except:
                lcfs_pad=lcfs_pad*10
                failflag+=1
                continue
            success=True

        if not success:
            print("Error while printing gfiles")

        batch_dataset_vec[i]=batch_dataset_vec[i].drop_vars(["mygs"])

    np.savetxt(out_path+f"/eqdsks_b{batch_num}of{total_batches}/MCindexes",np.array(MC_ind_batch_vec), fmt='%5i', delimiter=",")

    return batch_dataset_vec
"""
#end deprecated
