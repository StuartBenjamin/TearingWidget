from scipy.interpolate import CubicSpline, make_interp_spline
import numpy as np
import xarray as xr
import os
import sys
import math
import shutil
import time
import pandas as pd

import subprocess

#Codes to read the dcon and rdcon data
#Codes to record the dcon and rdcon data, add them to the xarray in gen_n_cases [tonight???]
# returns stab_metadata, n (toroidal mode) dimension xarray telling us if dcon, rdcon ran, ideal stability, and truncation point info
# returns MRE_combined, n x m (toroidal x poloidal mode) dimension xarray giving MRE data at each rational surface and also delta' info if rdcon sucessfully ran

#Current function: slow but thorough
#Best option...: limit the delta'
#Wishlist::: time to fiddle with the parameters and make sure they are all good...

#   delta_mhigh=16,delta_mlow=16
#   If we leave it like this, its slower, but we have flexibility to truncate the rdcon case differently to the dcon case
def run_DCON_on_equilibrium(eq_filename,newq0=0,qlow=1.015,gse_err_logtol=-1.5,working_dir='',dcon_executable_dir='',rdcon_executable_dir='',
                                grid_type="""'ldp'""",mtheta=1024,mpsi=512,mpsi_diagnose=1024,psi_search_range=0.02,psihigh=0.9999,nmin=1,nmax=4,rdcon_nlim=100,**kwargs):
    if len(working_dir)>0:
        os.chdir(working_dir)

    assert nmin != 0 and nmin*nmax>0 #makes sure we don't go through n=0

    #create local copies of our executables!!
    #we remove any pre-existing copies because they can cause instant aborts (zsh killed)
    if len(dcon_executable_dir)>0:
        if os.path.isfile(working_dir+"/dcon"):
            os.remove(working_dir+"/dcon")
        shutil.copy(dcon_executable_dir+"/dcon", working_dir+"/")
    if len(rdcon_executable_dir)>0:
        if os.path.isfile(working_dir+"/rdcon"):
            os.remove(working_dir+"/rdcon")
        shutil.copy(rdcon_executable_dir+"/rdcon", working_dir+"/")

    #Starting with a run to extract surface terms only..., this very quick (2s)
    #If it fails, returns instantly
    write_dcon_inputs(working_dir,eq_filename,
        nn=1,
        mpsi=256,
        etol=1e-10,
        mtheta=512,
        psihigh=0.995,
        gse_err_logtol=1, #inactive
        bal_flag='f',
        mat_flag='f',
        ode_flag='f',
        mer_flag='t', #true
        vac_flag='f',
        sas_flag='f', 
        psiedge=1.0,
        psi_search_range=0.0,
        qlow=0,
        qhigh=1e3,
        sing_start=0,
        bin_euler='f',
        netcdf_out='f',
        bin_eq_2d='f',
        gse_flag='f',
        grid_type="""'ldp'""",
        psilow=1e-4,
        **kwargs)
    dcon_run_surf = subprocess.run(working_dir+"/dcon")
    if dcon_run_surf.returncode!=0: #quick exit if this basic scan fails...
        surf_ran=False
        #return False, nones...
        equil_dat_xr=prof_dat_xr=edge_dat_xr=gse_dat_xr=MRE_xr=None
        return False,equil_dat_xr,prof_dat_xr,edge_dat_xr,gse_dat_xr,MRE_xr
    #reading dcon profile data, equilibrium data
    prof_dat_xr = pd.read_csv(working_dir+'/prof_dat.csv', dtype={'ipsis': 'int'}).to_xarray().set_index(index="ipsis").rename({'index': 'ipsis'})#, dtype={'ipsis': 'int'})
    equil_dat_xr = pd.read_csv(working_dir+'/equil_dat.csv', dtype={'mer_pass': 'int'}).to_xarray().squeeze(drop=True)#, dtype={'mer_pass': 'int'})
    equil_dat_xr=equil_dat_xr.assign(rdcon_nlim=rdcon_nlim)
    
    mer_pass=equil_dat_xr.mer_pass.values>0
    surf_ran=True

    #along with surf_ran, ns_dcon_ran and ns_rdcon_ran tell us if dcon/rdcon ran
    ns_dcon_ran = []
    ns_rdcon_ran = []
    #ns_dcon_ran.append(0) #needed to ensure continuity of xarray coordinates
    ns_rdcon_ran.append(0) #needed to ensure continuity of xarray coordinates

    #Initialise variables that will be re-written over the course of the dcon/rdcon n loop
    first_case_has_ran=False #at least one dcon run -> gives us our edge behaviour
    GSE_good=True #GSE tolerances are acceptable...
    dcon_pass=True #Ideal stability (at least past m=1 surface)
    free_pass=True #dW for an ideal, conformal shaped wall at 20 x minor radius (at least past m=1 surface)

    #stab_metadata__=None
    MRE_combined_vec=[] #vector of things that will have m,n behaviour

    for i in range(nmin,nmax+1):
        if not first_case_has_ran:
            print("Getting edge data from first run")
            #Important prints are 
                #First dcon run:::, differs from later cases by needing high res to truncate equil, also prints gsetol data, edge data
                #Subsequent runs:::, just run dcon...
                #   If first case, gal_solve = false...
                #Two generic printing functions to fix:
                #   dcon_dat [DONE]
                #   MRE_dat 
            write_dcon_inputs(working_dir,eq_filename,
                        nn=i,
                        mpsi=mpsi_diagnose,
                        etol=1e-10,
                        mtheta=mtheta,
                        psihigh=psihigh,
                        gse_err_logtol=gse_err_logtol,                #caps psihigh to keep error small, logs equil error tolerances
                        bal_flag='f',
                        mat_flag='f',
                        ode_flag='t',                       #true
                        mer_flag='f', 
                        vac_flag='f',                       #false since running psi_search_range
                        sas_flag='f', 
                        psiedge=1.0,                        #inactive
                        psi_search_range=psi_search_range,  #scans edge for max dW, saves as psilim
                        qlow=qlow,
                        newq0=newq0,
                        qhigh=1e3,
                        sing_start=0,
                        bin_euler='f',
                        netcdf_out='f',
                        bin_eq_2d='f',
                        gse_flag='f',
                        grid_type=grid_type,
                        psilow=1e-4,
                        **kwargs)
        else: #Don't run dcon, just run rdcon with vac_flag=true, and if stable, run RDCON
            write_dcon_inputs(working_dir,eq_filename,
                        nn=i,
                        mpsi=mpsi,
                        etol=1e-10,
                        mtheta=mtheta,
                        psihigh=psilim,
                        gse_err_logtol=1,       #inactive
                        bal_flag='f',
                        mat_flag='f',
                        ode_flag='t', #true
                        mer_flag='f', 
                        vac_flag='t', #true
                        sas_flag='f', 
                        psiedge=1.0,            #inactive
                        psi_search_range=0.0,   #inactive
                        psilow=max(qlow_psi,1e-4),
                        newq0=newq0,
                        qlow=0,                 #inactive
                        qhigh=1e3,
                        sing_start=0,
                        bin_euler='f',
                        netcdf_out='f',
                        bin_eq_2d='f',
                        gse_flag='f',
                        grid_type="""'ldp'""",
                        **kwargs)

        dcon_run = subprocess.run(working_dir+"/dcon")
        if dcon_run.returncode==0: 
            ns_dcon_ran.append(i)
            DCON_data_Xr = pd.read_csv(working_dir+'/dcon_dat.csv', dtype={'n': 'int'}).to_xarray().set_coords(names=["n"]).set_xindex(coord_names=["n"]).drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)
            MRE_data_Xr = pd.read_csv(working_dir+'/MRE_dat.csv',dtype={'m': 'int', 'n': 'int'}).to_xarray().drop_vars(names=["n"]).set_index(index="m").rename({'index': 'm'})
            MRE_data_Xr = MRE_data_Xr.assign(dcon_nzero=DCON_data_Xr.dcon_nzero.values[0],dW_total=DCON_data_Xr.dW_total.values[0])

            if not first_case_has_ran:
                psilim=DCON_data_Xr.psilim.values[0]
                qlow_psi=DCON_data_Xr.qlow_psi.values[0]

                GSE_good = psilim > psihigh-psi_search_range #big gs errors will truncate below the region we're tring to search
                #for j in range(len(GSE_data_DF.psifac)):
                #    if GSE_data_DF.psifac.values[j]<psilim and GSE_data_DF.errori.values[j] > gsei_tol:
                #        GSE_good=False

                #adding q bound data to equil_dat_xr, how has q<1 & gs error caused the computed region to shrink
                equil_dat_xr=equil_dat_xr.assign(GSE_good=GSE_good,qlow=DCON_data_Xr.qlow.values[0],qlow_psi=DCON_data_Xr.qlow_psi.values[0],qlow_rho=DCON_data_Xr.qlow_rho.values[0],psilim=DCON_data_Xr.psilim.values[0],qmax=DCON_data_Xr.qmax.values[0])
                #adding edge dW trace data
                edge_dat_xr = pd.read_csv(working_dir+'/dW_edge.csv').to_xarray().rename({'index': 'edge_index'})
                #adding edge gs error psi data
                gse_dat_xr = pd.read_csv(working_dir+'/gse_psi_tols.csv').to_xarray().rename({'index': 'gse_index'})

            first_case_has_ran=True

            dcon_pass=(DCON_data_Xr.dcon_nzero.values[0]==0)
            free_pass=(DCON_data_Xr.dW_total.values[0]>0)
            if dcon_pass and free_pass and i<rdcon_nlim:
                print('running RDCON')
                write_rdcon_inputs(working_dir,eq_filename,
                        nn=i,
                        mpsi=mpsi,
                        etol=1e-10,
                        mtheta=mtheta,
                        dx0=5e-4,
                        cutoff=10, 
                        bin_delmatch='f',
                        bal_flag='f',
                        ode_flag='f',
                        vac_flag='f',
                        gal_flag='t',
                        dump_MRE_data='f',
                        gse_flag='f',
                        sas_flag='f',
                        cyl_flag='f',
                        psihigh=psilim,
                        bin_eq_2d='f',
                        psilow=max(qlow_psi,1e-4),
                        newq0=newq0,
                        grid_type="""'ldp'""",
                        **kwargs)
                rdcon_run = subprocess.run(working_dir+"/rdcon")
                if rdcon_run.returncode==0:
                    ns_rdcon_ran.append(i)
                    print('merging Deltaprime data')
                    DeltaPrimes_Xr = pd.read_csv(working_dir+'/Single_Helicity_DeltaPrime.csv',dtype={'m': 'int'}).to_xarray().drop_vars(names=['psifac','n']).set_index(index="m").rename({'index': 'm'})
                    MRE_data_Xr=xr.merge([MRE_data_Xr,DeltaPrimes_Xr])

            MRE_combined_vec.append(MRE_data_Xr.assign(n=i).set_coords('n'))

    if len(MRE_combined_vec)>0:
        MRE_xr=xr.concat(MRE_combined_vec,dim="n")

    #Combining all info into a single xarray:
    equil_dat_xr=equil_dat_xr.assign(ns_dcon_ran=ns_dcon_ran,ns_rdcon_ran=ns_rdcon_ran)
    equil_dat_xr=equil_dat_xr.merge(prof_dat_xr)
    equil_dat_xr=equil_dat_xr.merge(edge_dat_xr)
    equil_dat_xr=equil_dat_xr.merge(gse_dat_xr)
    equil_dat_xr=equil_dat_xr.merge(MRE_xr)

    return (surf_ran and len(ns_dcon_ran)>0),(surf_ran and len(ns_rdcon_ran)>0),equil_dat_xr

#THIS IS THE ONE I'M GOING TO USE
def run_DCON_on_equilibrium2(eq_filename,newq0=0,qlow=1.015,gse_err_logtol=-1.5,working_dir='',dcon_executable_dir='',rdcon_executable_dir='',run_merscan=True,
                                eq_type="""'efit_tokamaker'""",grid_type_diagnose="""'ldp'""",mtheta=1024,mpsi=512,mpsi_diagnose=1024,psi_search_range=0.05,psihigh=0.9999,nmin=1,nmax=4,rdcon_nlim=100,special_debug=False,**kwargs):
    if len(working_dir)>0:
        os.chdir(working_dir)

    assert nmin != 0 and nmin*nmax>0 #makes sure we don't go through n=0

    #create local copies of our executables!!
    #we remove any pre-existing copies because they can cause instant aborts (zsh killed)
    if len(dcon_executable_dir)>0:
        if os.path.isfile(working_dir+"/dcon"):
            os.remove(working_dir+"/dcon")
        shutil.copy(dcon_executable_dir+"/dcon", working_dir+"/")
    if len(rdcon_executable_dir)>0:
        if os.path.isfile(working_dir+"/rdcon"):
            os.remove(working_dir+"/rdcon")
        shutil.copy(rdcon_executable_dir+"/rdcon", working_dir+"/")

    #Starting with a run to extract surface terms only..., this very quick (2s)
    #If it fails, returns instantly
    write_dcon_inputs(working_dir,eq_filename,
        nn=1,
        mpsi=256,
        etol=1e-9,
        mtheta=512,
        psihigh=0.995,#0.99983,
        gse_err_logtol=1, #inactive
        bal_flag='f',
        mat_flag='f',
        ode_flag='f',
        mer_flag='t', #true
        vac_flag='f',
        sas_flag='f', 
        psiedge=1.0,
        psi_search_range=0.0,
        qlow=0,
        qhigh=1e3,
        sing_start=0,
        bin_euler='f',
        netcdf_out='f',
        bin_eq_2d='f',
        gse_flag='f',
        grid_type="""'ldp'""",
        psilow=1e-4,
        **kwargs)
    dcon_run_surf = subprocess.run(working_dir+"/dcon")
    if dcon_run_surf.returncode!=0: #quick exit if this basic scan fails...
        print("""%%%%%%%%%%%%%%%%%%%%%%%%dcon run failed%%%%%%%%%%%%%%%%%%%%%%%%%""")
        surf_ran=False
        #return False, nones...
        equil_dat_xr=prof_dat_xr=edge_dat_xr=gse_dat_xr=MRE_xr=None
        return False,equil_dat_xr,prof_dat_xr,edge_dat_xr,gse_dat_xr,MRE_xr
    #reading dcon profile data, equilibrium data
    prof_dat_xr = pd.read_csv(working_dir+'/prof_dat.csv', dtype={'ipsis': 'int'}).to_xarray().set_index(index="ipsis").rename({'index': 'ipsis'})#, dtype={'ipsis': 'int'})
    equil_dat_xr = pd.read_csv(working_dir+'/equil_dat.csv', dtype={'mer_pass': 'int'}).to_xarray().squeeze(drop=True)#, dtype={'mer_pass': 'int'})
    equil_dat_xr=equil_dat_xr.assign(rdcon_nlim=rdcon_nlim)

    mer_pass=equil_dat_xr.mer_pass.values>0
    surf_ran=True

    #along with surf_ran, ns_dcon_ran and ns_rdcon_ran tell us if dcon/rdcon ran
    ns_dcon_ran = []
    ns_rdcon_ran = []
    #ns_dcon_ran.append(0) #needed to ensure continuity of xarray coordinates
    ns_rdcon_ran.append(0) #needed to ensure continuity of xarray coordinates

    #Initialise variables that will be re-written over the course of the dcon/rdcon n loop
    first_case_has_ran=False #at least one dcon run -> gives us our edge behaviour
    GSE_good=True #GSE tolerances are acceptable...
    dcon_pass=True #Ideal stability (at least past m=1 surface)
    free_pass=True #dW for an ideal, conformal shaped wall at 20 x minor radius (at least past m=1 surface)

    #stab_metadata__=None
    MRE_combined_vec=[] #vector of things that will have m,n behaviour

    dcon_nzeros=[]
    dW_totals=[]
    dW_plasmas=[]
    dW_vacuums=[]

    for i in range(nmin,nmax+1):
        if not first_case_has_ran:
            print("Getting edge data from first run")
            #Important prints are 
                #First dcon run:::, differs from later cases by needing high res to truncate equil, also prints gsetol data, edge data
                #Subsequent runs:::, just run dcon...
                #   If first case, gal_solve = false...
                #Two generic printing functions to fix:
                #   dcon_dat [DONE]
                #   MRE_dat 
            write_dcon_inputs(working_dir,eq_filename,
                        nn=i,
                        mpsi=mpsi_diagnose,
                        etol=1e-10,
                        mtheta=mtheta,
                        psihigh=psihigh,
                        gse_err_logtol=gse_err_logtol,      #caps psihigh to keep error small, logs equil error tolerances
                        bal_flag='f',
                        mat_flag='f',
                        ode_flag='t',                       #true
                        mer_flag='f', 
                        vac_flag='f',                       #false since running psi_search_range
                        sas_flag='f', 
                        psiedge=1.0,                        #inactive
                        psi_search_range=psi_search_range,  #scans edge for max dW, saves as psilim
                        qlow=qlow,
                        newq0=newq0,
                        qhigh=1e3,
                        sing_start=0,
                        bin_euler='f',
                        netcdf_out='f',
                        bin_eq_2d='f',
                        gse_flag='f',
                        grid_type=grid_type_diagnose,
                        eq_type=eq_type,
                        psilow=1e-4,
                        **kwargs)
            dcon_run = subprocess.run(working_dir+"/dcon")
            if dcon_run.returncode==0: 
                ns_dcon_ran.append(i)
                DCON_data_Xr = pd.read_csv(working_dir+'/dcon_dat.csv', dtype={'n': 'int'}).to_xarray().set_coords(names=["n"]).set_xindex(coord_names=["n"]).drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)
                MRE_data_Xr = pd.read_csv(working_dir+'/MRE_dat.csv',dtype={'m': 'int', 'n': 'int'}).to_xarray().drop_vars(names=["n"]).set_index(index="m").rename({'index': 'm'})
                #MRE_data_Xr = MRE_data_Xr.assign(dcon_nzero=DCON_data_Xr.dcon_nzero.values[0],dW_total=DCON_data_Xr.dW_total.values[0])
                dcon_nzeros.append(DCON_data_Xr.dcon_nzero.values[0])
                dW_totals.append(DCON_data_Xr.dW_total.values[0])
                dW_plasmas.append(DCON_data_Xr.dW_plasma.values[0])
                dW_vacuums.append(DCON_data_Xr.dW_vacuum.values[0])

                #FIRST RUN SPECIFIC STUFF:
                psilim=DCON_data_Xr.psilim.values[0]
                qlow_psi=DCON_data_Xr.qlow_psi.values[0]
                GSE_good = psilim > psihigh-psi_search_range #big gs errors will truncate below the region we're tring to search

                #adding q bound data to equil_dat_xr, how has q<1 & gs error caused the computed region to shrink
                equil_dat_xr=equil_dat_xr.assign(GSE_good=GSE_good,qlow=DCON_data_Xr.qlow.values[0],qlow_psi=DCON_data_Xr.qlow_psi.values[0],qlow_rho=DCON_data_Xr.qlow_rho.values[0],psilim=DCON_data_Xr.psilim.values[0],qmax=DCON_data_Xr.qmax.values[0])
                #adding edge dW trace data
                edge_dat_xr = pd.read_csv(working_dir+'/dW_edge.csv',dtype={'dW_e': 'float', 'q_e': 'float', 'psifac_e': 'float'}).to_xarray().rename({'index': 'edge_index'})
                #adding edge gs error psi data
                gse_dat_xr = pd.read_csv(working_dir+'/gse_psi_tols.csv').to_xarray().rename({'index': 'gse_index'})

                first_case_has_ran=True

                #Time to prepare rdcon
                dcon_pass=(DCON_data_Xr.dcon_nzero.values[0]==0)
                free_pass=(DCON_data_Xr.dW_total.values[0]>0)

                if dcon_pass and free_pass:
                    print('running RDCON')
                    write_rdcon_inputs(working_dir,eq_filename,
                            nn=i,
                            mpsi=mpsi,
                            etol=1e-10,
                            mtheta=mtheta,
                            psihigh=psilim,
                            psilow=max(qlow_psi,1e-4),
                            dx0=5e-4,
                            cutoff=10, 
                            newq0=newq0,
                            grid_type="""'ldp'""",
                            bin_delmatch='f',
                            bal_flag='f',
                            ode_flag='f', #false since ran dcon above for this nn
                            vac_flag='f', #false since ran dcon above for this nn
                            gal_flag='t', #true
                            dump_MRE_data='f',
                            gse_flag='f', #false, ran once by dcon
                            sas_flag='f',
                            cyl_flag='f',
                            bin_eq_2d='f',
                            eq_type=eq_type,
                            **kwargs)
                    rdcon_run = subprocess.run(working_dir+"/rdcon")
                    if rdcon_run.returncode==0:
                        ns_rdcon_ran.append(i)
                        print('merging Deltaprime data')
                        DeltaPrimes_Xr = pd.read_csv(working_dir+'/Single_Helicity_DeltaPrime.csv',dtype={'m': 'int'}).to_xarray().drop_vars(names=['psifac','n']).set_index(index="m").rename({'index': 'm'})
                        print(DeltaPrimes_Xr.Re_DeltaPrime.values)
                        if special_debug:
                            #MRE_data_Xr.to_csv("MRE_data_Xr_Emergency")
                            #DeltaPrimes_Xr.to_csv("DeltaPrimes_Xr_Emergency")
                            equil_dat_xr=equil_dat_xr.assign(ns_dcon_ran=ns_dcon_ran,ns_rdcon_ran=ns_rdcon_ran)
                            equil_dat_xr=equil_dat_xr.assign(dcon_nzero=equil_dat_xr["ns_dcon_ran"]*0+dcon_nzeros,dW_total=equil_dat_xr["ns_dcon_ran"]*0+dW_totals)
                            equil_dat_xr=equil_dat_xr.assign(dW_plasma=equil_dat_xr["ns_dcon_ran"]*0+dW_plasmas,dW_vacuum=equil_dat_xr["ns_dcon_ran"]*0+dW_vacuums)
                            if run_merscan:
                                equil_dat_xr=equil_dat_xr.merge(prof_dat_xr)
                            equil_dat_xr=equil_dat_xr.merge(edge_dat_xr)
                            equil_dat_xr=equil_dat_xr.merge(gse_dat_xr)
                            try:
                                equil_dat_xr.to_csv("equil_dat_xr_Emergency")
                                MRE_data_Xr=xr.merge([MRE_data_Xr,DeltaPrimes_Xr])
                                MRE_xr=xr.concat(MRE_data_Xr,dim="n")
                                equil_dat_xr=equil_dat_xr.merge(MRE_xr)
                            except:
                                return MRE_data_Xr,DeltaPrimes_Xr,equil_dat_xr
                        MRE_data_Xr=xr.merge([MRE_data_Xr,DeltaPrimes_Xr])
        else: #now just running rdcon on its own
            write_rdcon_inputs(working_dir,eq_filename,
                    nn=i,
                    mpsi=mpsi,
                    etol=1e-10,
                    mtheta=mtheta,
                    psihigh=psilim,
                    psilow=max(qlow_psi,1e-4),
                    dx0=5e-4,
                    cutoff=10, 
                    newq0=newq0,
                    grid_type="""'ldp'""",
                    bin_delmatch='f',
                    bal_flag='f',
                    ode_flag='t', #true
                    vac_flag='t', #true
                    gal_flag='t', #true
                    dump_MRE_data='f',
                    gse_flag='f', #false, ran once by dcon
                    sas_flag='f',
                    cyl_flag='f',
                    bin_eq_2d='f',
                    eq_type=eq_type,
                    **kwargs)
            rdcon_run = subprocess.run(working_dir+"/rdcon")
            if rdcon_run.returncode==0:
                ns_dcon_ran.append(i)
                ns_rdcon_ran.append(i)
                DCON_data_Xr = pd.read_csv(working_dir+'/dcon_dat.csv', dtype={'n': 'int'}).to_xarray().set_coords(names=["n"]).set_xindex(coord_names=["n"]).drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)
                MRE_data_Xr = pd.read_csv(working_dir+'/MRE_dat.csv',dtype={'m': 'int', 'n': 'int'}).to_xarray().drop_vars(names=["n"]).set_index(index="m").rename({'index': 'm'})
                #MRE_data_Xr = MRE_data_Xr.assign(dcon_nzero=DCON_data_Xr.dcon_nzero.values[0],dW_total=DCON_data_Xr.dW_total.values[0])
                dcon_nzeros.append(DCON_data_Xr.dcon_nzero.values[0])
                dW_totals.append(DCON_data_Xr.dW_total.values[0])
                dW_plasmas.append(DCON_data_Xr.dW_plasma.values[0])
                dW_vacuums.append(DCON_data_Xr.dW_vacuum.values[0])

                dcon_pass=(DCON_data_Xr.dcon_nzero.values[0]==0)
                free_pass=(DCON_data_Xr.dW_total.values[0]>0)
                if dcon_pass and free_pass:
                    DeltaPrimes_Xr = pd.read_csv(working_dir+'/Single_Helicity_DeltaPrime.csv',dtype={'m': 'int', 'n': 'int'}).to_xarray().drop_vars(names=['psifac','n']).set_index(index="m").rename({'index': 'm'})
                    MRE_data_Xr=xr.merge([MRE_data_Xr,DeltaPrimes_Xr])
            
        MRE_combined_vec.append(MRE_data_Xr.assign(n=i).set_coords('n'))

    if len(MRE_combined_vec)>0:
        MRE_xr=xr.concat(MRE_combined_vec,dim="n")

    #Combining all info into a single xarray:
    equil_dat_xr=equil_dat_xr.assign(ns_dcon_ran=ns_dcon_ran,ns_rdcon_ran=ns_rdcon_ran)
    equil_dat_xr=equil_dat_xr.assign(dcon_nzero=equil_dat_xr["ns_dcon_ran"]*0+dcon_nzeros,dW_total=equil_dat_xr["ns_dcon_ran"]*0+dW_totals)
    equil_dat_xr=equil_dat_xr.assign(dW_plasma=equil_dat_xr["ns_dcon_ran"]*0+dW_plasmas,dW_vacuum=equil_dat_xr["ns_dcon_ran"]*0+dW_vacuums)
    if run_merscan:
        equil_dat_xr=equil_dat_xr.merge(prof_dat_xr)
    equil_dat_xr=equil_dat_xr.merge(edge_dat_xr)
    equil_dat_xr=equil_dat_xr.merge(gse_dat_xr)
    equil_dat_xr=equil_dat_xr.merge(MRE_xr)

    return (surf_ran and len(ns_dcon_ran)>0),(surf_ran and len(ns_rdcon_ran)>0),equil_dat_xr

def run_rdconMRE_on_equilibrium(eq_filename,working_dir='',rdcon_executable_dir='',
                                     nmin=1,nmax=8,
                                     **kwargs):
    if len(rdcon_executable_dir)>0:
        shutil.copy(rdcon_executable_dir+"/rdcon", working_dir+"/")

    for i in range(nmin,nmax+1):
        write_rdcon_inputs(working_dir,eq_filename,
                        nn=i,
                        dump_MRE_data='t',
                        gal_flag='t',
                        **kwargs)
        rdcon_run = subprocess.run(working_dir+"/rdcon")

        if rdcon_run.returncode==0:
            MRE_data_DF = pd.read_csv('MRE_data.csv')
    return 

def write_equil_in(working_dir,eq_filename,write_equil_filename='/equil.in',
        #EQUIL_CONTROL
        eq_type="""'efit'""",     #Type of the input 2D equilibrium file. Accepts efit, chease, fluxgrid, transp, jsolver, lar, sol, etc.
        jac_type="""'hamada'""",  # Working coordinate system for all DCON and GPEC calculations. Overrides individual powers. Accepts hamada, pest, boozer, equal_arc
        power_bp=0,         #del.B ~ B_p**power_bp * B**power_b / R**power_r
        power_b=0,          #del.B ~ B_p**power_bp * B**power_b / R**power_r
        power_r=0,          #del.B ~ B_p**power_bp * B**power_b / R**power_r
        grid_type="""'ldp'""",    #Radial grid packing of equilibrium quantities. Accepts rho, ldp, pow1, pow2, or original. ldp packs points near the core and edge. pow* packs near the edge.
        psilow=1e-4,        #Minimum value of psi, normalized from 0 to 1
        psihigh=0.993,      #Maximum value of psi, normalized from 0 to 1
        mpsi=256,           #Number of radial grid intervals for equilibrium quantities
        mtheta=512,         #Number of equally spaced poloidal grid intervals for all splines
        nstepd=100000,
        etol=1e-10,
        newq0=0,            #Grad-Shafranov solution invariant adjustment of the q profile to give the specified value of q at the axis. Default 0 uses input file value.
        use_classic_splines='f', # Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions
        input_only='f',      #Generate information about the input and then quit with no further calculation
        #EQUIL_OUTPUT
        gse_flag='f',       #Produces diagnostic output for accuracy of solution to Grad-Shafranov equation
                            #I'm just printing 
        out_eq_1d='f',      #Ascii output of 1D equilibrium file data
        bin_eq_1d='f',      #Binary output of 1D equilibrium file data
        out_eq_2d='f',      #Ascii output of 2D equilibrium file data
        bin_eq_2d='f',      #Binary output of 2D equilibrium file data (set true for GPEC)
        out_2d='f',         #Ascii output of processed 2D data
        bin_2d='f',         #Binary output of processed 2D data
        dump_flag='f',      #Binary dump of basic equilibrium data and 2D rzphi spline
        a_wall=20           #see vac.in description below
        ):

    f = open(working_dir+write_equil_filename, 'w')

    f.write('&EQUIL_CONTROL'+'\n')
    f.write('    eq_type='+eq_type +'\n') #Type of the input 2D equilibrium file. Accepts efit, chease, fluxgrid, transp, jsolver, lar, sol, etc.
    f.write('    eq_filename='+"""'"""+eq_filename+"""'"""'\n') #Path to input file

    f.write('    jac_type='+jac_type +'\n') # Working coordinate system for all DCON and GPEC calculations. Overrides individual powers. Accepts hamada, pest, boozer, equal_arc
    f.write('    power_bp='+str(power_bp)+'\n') #del.B ~ B_p**power_bp * B**power_b / R**power_r
    f.write('    power_b='+str(power_b) +'\n') #del.B ~ B_p**power_bp * B**power_b / R**power_r
    f.write('    power_r='+str(power_r) +'\n') #del.B ~ B_p**power_bp * B**power_b / R**power_r

    f.write('    grid_type='+grid_type +'\n') #Radial grid packing of equilibrium quantities. Accepts rho, ldp, pow1, pow2, or original. ldp packs points near the core and edge. pow* packs near the edge.
    f.write('    psilow='+str(psilow) +'\n') #Minimum value of psi, normalized from 0 to 1
    f.write('    psihigh='+str(psihigh)   +'\n') #Maximum value of psi, normalized from 0 to 1
    f.write('    mpsi='+str(mpsi)  +'\n') #Number of radial grid intervals for equilibrium quantities
    f.write('    mtheta='+str(mtheta)+'\n') #Number of equally spaced poloidal grid intervals for all splines
    f.write('    nstepd='+str(nstepd)+'\n')
    f.write('    etol='+str(etol)+'\n')
    f.write('    newq0='+str(newq0)   +'\n') #Grad-Shafranov solution invariant adjustment of the q profile to give the specified value of q at the axis. Default 0 uses input file value.
    f.write('    use_classic_splines ='+use_classic_splines +'\n') # Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions

    f.write('    input_only='+input_only+'\n') #Generate information about the input and then quit with no further calculation
    f.write('/'+'\n')
    f.write('&EQUIL_OUTPUT'+'\n')
    f.write('    gse_flag='+gse_flag+'\n') #Produces diagnostic output for accuracy of solution to Grad-Shafranov equation
    f.write('    out_eq_1d='+out_eq_1d +'\n') #Ascii output of 1D equilibrium file data
    f.write('    bin_eq_1d='+bin_eq_1d +'\n') #Binary output of 1D equilibrium file data
    f.write('    out_eq_2d='+out_eq_2d +'\n') #Ascii output of 2D equilibrium file data
    f.write('    bin_eq_2d='+bin_eq_2d +'\n') #Binary output of 2D equilibrium file data (set true for GPEC)
    f.write('    out_2d='+out_2d  +'\n') #Ascii output of processed 2D data
    f.write('    bin_2d='+bin_2d  +'\n') #Binary output of processed 2D data
    f.write('    dump_flag='+dump_flag +'\n') #Binary dump of basic equilibrium data and 2D rzphi spline
    f.write('/'+'\n')

    f.close()

    #writing vac.in... Glasser didn't suggest modifying this in the DCON readme, outside of a, which is the distance of the conformal ideal wall
    #from the plasma in units of minor radius. a>20 is infinite vacuum. a=0 is close fitting ideal wall. small a>0 is ill-advised.
    f = open(working_dir+'/vac.in', 'w')
    f.write('&MODES\n')
    f.write('   mth = 480\n')
    f.write('   xiin(1:9) = 0 0 0 0 0 0 0 1 0\n')
    f.write('   lsymz = .TRUE.\n')
    f.write('   leqarcw = 1\n')
    f.write('   lzio = 0\n')
    f.write('   lgato = 0\n')
    f.write('   lrgato = 0\n')
    f.write('/\n')
    f.write('&DEBUGS\n')
    f.write('   checkd = .FALSE.\n')
    f.write('   check1 = .FALSE.\n')
    f.write('   check2 = .FALSE.\n')
    f.write('   checke = .FALSE.\n')
    f.write('   checks = .FALSE.\n')
    f.write('   wall = .FALSE.\n')
    f.write('   lkplt = 0\n')
    f.write('   verbose_timer_output = f\n')
    f.write('/\n')
    f.write('&VACDAT\n')
    f.write('   ishape = 6\n')
    f.write('   aw = 0.05\n')
    f.write('   bw = 1.5\n')
    f.write('   cw = 0\n')
    f.write('   dw = 0.5\n')
    f.write('   tw = 0.05\n')
    f.write('   nsing = 500\n')
    f.write('   epsq = 1e-05\n')
    f.write('   noutv = 37\n')
    f.write('   idgt = 6\n')
    f.write('   idot = 0\n')
    f.write('   idsk = 0\n')
    f.write('   delg = 15.01\n')
    f.write('   delfac = 0.001\n')
    f.write('   cn0 = 1\n')
    f.write('/\n')
    f.write('&SHAPE\n')
    f.write('   ipshp = 0\n')
    f.write('   xpl = 100\n')
    f.write('   apl = 1\n')
    f.write('   a = '+str(a_wall)+'\n')
    f.write('   b = 170\n')
    f.write('   bpl = 1\n')
    f.write('   dpl = 0\n')
    f.write('   r = 1\n')
    f.write('   abulg = 0.932\n')
    f.write('   bbulg = 17.0\n')
    f.write('   tbulg = 0.02\n')
    f.write('   qain = 2.5\n')
    f.write('/\n')
    f.write('&DIAGNS\n')
    f.write('   lkdis = .FALSE.\n')
    f.write('   ieig = 0\n')
    f.write('   iloop = 0\n')
    f.write('   lpsub = 1\n')
    f.write('   nloop = 128\n')
    f.write('   nloopr = 0\n')
    f.write('   nphil = 3\n')
    f.write('   nphse = 1\n')
    f.write('   xofsl = 0\n')
    f.write('   ntloop = 32\n')
    f.write('   aloop = 0.01\n')
    f.write('   bloop = 1.6\n')
    f.write('   dloop = 0.5\n')
    f.write('   rloop = 1.0\n')
    f.write('   deloop = 0.001\n')
    f.write('   mx = 21\n')
    f.write('   mz = 21\n')
    f.write('   nph = 0\n')
    f.write('   nxlpin = 6\n')
    f.write('   nzlpin = 11\n')
    f.write('   epslp = 0.02\n')
    f.write('   xlpmin = 0.7\n')
    f.write('   xlpmax = 2.7\n')
    f.write('   zlpmin = -1.5\n')
    f.write('   zlpmax = 1.5\n')
    f.write('   linterior = 2\n')
    f.write('/\n')
    f.write('&SPRK\n')
    f.write('   nminus = 0\n')
    f.write('   nplus = 0\n')
    f.write('   mphi = 16\n')
    f.write('   lwrt11 = 0\n')
    f.write('   civ = 0.0\n')
    f.write('   sp2sgn1 = 1\n')
    f.write('   sp2sgn2 = 1\n')
    f.write('   sp2sgn3 = 1\n')
    f.write('   sp2sgn4 = 1\n')
    f.write('   sp2sgn5 = 1\n')
    f.write('   sp3sgn1 = -1\n')
    f.write('   sp3sgn2 = -1\n')
    f.write('   sp3sgn3 = -1\n')
    f.write('   sp3sgn4 = 1\n')
    f.write('   sp3sgn5 = 1\n')
    f.write('   lff = 0\n')
    f.write('   ff = 1.6\n')
    f.write('   fv = 1.6 1.6 1.6 1.6 1.6 1.0 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6 1.6\n')
    f.write('/\n')
    f.close()
    return

def write_rdcon_inputs(working_dir,eq_filename,write_equil_filename='/equil.in',write_rdcon_filename='/rdcon.in',
            ##GAL_INPUT
            nx=256,               # The number of elements in each interval between two singular surfaces
            pfac=0.001,           # Packing ratio near the singular surface
            gal_tol=1e-10,        # Tolerance of lsode integration
            dx1dx2_flag='t',        # Flag to include the special dx1 and dx2 treatments for resonant and extension element
            dx0=5e-4,             # The distance to the singular surface to truncate the lsode integration in resonant element
            dx1=1.e-3,            # The size of resonant element
            dx2=1.e-3,            # The size of extension element
            cutoff=10,            # The number of elements include the large solution as the driving term
            solver="""'LU'""",          #LU factorization of solving Galerkinn matrix
            nq=6,                 # The number of Gaussian points in each Galerkin element
            ##GAL_OUTPUT
            interp_np=3,          # The number of interpration points for outputting Galerkin solution
            restore_uh='t',         # Include the Hermite solution in Galerkin soluitn
            restore_us='t',         # Include the small solution in Galerkin solution
            restore_ul='t',         # Include the larger solution in Galerkin solution
            bin_delmatch='t',       # Output solution for rmatch
            out_galsol='f',         # Output Galerkin solution in ascii files
            bin_galsol='f',         # Output Galerkin solution in binary files
            b_flag='f',             # Output the perturbation of radial b field
            rpec_flag='f',         # Resistive perturbed equilibrium computation
            bin_coilsol='f',        # Output binary files for every unit-m driven solution
            ##RDCON_CONTROL
            bal_flag='t',           # Ideal MHD ballooning criterion for short wavelengths
            mat_flag='t',           # Construct coefficient matrices for diagnostic purposes
            ode_flag='t',           # Integrate ODE's for determining stability of internal long-wavelength mode (must be true for GPEC)
            vac_flag='t',           # Compute plasma, vacuum, and total energies for free-boundary modes
            gal_flag='t',           # Compute outer regime using resonant Galerkin method
            dump_MRE_data='f',      # Dump MRE data, exits before gal_flag can run

            sas_flag='f',         # Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
            dmlim=0.2,            # See sas_flag
            sing_start=0,         # Start integration at the sing_start'th rational from the axis (psilow)

            nn=1,                 # Toroidal mode number
            delta_mlow=8,         # Expands lower bound of Fourier harmonics
            delta_mhigh=8,        # Expands upper bound of  Fourier harmonics
            delta_mband=0,        # Integration keeps only this wide a band of solutions along the diagonal in m,m'
            mthvac=960,           # Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
            thmax0=1,             # Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

            tol_nr=1e-6,          # Relative tolerance of dynamic integration steps away from rationals
            tol_r=1e-7,           # Relative tolerance of dynamic integration steps near rationals
            crossover=1e-2,       # Fractional distance from rational q at which tolerance is switched to tol_r
            singfac_min=1e-4,     # Fractional distance from rational q at which ideal jump condition is enforced
            ucrit=1e3,            # Maximum fraction of solutions allowed before re-normalized

            cyl_flag='f',           # Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).

            sing1_flag='f',         # Special power series treatment
            sing_order=6,         # The highest order of power series to be retained
            sing_order_ceiling='t', # Auto detect the minium order to be retained in power series

            regrid_flag='f',        # Redo the grid generation for galerkin method

            #RDCON_OUTPUT
            crit_break='t',         # Color of the crit curve changes when crossing a singular surface

            ahb_flag='f',           # Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
            msol_ahb=1,           # Number of eigenfunctions output by ahb_flag=t ?
            mthsurf0=1,           # Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag=t

            bin_euler='f',          # Output M psi-by-M euler-lagrange solutions to binary file euler.bin
            euler_stride=1,       # Output only every euler_stride'th psi step to binary file

            out_bal1='f',           # Ascii output for bal_flag poloidal functions
            bin_bal1='f',           # Binary output for bal_flag poloidal functions
            out_bal2='f',           # Ascii output for bal_flag functions
            bin_bal2='f',           # Binary output for bal_flag functions

            #UA_DIAGNOSE_LIST
            flag='f',
            phase='t',
            **kwargs):
    
    write_equil_in(working_dir,eq_filename,write_equil_filename=write_equil_filename,**kwargs)
    
    f = open(working_dir+write_rdcon_filename, 'w')

    f.write('&GAL_INPUT'+'\n')
    f.write('    nx='+str(nx)+'\n')  #The number of elements in each interval between two singular surfaces
    f.write('    pfac='+str(pfac)+'\n') #Packing ratio near the singular surface
    f.write('    gal_tol='+str(gal_tol) +'\n') #Tolerance of lsode integration
    f.write('    dx1dx2_flag='+dx1dx2_flag+'\n')  #Flag to include the special dx1 and dx2 treatments for resonant and extension element
    f.write('    dx0='+str(dx0)+'\n') #The distance to the singular surface to truncate the lsode integration in resonant element
    f.write('    dx1='+str(dx1) +'\n') #The size of resonant element
    f.write('    dx2='+str(dx2) +'\n') #The size of extension element
    f.write('    cutoff='+str(cutoff)+'\n')  #The number of elements include the large solution as the driving term
    f.write('    solver='+solver+'\n') #LU factorization of solving Galerkinn matrix
    f.write('    nq='+str(nq)+'\n')  #The number of Gaussian points in each Galerkin element
    f.write('/'+'\n')
    f.write('&GAL_OUTPUT'+'\n')
    f.write('    interp_np='+str(interp_np)+'\n') #The number of interpration points for outputting Galerkin solution
    f.write('    restore_uh='+restore_uh   +'\n') #Include the Hermite solution in Galerkin soluitn
    f.write('    restore_us='+restore_us   +'\n') #Include the small solution in Galerkin solution
    f.write('    restore_ul='+restore_ul   +'\n') #Include the larger solution in Galerkin solution
    f.write('    bin_delmatch='+bin_delmatch +'\n') #Output solution for rmatch
    f.write('    out_galsol='+out_galsol   +'\n') #Output Galerkin solution in ascii files
    f.write('    bin_galsol='+bin_galsol   +'\n') #Output Galerkin solution in binary files
    f.write('    b_flag='+b_flag   +'\n') #Output the perturbation of radial b field
    f.write('    coil%rpec_flag='+rpec_flag +'\n') #Resistive perturbed equilibrium computation
    f.write('    bin_coilsol='+bin_coilsol +'\n') #Output binary files for every unit-m driven solution
    f.write('/'+'\n')
    f.write('&RDCON_CONTROL'+'\n')
    f.write('    bal_flag='+bal_flag +'\n') #Ideal MHD ballooning criterion for short wavelengths
    f.write('    mat_flag='+mat_flag +'\n') #Construct coefficient matrices for diagnostic purposes
    f.write('    ode_flag='+ode_flag +'\n') #Integrate ODE's for determining stability of internal long-wavelength mode (must be true for GPEC)
    f.write('    vac_flag='+vac_flag +'\n') #Compute plasma, vacuum, and total energies for free-boundary modes
    f.write('    gal_flag='+gal_flag +'\n') #Compute outer regime using resonant Galerkin method
    f.write('    dump_MRE_data='+dump_MRE_data +'\n') #Dump MRE data, exits before gal_flag can run

    f.write('    sas_flag='+sas_flag +'\n') #Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
    f.write('    dmlim='+str(dmlim)  +'\n') #See sas_flag
    f.write('    sing_start='+str(sing_start)   +'\n') #Start integration at the sing_start'th rational from the axis (psilow)

    f.write('    nn='+str(nn)   +'\n') #Toroidal mode number
    f.write('    delta_mlow='+str(delta_mlow)   +'\n') #Expands lower bound of Fourier harmonics
    f.write('    delta_mhigh='+str(delta_mhigh)  +'\n') #Expands upper bound of  Fourier harmonics
    f.write('    delta_mband='+str(delta_mband)  +'\n') #Integration keeps only this wide a band of solutions along the diagonal in m,m'
    f.write('    mthvac='+str(mthvac) +'\n') #Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
    f.write('    thmax0='+str(thmax0)   +'\n') #Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

    f.write('    tol_nr='+str(tol_nr)+'\n') #Relative tolerance of dynamic integration steps away from rationals
    f.write('    tol_r='+str(tol_r) +'\n') #Relative tolerance of dynamic integration steps near rationals
    f.write('    crossover='+str(crossover) +'\n') #Fractional distance from rational q at which tolerance is switched to tol_r
    f.write('    singfac_min='+str(singfac_min)  +'\n') # Fractional distance from rational q at which ideal jump condition is enforced
    f.write('    ucrit='+str(ucrit)  +'\n') #Maximum fraction of solutions allowed before re-normalized

    f.write('    cyl_flag='+cyl_flag +'\n') #Make delta_mlow and delta_mhigh set the actual m truncation bounds. Default is to expand (n*qmin-4, n*qmax).

    f.write('    sing1_flag='+sing1_flag   +'\n') #Special power series treatment
    f.write('    sing_order='+str(sing_order)   +'\n') #The highest order of power series to be retained
    f.write('    sing_order_ceiling='+sing_order_ceiling +'\n') # Auto detect the minium order to be retained in power series

    f.write('    regrid_flag='+regrid_flag  +'\n') #Redo the grid generation for galerkin method
    f.write('/'+'\n')

    f.write('&RDCON_OUTPUT'+'\n')
    f.write('    crit_break='+crit_break  +'\n') #Color of the crit curve changes when crossing a singular surface

    f.write('    ahb_flag='+ahb_flag  +'\n') #Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
    f.write('    msol_ahb='+str(msol_ahb)   +'\n') #Number of eigenfunctions output by ahb_flag='t ?
    f.write('    mthsurf0='+str(mthsurf0)  +'\n') #Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag='t

    f.write('    bin_euler='+bin_euler  +'\n') #Output M psi-by-M euler-lagrange solutions to binary file euler.bin
    f.write('    euler_stride='+str(euler_stride)  +'\n') #Output only every euler_stride'th psi step to binary file

    f.write('    out_bal1='+out_bal1  +'\n') #Ascii output for bal_flag poloidal functions
    f.write('    bin_bal1='+bin_bal1  +'\n') #Binary output for bal_flag poloidal functions
    f.write('    out_bal2='+out_bal2  +'\n') #Ascii output for bal_flag functions
    f.write('    bin_bal2='+bin_bal2  +'\n') #Binary output for bal_flag functions
    f.write('/'+'\n')

    f.write('&UA_DIAGNOSE_LIST'+'\n')
    f.write('    uad%flag='+flag  +'\n')
    f.write('    uad%phase='+phase  +'\n')
    f.write('/'+'\n')


    f.close()
    return

def write_dcon_inputs(working_dir,eq_filename,write_equil_filename='/equil.in',write_dcon_filename='/dcon.in',
        #DCON_CONTROL
        bal_flag='f',   # Ideal MHD ballooning criterion for short wavelengths
        mat_flag='f',   # Construct coefficient matrices for diagnostic purposes
        ode_flag='t',   # Integrate ODE's for determining stability of internal long-wavelength mode (must be true for GPEC)
        vac_flag='t',   # Compute plasma, vacuum, and total energies for free-boundary modes
        mer_flag='f',   # Evaluate the Mercier criterian

        sas_flag='f',   # Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
        dmlim=0.2,     # See sas_flag
	    psiedge=1.0,   # If less then psilim, calculates dW(psi) between psiedge and psilim, then runs with truncation at max(dW)
        psi_search_range=0.02, # If greater than 0, calculates dW(psi) between psihigh and psihigh-psi_search_range, then runs with truncation at max(dW)
	    qlow=0,     # 1.015 Integration initiated at q determined by minimum of qlow and q0 from equil
        qhigh=1e3 ,     # Integration terminated at q limit determined by minimum of qhigh and qa from equil
        sing_start=0,   # Start integration at the sing_start'th rational from the axis (psilow)
        reform_eq_with_psilim='f',#Reforms equilibrium splines between psilow and psilim determined by psihigh/sas_flag/qhigh/peak_flag
        gse_err_logtol=-1,#Sets psihigh to the largest psifac grid point such that relative Grad-Shafranov error remains less than 10^gse_err_logtol (set to 1 to turn off, otherwise takes values (-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0))

        nn=1,           # Toroidal mode number
        delta_mlow=8,   # Expands lower bound of Fourier harmonics
        delta_mhigh=8,  # Expands upper bound of  Fourier harmonics
        delta_mband=0,  # Integration keeps only this wide a band of solutions along the diagonal in m,m'
        mthvac=960,     # Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
        thmax0=1  ,     # Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

        kin_flag = 'f', # Kinetic EL equation (default: false)
        con_flag = 'f', # Continue integration through layers (default: false)
        kinfac1 = 1.0,  # Scale factor for energy contribution (default : 1.0)
        kinfac2 = 1.0,  # Scale factor for torque contribution (default : 1.0)
        kingridtype = 0 ,# Regular grid method (default : 0)
        passing_flag ='t',# Includes passing particle effects (default: false)
        ktanh_flag = 't',# Ignore kinetic effects in the core smoothly (default: false)
        ktc = 0.1 ,     # Parameter activated by ktanh_flag: roughly corresponds to core width ignored (default: 0.1)
        ktw = 50.0,     # Parameter activated by ktanh_flag: width of hyper-tangential functions (default: 50.0)
        ion_flag ='t',  # Include ion dW_k when kin_flag is true (summed with electron contribution if electron_flag true)
        electron_flag ='f',# Include electron dW_k when kin_flag is true (summed with ion contribution if ion_flag true)

        tol_nr=1e-6 ,   # Relative tolerance of dynamic integration steps away from rationals
        tol_r=1e-7,     # Relative tolerance of dynamic integration steps near rationals
        crossover=1e-2  ,# Fractional distance from rational q at which tolerance is switched to tol_r
        singfac_min=1e-4,# Fractional distance from rational q at which ideal jump condition is enforced
        ucrit=1e4,     # Maximum fraction of solutions allowed before re-normalized

        termbycross_flag ='f',# Terminate ODE solver in the event of a zero crossing

        use_classic_splines ='f',# Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions

        #DCON_OUTPUT
        crit_break='t', # Color of the crit curve changes when crossing a singular surface

        ahb_flag='f',   # Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
        msol_ahb=1,     # Number of eigenfunctions output by ahb_flag=t ?
        mthsurf0=1,     # Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag=t

        bin_euler='f' , # Output M psi-by-M euler-lagrange solutions to binary file euler.bin
        euler_stride=1  ,# Output only every euler_stride'th psi step to binary file

        out_bal1='f',   # Ascii output for bal_flag poloidal functions
        bin_bal1='f',   # Binary output for bal_flag poloidal functions
        out_bal2='f',   # Ascii output for bal_flag functions
        bin_bal2='f',   # Binary output for bal_flag functions

        netcdf_out='t',  # Replicate ascii dcon.out information in a netcdf file
        **kwargs):
    
    write_equil_in(working_dir,eq_filename,write_equil_filename=write_equil_filename,**kwargs)

    f = open(working_dir+write_dcon_filename, 'w')

    f.write('&DCON_CONTROL'+'\n')
    f.write('    bal_flag='+bal_flag+'\n')# Ideal MHD ballooning criterion for short wavelengths
    f.write('    mat_flag='+mat_flag+'\n')# Construct coefficient matrices for diagnostic purposes
    f.write('    ode_flag='+ode_flag+'\n')# Integrate ODE's for determining stability of internal long-wavelength mode (must be true for GPEC)
    f.write('    vac_flag='+vac_flag+'\n')# Compute plasma, vacuum, and total energies for free-boundary modes
    f.write('    mer_flag='+mer_flag+'\n')# Evaluate the Mercier criterian

    f.write('    sas_flag='+sas_flag+'\n')# Safety factor (q) limit determined as q_ir+dmlim where q_ir is the equil outermost rational
    f.write('    dmlim='+str(dmlim) +'\n')# See sas_flag
    f.write('    psiedge='+str(psiedge)+'\n')# If less then psilim, calculates dW(psi) between psiedge and psilim, then runs with truncation at max(dW)
    f.write('    psi_search_range='+str(psi_search_range)+'\n')
    f.write('    qlow='+str(qlow) +'\n')# Integration initiated at q determined by minimum of qlow and q0 from equil
    f.write('    qhigh='+str(qhigh) +'\n')# Integration terminated at q limit determined by minimum of qhigh and qa from equil
    f.write('    sing_start='+str(sing_start)+'\n')# Start integration at the sing_start'th rational from the axis (psilow)
    f.write('    reform_eq_with_psilim='+reform_eq_with_psilim+'\n')#Reforms equilibrium splines between psilow and psilim determined by psihigh/sas_flag/qhigh/peak_flag
    f.write('    gse_err_logtol='+str(gse_err_logtol)+'\n')#Sets psihigh to the largest psifac grid point such that relative Grad-Shafranov error remains less than 10^gse_err_logtol (set to 1 to turn off, otherwise takes values (-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0))


    f.write('    nn='+str(nn)    +'\n')# Toroidal mode number
    f.write('    delta_mlow='+str(delta_mlow)+'\n')# Expands lower bound of Fourier harmonics
    f.write('    delta_mhigh='+str(delta_mhigh)+'\n')# Expands upper bound of  Fourier harmonics
    f.write('    delta_mband='+str(delta_mband)+'\n')# Integration keeps only this wide a band of solutions along the diagonal in m,m'
    f.write('    mthvac='+str(mthvac)+'\n')# Number of points used in splines over poloidal angle at plasma-vacuum interface. Overrides vac.in mth.
    f.write('    thmax0='+str(thmax0)  +'\n')# Linear multiplier on the automatic choice of theta integration bounds for high-n ideal ballooning stability computation (strictly, -inf to -inf)

    f.write('    kin_flag ='+ kin_flag+'\n')# Kinetic EL equation (default: false)
    f.write('    con_flag ='+ con_flag+'\n')# Continue integration through layers (default: false)
    f.write('    kinfac1 ='+ str(kinfac1)+'\n')# Scale factor for energy contribution (default : 1.0)
    f.write('    kinfac2 ='+ str(kinfac2)+'\n')# Scale factor for torque contribution (default : 1.0)
    f.write('    kingridtype ='+ str(kingridtype) +'\n')# Regular grid method (default : 0)
    f.write('    passing_flag ='+ passing_flag+'\n')# Includes passing particle effects (default: false)
    f.write('    ktanh_flag ='+ ktanh_flag  +'\n')# Ignore kinetic effects in the core smoothly (default: false)
    f.write('    ktc ='+ str(ktc) +'\n')# Parameter activated by ktanh_flag: roughly corresponds to core width ignored (default: 0.1)
    f.write('    ktw ='+ str(ktw)+'\n')# Parameter activated by ktanh_flag: width of hyper-tangential functions (default: 50.0)
    f.write('    ion_flag ='+ ion_flag+'\n')# Include ion dW_k when kin_flag is true (summed with electron contribution if electron_flag true)
    f.write('    electron_flag ='+ electron_flag +'\n')# Include electron dW_k when kin_flag is true (summed with ion contribution if ion_flag true)

    f.write('    tol_nr='+str(tol_nr) +'\n')# Relative tolerance of dynamic integration steps away from rationals
    f.write('    tol_r='+str(tol_r)+'\n')# Relative tolerance of dynamic integration steps near rationals
    f.write('    crossover='+str(crossover)  +'\n')# Fractional distance from rational q at which tolerance is switched to tol_r
    f.write('    singfac_min='+str(singfac_min)+'\n')# Fractional distance from rational q at which ideal jump condition is enforced
    f.write('    ucrit='+str(ucrit) +'\n')# Maximum fraction of solutions allowed before re-normalized

    f.write('    termbycross_flag ='+ termbycross_flag+'\n')# Terminate ODE solver in the event of a zero crossing

    f.write('    use_classic_splines ='+ use_classic_splines+'\n')# Use a classical cubic spline instead of tri-diagonal solution for splines with extrapolation boundary conditions
    f.write('/'+'\n')
    f.write('&DCON_OUTPUT'+'\n')
    f.write('    crit_break='+crit_break+'\n')# Color of the crit curve changes when crossing a singular surface

    f.write('    ahb_flag='+ahb_flag+'\n')# Output normal magnetic field eigenvalues and eigenfunctons at plasma-vacuum interface (must be false for GPEC)
    f.write('    msol_ahb='+str(msol_ahb)+'\n')# Number of eigenfunctions output by ahb_flag=t ?
    f.write('    mthsurf0='+str(mthsurf0)+'\n')# Linear multiplier on number of boundary points used to display surface eigenfunctions for ahgb_flag=t

    f.write('    bin_euler='+bin_euler +'\n')# Output M psi-by-M euler-lagrange solutions to binary file euler.bin
    f.write('    euler_stride='+str(euler_stride)  +'\n')# Output only every euler_stride'th psi step to binary file

    f.write('    out_bal1='+out_bal1+'\n')# Ascii output for bal_flag poloidal functions
    f.write('    bin_bal1='+bin_bal1+'\n')# Binary output for bal_flag poloidal functions
    f.write('    out_bal2='+out_bal2+'\n')# Ascii output for bal_flag functions
    f.write('    bin_bal2='+bin_bal2+'\n')# Binary output for bal_flag functions

    f.write('    netcdf_out='+netcdf_out+'\n')# Replicate ascii dcon.out information in a netcdf file
    f.write('/'+'\n')

    f.close()
    return

###Current situation:::
    ### ASS