#exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/test_case.py").read())


import os
os.getcwd()
os.chdir("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts")
exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/random_start.py").read())
point_test=False
#if 'mygs' not in globals():
#  mygs = TokaMaker()

#working_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/docs/examples/DIIID_resistive_example"
#eq_filename="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT"
#os.chdir(working_dir)

#We are in sawtooth realm...



os.chdir("/Users/sbenjamin/Desktop/ptestNew5")

#Batches, tokamaker info:
tokamak_xr_2of4=xr.open_dataset("/Users/sbenjamin/Desktop/ptestNew5/tokamak_xr_2of4")
#Specific dcon run:
tokamak_xr_2of4

#PLan:
    #We split up equilibrium generation and dcon running... 
    #Why? because shit is fucked thats why...



exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/random_start.py").read())

n=10
cases_in_batch=1
working_dir="/Users/sbenjamin/Desktop/ptestNew4/"
out_path="/Users/sbenjamin/Desktop/ptestNew4/"
dcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/dcon"
rdcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/rdcon"

out = gen_n_successful_cases(n,
            out_path=out_path,
            working_dir=working_dir,
            dcon_executable_dir=dcon_executable_dir,
            rdcon_executable_dir=rdcon_executable_dir,
            radas_dir_path='/Users/sbenjamin/Desktop/TEARING_WIDGET/cfspopcon/radas_dir',
            inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/",
            cases_in_batch=cases_in_batch,
            run_phys_suite=False,
            save_equil=True,
            return_total=True,save=True,just_dynamo=True,
            verbose=True,back_down_current=True,run_tokamaker=True,
            max_high_def_iterations=3,
            high_def_plasma_dx=0.005,
            gse_err_logtol=0.0,
            psihigh=0.9999,
            req_mhd_stable=False,
            nmax=3,
            nr=500,nz=500,maxsteps=1000000,lcfs_pad=1e-5,ttol=1e-9,
            plot_machine=False,plot_popcons=False,plot_mesh=False,plot_coils=False,plot_iteration=False,
            )

run_dcon_on_equilibria(out_path,dcon_executable_dir,rdcon_executable_dir,
            gse_err_logtol=0.0,
            psihigh=0.9999,
            grid_type_diagnose="""'pow1'""",
            mpsi_diagnose=512,
            qlow=1.01,
            nmax=3)


out_path2="/Users/sbenjamin/Desktop/ptestNew5/"
out = gen_n_successful_cases(10,
            cases_in_batch=3,
            out_path=out_path2,
            radas_dir_path='/Users/sbenjamin/Desktop/TEARING_WIDGET/cfspopcon/radas_dir',
            inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/",
            run_phys_suite=False,
            save_equil=True,
            return_total=False,save=True,just_dynamo=True,
            verbose=True,back_down_current=True,run_tokamaker=True,
            max_high_def_iterations=5,
            high_def_plasma_dx=0.004,
            nr=600,nz=600,maxsteps=1000000,lcfs_pad=1e-5,ttol=1e-9,
            plot_machine=False,plot_popcons=False,plot_mesh=False,plot_coils=False,plot_iteration=False,
            )



out_path2="/Users/sbenjamin/Desktop/ptestNew5/"
dcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/dcon"
rdcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/rdcon"
run_dcon_on_equilibria(out_path2,dcon_executable_dir,rdcon_executable_dir,
            gse_err_logtol=0.0,
            psihigh=0.9999,
            grid_type_diagnose="""'pow1'""",
            mpsi_diagnose=512,
            qlow=1.01,
            nmax=3)



#A couple options...
#Make loop to just read equilibria and run rdcon


#working on just equilibria... with truncating at current = .9 for speed
#trying again with dcon included.. it fcking dies

#Fix run_DCON_on_equilibrium2, its currently not saving dW_total, dcon_nzero correctly

#its dying when it finds case, has to restart again...


exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/run_local_RDCON.py").read())
working_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/"
dcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/dcon"
rdcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/rdcon"
g_file1_name="/Users/sbenjamin/Desktop/g_eqdsk_temp"
g_file3_name="/Users/sbenjamin/Desktop/g_dyn_eqdsk_temp"
gfile_name="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/g_dyn_eqdsk_temp"
dcon_ran1,rdcon_ran1,MRE_xr1 =run_DCON_on_equilibrium2(g_file3_name,
                newq0=0,working_dir=working_dir,dcon_executable_dir=dcon_executable_dir,
                rdcon_executable_dir=rdcon_executable_dir,
                gse_err_logtol=0.0,psihigh=0.9999)




prof_dat_xr2 = pd.read_csv(working_dir+'/bongly.csv', dtype={'ipsis': 'int'}).to_xarray().set_index(index="ipsis").rename({'index': 'ipsis'})#, dtype={'ipsis': 'int'})


point_test=True
if point_test: #Largest tokamak
    a = np.zeros(7)
    a[0]=3.7
    a[1]=0.3
    a[2]=0.4
    a[3]=1.9
    a[6]=0.8
    a[5]=field_on_axis(a[0],a[1])  
    a[4]=calc_plasma_current_from_marginally_stable_current_fraction(
                a[6],
                a[5],
                a[0],
                a[1], 
                a[3])
    print(calc_plasma_surface_area(a[0],a[3],a[1]))
    point=run_input_case(a,0)[1]
    if run_input_case(a,0)[0]:
        point=point.sel(point_type='min_pressure')
        plot_case_from_dataset_point(point)

exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/random_start.py").read())


outdir="/Users/sbenjamin/Desktop/"
flag1, g_file1_name, J_tot_true1, f_tr1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1,qvals1, flag3, g_file3_name, J_tot_true3, f_tr3, j_BS3, jtor_total3, flux_surf_avg_of_B_timesj_BS3, qvals3, q0_vals, q1_psi_surfs, q0new3 = run_tokamaker_case_on_point_macro(point, mygs, True, 24,outdir,2,3,1000,1000,
                                        high_def_plasma_dx=0.01, max_high_def_iterations=6,
                                        rescale_ind_Jtor_by_ftr=False,
                                        smooth_dynamo_inputs=True, 
                                        save_equil=True, run_phys_suite=True, verbose=False,
                                        lcfs_pad=1e-6,ttol=1e-9,
                                        plot_iteration_internal=False,
                                        plot_machine=False,plot_mesh=False,plot_coils=False)


#Can't go smaller than high_def_plasma_dx=0.001
#Quite a lot of edge error at high_def_plasma_dx=0.002

#This currently breaking things.... ?
#plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250, rho=False)

"""
mygs, flag1, j_BS1, jtor_total1, flux_surf_avg_of_B_timesj_BS1, ne, Te, ni, Ti, jtor_noBS1, zeffs, pp_prof1, ffp_prof1 = run_tokamaker_case_on_point(
                                point, mygs, plasma_dx=0.01, coil_dx=0.005, vac_dx=0.06, coil_set=2, include_bs_solver=True, 
                                max_iterations=5, Z0=0.0, smooth_inputs=True, rescale_ind_Jtor_by_ftr=True,
                                out_all=True,Iptarget=True,initialize_eq=False, plot_iteration=True)
_,qvals1,_,_,_,_ = mygs.get_q(npsi=len(j_BS1)) 
qvals1[0]

#current frac 0.5, q0=0.69 no rescale_ind_Jtor_by_ftr, didn't converge with it
#another tokamak frac 0.65, q0=0.4 no rescale_ind_Jtor_by_ftr, q0=0.2 with it
plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250, rho=True)

fig2,ax2=plt.subplots(3,1,figsize=(4,6),constrained_layout=True)
flag2, j_BS2, jtor_total2, flux_surf_avg_of_B_timesj_BS2, jtor_noBS2, pp_prof2, ffp_prof2, q0_vals, q1_psi_surfs, qvals2, pax2, flag_end2 = basic_dynamo_w_bootstrap(mygs,ne,Te,ni,Ti,jtor_noBS1,zeffs,
                                    smooth_inputs=True,max_iterations=10,initialize_eq=False,fig=fig2,ax=ax2, rescale_ind_Jtor_by_ftr=False,plot_iteration=True)
plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250, rho=True)


#Do we add a clause, ohmic bootstrap not included...
# 
"""







#Debug ordering::::
#1: point test run_tokamaker_case_on_point_macro and basic_dynamo_w_bootstrap therewithin
#2: point test run_DCON_on_equilibrium2...
    # try it with some balooning cases
    #^these two combined is basically the inner step of the loops
#3: debug _gen_n_successful_cases
#4: work on a program for initiating _gen_n_successful_cases within local areas
#5: make sure we can load that shit back in...

if False and point_test:
    mygs,err_flag,j_BS,jtor_total,flux_surf_avg_of_B_timesj_BS,ne,Te,ni,Ti,jtor_noBS,zeffs=run_tokamaker_case_on_point(point, mygs, out_all=True,
        qmin=-1,max_iterations=1,Iptarget=True,plot_iteration=False,initialize_eq=False, plasma_dx=0.002,plot_machine=True,plot_mesh=False,plot_coils=False)
    #plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250)

"""
exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/q_iterator_draft.py").read())
fig,ax=plt.subplots(3,1,sharex=True)
flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, q0_vals, q1_psi_surf, flag_end=solve_with_bootstrap_q0_addition5(mygs,ne,Te,ni,Ti,jtor_noBS,zeffs,fig=fig,ax=ax,
                                     rescale_ind_Jtor_by_ftr=True,initialize_eq=False,
                                     max_iterations=50,smooth_inputs=False,jBS_scale=1.0,plot_iteration=True)
plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250)
plot_Jbs(mygs,j_BS,point)
"""

#what is faster? Basically, we separate the solving of the q convergence, and the high-res equilibrium
#Do we ... or ...?
#We gotta separate them

#What do we do with this monstrosity???
    #Well, we can add an option to the loop
    #Calc dynamo 
    #When we do this, we float q-profile in RDCON by enough to make sure q>1.02 everywhere...
        #Gotta check none of the surface terms blow up too much
            #They can stay in the q>1.02 section...

#Best option:::
#   We go all in on what we want... which is 
#   Solve good, output all, save [minor changes]
#       reset, re_initialise, take fast inputs and go...
#           output all, resolve sharp and thats it.

#Tomorrow, loop becomes real
"""
plt.plot(jtor_total)

plt.plot(psi,orig_Jtor)
jtspline=CubicSpline(psi,orig_Jtor)
plt.scatter(psi_of_qmin,jtspline(psi_of_qmin))
plt.plot(psi,new_equil_Jtor)
plt.show()
#Gotta rescale the inductive current only friend....

psi,qvals,ravgs,_,_,_ = mygs.get_q(npsi=250)

plt.plot(psi,orig_qvals)
qpline=CubicSpline(psi,orig_qvals)
plt.scatter(psi_of_qmin,qpline(psi_of_qmin))
plt.plot(psi,new_qvals)
plt.show()




if True and point_test:
    mygs.reset()
    mygs, err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, ne,Te,ni,Ti,jtor_noBS,zeffs=run_tokamaker_case_on_point(point, mygs, out_all=True,
        qmin=-1,max_iterations=2,Iptarget=True,plot_iteration=False,initialize_eq=False, plasma_dx=0.01,plot_machine=True,plot_mesh=False,plot_coils=False)
    plot_equilibrium_profiles(mygs, psi_pad=1.E-4, npsi=250)
    exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/q_iterator_draft.py").read())
    flattened_Jtor,psi_flat,psi,orig_Jtor,new_equil_Jtor,orig_qvals,new_qvals,psi_of_qmin=dynamo_time(self,1.02,
                    mygs._Ip_target.value,simple_flat=True,n_psi=300,smooth_inputs=False)



plt.plot(psi,orig_Jtor)
jtspline=CubicSpline(psi,orig_Jtor)
plt.scatter(psi_of_qmin,jtspline(psi_of_qmin))
plt.plot(psi,new_equil_Jtor)
plt.show()
#Gotta rescale the inductive current only friend....

plt.plot(psi,orig_qvals)
qpline=CubicSpline(psi,orig_qvals)
plt.scatter(psi_of_qmin,qpline(psi_of_qmin))
plt.plot(psi,new_qvals)
plt.show()


ran,flattened_Jtor,orig_Jtor,new_FFp,psi_flat,psi,psi_of_qmin,a_avgs,qorig,pp_on_mu0=dynamo_Jtor_profile(self,1.02,mygs._Ip_target.value,n_psi=300)


plt.plot(psi,flattened_Jtor)
plt.plot(psi,orig_Jtor)
plt.scatter(psi_of_qmin,jtspline(psi_of_qmin))
tot_current_orig=cross_section_surf_int(orig_Jtor,a_avgs)
tot_current_flat=cross_section_surf_int(flattened_Jtor,a_avgs)
print(tot_current_orig,tot_current_flat)

plt.show()




fig,ax=plt.subplots(5,1,sharex=True)
exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/q_iterator_draft.py").read())
pp_prof2, ffp_prof2, j_BS2, new_Jtorr2, flux_surf_avg_of_B_timesj_BS2=solve_with_flat_qmin4(mygs,1.02,ne,Te,ni,Ti,zeffs,
                    smooth_inputs=False,fig=fig,ax=ax,plot_iteration=True,iterate_psi_step=0.05)


psi_q,qvals,ravgs,dl,rbounds,zbounds = mygs.get_q(psi_pad=1.E-4,npsi=250)



fig,ax=plt.subplots(2,1,sharex=True)
flag, j_BS, new_Jtorr, flux_surf_avg_of_B_timesj_BS=solve_with_bootstrap(mygs,ne,Te,ni,Ti,jtor_noBS,point.z_effective.values,max_iterations=4,initialize_eq=False,fig=fig,ax=ax,plot_iteration=True)

"""
if False and point_test:
    solve_with_flat_qmin1(mygs,qmin,ne,Te,ni,Ti,zeffs)
    psi_q,qvals,ravgs,dl,rbounds,zbounds = mygs.get_q(psi_pad=1.E-4,npsi=250)
    q_spline=CubicSpline(psi_q,qvals)
    fig, ax = plt.subplots(1,1,sharex=True)
    ax.scatter(np.sqrt(psi_q),qvals)
    ax.set_ylim(bottom=0.0,top=q_spline(0.98))
    ax.axhline([1.0],color='black',linestyle='dotted')
    ax.set_ylabel("q")
    plt.show()

    err_flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS = solve_with_flat_qmin(mygs,qmin,ne,Te,ni,Ti,point.z_effective.values,iterate_psi_step=iterate_psi_step,max_iterations=15,fig=fig)
    mygs.reset()



    #mygscopy, err_flag, J_BS=run_tokamaker_case_on_point(point, mygs, 
    #    Iptarget=True,plot_iteration=False,initialize_eq=True)
    #TokaMaker.save_eqdsk(mygscopy,"eqdsk_PAIN")            
    #plot_case_from_dataset_point(point)
    #mygs.print_info()
    #plot_machine(mygs)
    #plot_Jbs(mygs,J_BS,point)
    #plot_equilibrium_profiles(mygs)
    #plt.show()


if False:
    out=gen_n_successful_cases(2,cases_in_batch=100,min_surface_area=200,out_path="loop_tests",
        working_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/",
        dcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/dcon",
        rdcon_executable_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/rdcon",
        run_tokamaker=True,save=True,plot_mesh=True,plot_coils=True,plot_iteration=True,
        plot_machine=True,back_down_current=False,
        return_total=True,plot_popcons=True,run_phys_suite=True,
        lcfs_pad=1e-6,save_equil=False,
        inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",
        gse_err_logtol=-0.5,nmax=2,
        plasma_dx=0.01,mpsi=256,mpsi_diagnose=256,grid_type="""'pow2'""",
        verbose=True)



#put=run_DCON_on_equilibrium('loop_testsg_eqdsk_temp',**out)



#working_dir="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/docs/examples/DIIID_resistive_example"
#eq_filename="/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT"
#os.chdir(working_dir)

#exec(open("/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/run_local_RDCON.py").read())
#case_ran,dconxr=run_DCON_on_equilibrium(eq_filename,psihigh=0.99,mpsi=256,working_dir=working_dir,nmin=1,nmax=2,write_equil_filename='/equil.in')






#case_ran,ns_dcon_ran,ns_rdcon_ran,equil_dat_xr,prof_dat_xr,edge_dat_xr,gse_dat_xr,MRE_combined_vec=run_DCON_on_equilibrium(eq_filename,
#    psihigh=0.99,mpsi=256,working_dir=working_dir,nmin=1,nmax=2,write_equil_filename='/equil.in')


###FIX RDCON INPUTS!!!!!!
"""
ns_dcon_ran1=ns_dcon_ran
ns_rdcon_ran1=ns_rdcon_ran

case_ran1=case_ran
equil_dat_xr1=equil_dat_xr
prof_dat_xr1=prof_dat_xr
edge_dat_xr1=edge_dat_xr
gse_dat_xr1=gse_dat_xr
MRE_combined_vec1=MRE_combined_vec

edge_dat_xr1=edge_dat_xr1.rename({'index': 'edge_index'})
gse_dat_xr1=gse_dat_xr1.rename({'index': 'gse_index'})
equil_dat_xr1.assign(ns_dcon_ran=ns_dcon_ran,ns_rdcon_ran1=ns_rdcon_ran1)

equil_dat_xr1=equil_dat_xr1.merge(prof_dat_xr1)
equil_dat_xr1=equil_dat_xr1.merge(edge_dat_xr1)
equil_dat_xr1=equil_dat_xr1.merge(gse_dat_xr1)

#How to deal with MRE_combined_vec
for i in range(len(MRE_combined_vec1)):
    MRE_combined_vec1[i]=MRE_combined_vec1[i].drop_vars(names=["n"]).assign(n=ns_dcon_ran1[i]).set_coords('n')

    MRE_xr=xr.concat(MRE_combined_vec1,dim="n")

equil_dat_xr1=equil_dat_xr1.merge(MRE_xr)
"""










#xr.merge([stab_metadata,MRE_combined])

#point.combine_first(stab_metadata)

"""

stab_metadata__,DCON_data_Xr,MRE_combined,MRE_data_Xr=run_DCON_on_equilibrium(eq_filename,working_dir=working_dir,nmin=1,nmax=8,write_equil_filename='/equil.in')

#MRE_data_DFahhh = pd.read_csv(working_dir+'/MRE_dat.csv', sep='\t')
MRE_data_Xrahhh = MRE_data_DFahhh.to_xarray() #.set_coords(names=["n","m"])
#MRE_data_DFahhh2 = pd.read_csv(working_dir+'/MRE_dat.csv', sep='\t')
MRE_data_Xrahhh2 = MRE_data_DFahhh2.to_xarray() #.set_coords(names=["n","m"])

MRE_data_Xrahhh=MRE_data_Xrahhh.set_index(index="m").rename({'index': 'm'}).drop_vars(names=["n"]).assign(n=1)
MRE_data_Xrahhh2=MRE_data_Xrahhh2.set_index(index="m").rename({'index': 'm'}).drop_vars(names=["n"]).assign(n=2)

beast=xr.concat([MRE_data_Xrahhh,MRE_data_Xrahhh2],dim="n")

xr.combine([MRE_data_Xrahhh,MRE_data_Xrahhh2])

MRE_data_Xrahhh=MRE_data_Xrahhh.set_xindex(coord_names=["n"]) #.drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)
MRE_data_Xrahhh2=MRE_data_Xrahhh2.set_xindex(coord_names=["n"]) #.drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)

xr.concat([MRE_data_Xrahhh,MRE_data_Xrahhh2],dim="n")


MRE_combinedcopy=MRE_combined
MRE_data_Xrcopy=MRE_data_Xr

MRE_combinedcopy.loc[n=2,m=1]

MRE_combined=MRE_combined.set_xindex(coord_names=["n"])
MRE_combined=MRE_combined.drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)

MRE_data_Xr=MRE_data_Xr.set_xindex(coord_names=["n"]).drop_indexes(coord_names=["index"]).reset_coords(names=["index"],drop=True)

MRE_data_Xr=MRE_data_Xr.set_index(index="m")
MRE_data_Xr=MRE_data_Xr.rename({'index': 'm'})

MRE_data_Xrtest=MRE_data_Xr.set_xindex(coord_names=["m"])
MRE_combinedtest=MRE_combined.set_xindex(coord_names=["m"])
stab_metadata__=xr.merge([MRE_data_Xrtest,MRE_combinedtest])

stab_metadata__test=stab_metadata__.drop_indexes(coord_names=["index"],)
DCON_data_Xr_test=DCON_data_Xr.drop_indexes(coord_names=["index"])
stab_metadata__test=stab_metadata__.reset_coords(names=["index"])
DCON_data_Xr_test=DCON_data_Xr.drop_indexes(coord_names=["index"])

stab_metadata__=xr.merge([stab_metadata__,DCON_data_Xr])
stab_metadata__=xr.merge([stab_metadata__test,DCON_data_Xr_test])

MRE_combined
MRE_data_Xr
MRE_combined=xr.merge([MRE_combined,MRE_data_Xr])

gsei_tol=1e-2
GSE_data_DF = pd.read_csv(working_dir+'/gsei.out', sep='\t')
for i in range(len(GSE_data_DF.psifac)):
    if GSE_data_DF.psifac.values[i]<DCON_data_DF.psihigh.values[0] and GSE_data_DF.errori.values[i] > gsei_tol:
        print(i,GSE_data_DF.psifac.values[i],GSE_data_DF.errori.values[i])

"""
#out=gen_n_successful_cases(1,cases_in_batch=100,min_surface_area=60,out_path="test_spot/good_equils_propa3",
#                           run_tokamaker=True,save=True,plot_coils=True,plot_machine=True,plot_iteration=True,back_down_current=True,
#                           return_total=True,nr=860,nz=860,lcfs_pad=1e-9,test_jtot=True,inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts",verbose=True)

#SUCESS: gen_n_successful_cases back_down_current=False, run_tokamaker=False, save=True (batches working)
#:        gen_n_successful_cases back_down_current=True, run_tokamaker=False, save=True (batch & no batch)
#SUCESS: gen_n_successful_cases back_down_current=False, run_tokamaker=True, save=False

#weprintin_popcon_b2of2=xr.open_dataset('weprintin_popcon_b2of2')
#pott=weprintin_popcon_b2of2.isel(MC_index=2).sel(point_type='min_pressure')
#plot_case_from_dataset_point(pott)

#plt.plot(out.isel(MC_index=0).Jtor_on_linspace_psi_from_TokaMaker_min_P_case)
#plt.plot(out.isel(MC_index=0).Jtor2_on_linspace_psi_from_TokaMaker_min_P_case)


#point=point.assign(J_BS_TokaMaker=0.0*point.sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"])
#point[dict(point_type='min_pressure')]=point[dict(point_type='min_pressure')].assign(J_BS_TokaMaker11=0.0*point.sel(point_type='min_pressure')["normalised_inductive_plasma_current_profile"])

#Questions:
    #can we re-run tokamaker from a popcon after its been saved? I don't know right now... [want to run tokamaker at that point as is]
    #can we re-run get gfile straight from tokamker? probably... do I want to save as we go or in a batch? probably in a batch

#two options:
    #save mygs for a batch of cases or don't save mygs (print g-files one at time)
    #if print g-file, run DCON, get bad result... we sad. however I don't feel comfortable running rdcon raw from python (would need to mess around with wrappers, bash file probs easier)

#what I think is best:
    #we print batches of g-files into directories
    #we cycle running the same compiled rdcon code on those g-files, outputting key terms and stability info
        #use bash scripts to do this
        #if we are looking at inherently unstable current profiles (to anything other than 1/1), idk what to do...

#WEEK PLAN:
    #TN#!!get the batch printing of g-files and popcon info working
        #DEBUG
    #DONE:3#!!make sure I can re-load the important popcon info (profiles etc):: just need to re-run from point etc
        #!!use the profiles from the g-file as the 'REAL' profile
    #3day#!!run rdcon on a few cases (make sure it roughly works)
    #3day#!start generating g-files en masse
    #3day#!fine-tune the rdcon case and post-processing (creation of MRE terms, q-scans etc)
        #!function for best delta-W truncation point
    #Then we in the weeds...

    #TN# emails about k0,k1, poloidal flow stuff

#Tuesday plan:
    #Get geqdsk printing working:
        #Basically it needs to pick up values before copying I reckon, though i'm not sure
    #Consider:
        #Fixing tokamaker coils so they stick to the fence
        #Test total flux thru surface error
    #Priority:::
        #Printing g-files
        #Running a few DCON cases
        #Cooking w equilibria 
        #Fine tuning DCON analysis


#Current situation:

#Clean up WF2 Items [optional]
    #!!!sauter fc need to output those
    #!! validate bootstrap function POPCON vs actual calculated values
    #!  quick check edge match...
    #   -why those errors kinda big...
    #   -just calc edge further then we run rdcon?


#Get WF3 running proper
    #q-scans?

#THEN:
    #Main delta', Dr, Dnc etc what are the trends?
    #Statistical analyses
    #M3DC1 confirmation of linear effects... Dnc effects confirmed with Slayer model
    #q-value scans

#Mystery seg fault:
    #When I try:except: while saving the eqdsk, we later seg fault when doing something with the xarrays
        #either during dataset = algorithm.update_dataset(dataset), or MC_case_dataset_loc=xr.concat(type_datasets,dim="point_type")

#Write to Jeff



#Current plan:  
#from within the python loop:
    #1run dcon low res, solving for best dw between (0.98 and 0.999 or whatever)
        #choose correct psihigh
        #check gse doesn't get too big... 
        #check no zero crossings...
        #DONE(untest)#dump MRE data
        #if all this works we go straight to 4
        #ADD PRINT:
            #psilim from ODE stuff, dW
            #psilim from GSE, dW
    #-----#2re-run dcon medium res, checking for zero crossings up to that q-value... 
        #alternatively, check for zero crossings beneath the psihigh etc... blah blah blah
    #-----#3run rdcon low res, dumping MRE data
    #4run rdcon high res, at the psihigh chosen by step 1... get out what we want...

    #extra option::: make sure we only evaluate sing%psifac < 0.9 (not interested in resistive pedestal physics)
    #extra option 2::: make sure Mercier being actually checked somewhere...

#When we return:::
    #Test run_DCON_on_equilibrium
        #how faster??? Do we Dr/Dh/Dnc_prefac/Wc_prefac etc for the whole profile?
            #maybe this is easier to understand...
        #Maybe we limit sing for delta' calculation...
    #Add run_DCON_on_equilibrium to my python loop
    #Add reason_failed cases (MHD unstable, dcon didn't run, rdcon didnt run 
    #                         (equil didn't compute, equil didn't save)
    #Use sauter_FC to reshape ohmic current profile 
        #Sauter fig 2b
        #Use Keeling 2018 study as validation
        #Record current profiles during loop
    #Make sure I'm using the correct q0 rule... 
    #   Check if tokamaker has a built in method
    #   IS GS-invariant better, or making a flat q-profile... IDK
    #Parralelise whole workflow, make it initialise and run on its own
        #build in hotstart... (?) 

#PLAN:
    #take break, go gym
    #buy plane tickets, figure out what I need to get this show on the road
    #Two options... brute force or smarter/slower (ask Cristina, in the meantime get the Sauter stuff working)
    #Also get the 

#To test:
#dcon print MRE_dat.out
#dcon print equil_dat.out
	#mer_passed
	#dcon_passed
#rdcon Single_Helicity_DeltaPrime.csv
    #seems to be working broh

#To fixL
#Equil.in g file make consistent...

#To do:::

#os.chdir('/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/docs/examples/DIIID_ideal_example')
#MRE_data_DF = pd.read_csv('MRE_data.csv', sep='\t')
#MRE_data_Xr = MRE_data_DF.to_xarray()
#MRE_data_Xr=MRE_data_Xr.set_coords(names=["n","m"])

#_data_DF = pd.read_csv('MRE_data.csv', sep='\t')

#os.chdir('/Users/sbenjamin/Desktop/TEARING_WIDGET/GPEC/docs/examples/DIIID_resistive_example')