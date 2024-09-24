#/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/random_start.py 
import sys
import os

debug=False

if not debug:
    if len(sys.argv)==5:
        array_id=sys.argv[1]
        studyname=sys.argv[2]
        tot_cases=int(sys.argv[3])
        cases_in_batch=int(sys.argv[4])
        out_path ="/home/sbenjamin/TearingMacroFolder/runs/"+studyname+"_proc"+array_id+'/'
    elif len(sys.argv)==4:
        out_path=sys.argv[1]
        tot_cases=int(sys.argv[2])
        cases_in_batch=int(sys.argv[3])
else:
    studyname='cust_test'
    array_id='1'
    tot_cases=2000
    cases_in_batch=5
    out_path ="/home/sbenjamin/TearingMacroFolder/runs/"+studyname+"_proc"+array_id+'/'

radas_dir_path="/home/sbenjamin/TearingMacroFolder/radas_dir"
inputpath="/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/"
dcon_executable_dir="/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/"
rdcon_executable_dir="/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/"
#plot_style_yaml="/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/plot_popcon.yaml"

#Read in scripts
tokamaker_python_path = "/home/sbenjamin/TearingMacroFolder/OFTBuilds/install_release/"
os.chdir("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/")
exec(open("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/random_start.py").read())

#Popcon options
current_frac_jump=0.05  #0.05
#Tokamaker options
max_high_def_iterations=3  #3
high_def_plasma_dx=0.002   #0.002
nr=900
nz=900
maxsteps=1000000
lcfs_pad=1e-5
ttol=1e-9
#DCON options
gse_err_logtol=0.0         #-0.5
psihigh=0.9999
req_mhd_stable=False
nmax=3                     #4 
grid_type_diagnose="""'pow1'"""
mpsi_diagnose=512           
qlow=1.01


#Best option: save equilibria but run suites at same time + save_total option engaged
out = gen_n_successful_cases(
            tot_cases,
            cases_in_batch=cases_in_batch,
            out_path=out_path,
            radas_dir_path=radas_dir_path,
            inputpath=inputpath,
            dcon_executable_dir=dcon_executable_dir,
            rdcon_executable_dir=rdcon_executable_dir,
            save=True,  

            # Loop options:
            save_equil=True, #I want to be able to re-run stab squite if needed
            return_total=True, #Will save total, less effort 
            run_phys_suite=False,

            # Tokamaker options:
            just_dynamo=True, #Ignore super unrealistic q0<<1 current profiles
            verbose=debug,
            back_down_current=True,
            run_tokamaker=True,

            # Popcon inputs: 
            current_frac_jump=current_frac_jump,
            # Tokamaker inputs: 
            max_high_def_iterations=max_high_def_iterations,
            high_def_plasma_dx=high_def_plasma_dx,
            nr=nr,
            nz=nz,
            maxsteps=maxsteps,
            lcfs_pad=lcfs_pad,
            ttol=ttol,
            # DCON inputs:
            gse_err_logtol=gse_err_logtol,
            psihigh=psihigh,
            grid_type_diagnose=grid_type_diagnose,
            mpsi_diagnose=mpsi_diagnose,
            qlow=qlow,
            nmax=nmax,
            req_mhd_stable=req_mhd_stable,

            #Plot/debug options:
            plot_popcons=True,
            plot_machine=True,
            plot_mesh=False,
            plot_coils=False,
            plot_iteration=False,
            )



#os.chdir("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/")
#exec(open("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/random_start.py").read())
#exec(open("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/run_local_RDCON.py").read())
#run_dcon_on_equilibria(out_path,dcon_executable_dir,rdcon_executable_dir,
#                gse_err_logtol=gse_err_logtol,
#                psihigh=psihigh,
#                grid_type_diagnose=grid_type_diagnose,
#                nmax=nmax)


if False:
    out = gen_n_successful_cases(tot_cases,
            out_path=out_path,
            working_dir=working_dir,
            dcon_executable_dir=dcon_executable_dir,
            rdcon_executable_dir=rdcon_executable_dir,
            radas_dir_path='/Users/sbenjamin/Desktop/TEARING_WIDGET/cfspopcon/radas_dir',
            inputpath="/Users/sbenjamin/Desktop/TEARING_WIDGET/scripts/",

            run_phys_suite=False,
            save_equil=True,
            return_total=True,save=True,just_dynamo=True,
            verbose=True,back_down_current=True,run_tokamaker=True,
            max_high_def_iterations=max_high_def_iterations,
            high_def_plasma_dx=high_def_plasma_dx,
            gse_err_logtol=gse_err_logtol,
            psihigh=psihigh,
            req_mhd_stable=False,
            nmax=nmax,
            nr=nr,nz=nz,maxsteps=maxsteps,lcfs_pad=lcfs_pad,ttol=ttol,
            plot_machine=False,plot_popcons=False,plot_mesh=False,plot_coils=False,plot_iteration=False,
            )
    


#cfspopcon home:
#/home/sbenjamin/.local/lib/python3.9/site-packages