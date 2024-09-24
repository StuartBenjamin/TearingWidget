#/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/random_start.py 
import sys
import os
import time
import subprocess
import shutil

debug=False

array_id=int(sys.argv[1])
if debug:
    outpathdb='/home/sbenjamin/TearingMacroFolder/runs/v1001_run1__proc10'
    #working_dir=outpathdb+"/dcon_working_dir"
    #if os.path.isfile(outpathdb+'/doccupied'):
    #    os.remove(outpathdb+'/doccupied')
    #dcon_dirs=[i for i in os.listdir(outpathdb) if "dcon_xrs_b" in i]
    #for i in dcon_dirs:
    #    print('hi ',outpathdb+'/'+i)
    #    shutil.rmtree(outpathdb+'/'+i)

print("Sleeping for 2x",array_id,"s")
time.sleep(2*array_id)
print("Sleeping complete")

radas_dir_path="/home/sbenjamin/TearingMacroFolder/radas_dir"
inputpath="/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/"
dcon_executable=''  #"/home/sbenjamin/TearingMacroFolder/GPEC/dcon/dcon"
rdcon_executable='' #"/home/sbenjamin/TearingMacroFolder/GPEC/rdcon/rdcon"

#Read in scripts
tokamaker_python_path = "/home/sbenjamin/TearingMacroFolder/OFTBuilds/install_release/"
exec(open("/home/sbenjamin/TearingMacroFolder/TearingWidget/scripts/random_start.py").read())

#Popcon options
#   current_frac_jump=0.05  #0.05
#Tokamaker options
#   max_high_def_iterations=3  #3
#   high_def_plasma_dx=0.002   #0.002
#   nr=900
#   nz=900
#   maxsteps=1000000
#   lcfs_pad=1e-5
#   ttol=1e-9
#   req_mhd_stable=False
#DCON options
gse_err_logtol=-0.5          #-0.5
psihigh=0.99999              #0.99999
nmax=4                       #4 
grid_type_diagnose="""'pow1'"""
mpsi_diagnose=800           #800
mpsi=mpsi_diagnose          #800
mtheta=2000                 #2000
qlow=1.01

if False: #deprecated debug
    os.chdir(outpathdb+"/dcon_working_dir")
    working_dir=outpathdb+"/dcon_working_dir"
    rdcon_run = subprocess.run(outpathdb+"/dcon_working_dir"+"/./dcon",shell=True)

if False: #second debug
    dcon_ran,rdcon_ran,MRE_xr=run_DCON_on_equilibrium2('/home/sbenjamin/TearingMacroFolder/runs/v1001_run1__proc10/eqdsks_b1of400/g_dyn_MCi0_Jfrac0p95',
                            working_dir=working_dir,dcon_executable=dcon_executable,
                            rdcon_executable=rdcon_executable,
            gse_err_logtol=gse_err_logtol,
            psihigh=psihigh,
            nmax=nmax,
            grid_type_diagnose=grid_type_diagnose,
            mpsi_diagnose=mpsi_diagnose,
            mpsi=mpsi,
            qlow=qlow,
            verbose=debug,
            mtheta=mtheta
            )

if False: #deprecated again
    run_dcon_on_equilibria(outpathdb,dcon_executable,rdcon_executable,
            gse_err_logtol=gse_err_logtol,
            psihigh=psihigh,
            nmax=nmax,
            grid_type_diagnose=grid_type_diagnose,
            mpsi_diagnose=mpsi_diagnose,
            mpsi=mpsi,
            qlow=qlow,
            verbose=debug,
            mtheta=mtheta
            )
    
runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
os.chdir(runs_dir)

if debug:
    print(os.listdir(runs_dir))

#Loop operation:
valid_dirs=1
while valid_dirs!=0:
    os.chdir(runs_dir)
    run_dirs=[i for i in os.listdir(runs_dir) if "_run" in i]
    if debug:
        print("Run ders = ",run_dirs[0]+", "+run_dirs[1]+", ",run_dirs[2]+"...")
        print("Total run_dirs = ",len(run_dirs))

    valid_for_dcon=False
    out_path=''
    for run_dir in run_dirs:
        num_validated_gfiles=0
        num_validated_dcon_outfiles=0

        in_run=os.listdir(run_dir)
        eqdsk_dirs=[i for i in in_run if "eqdsks_b" in i]
        dcon_dirs=[i for i in in_run if "dcon_xrs_b" in i]
        tokamak_xrs=[i for i in in_run if "tokamak_xr_" in i]

        xr_filenames=['tokamak_xr_'+j[8:] for j in eqdsk_dirs]

        for i in range(len(eqdsk_dirs)):
            if os.path.isfile(run_dir+"/"+xr_filenames[i]):
                num_validated_gfiles+=len(os.listdir(run_dir+"/"+eqdsk_dirs[i]))

        for i in dcon_dirs:
            num_validated_dcon_outfiles+=len(os.listdir(run_dir+"/"+i))
        
        print("Run dir: ",run_dir) 
        print("     num gfiles = ",num_validated_gfiles)
        print("     num dcon_xrs = ",num_validated_dcon_outfiles)
        occupied=os.path.isfile(run_dir+"/doccupied")
        print("     occupied = ",occupied)
        valid_for_dcon = (not occupied) and (len(tokamak_xrs)>0) and (num_validated_dcon_outfiles<num_validated_gfiles)

        if valid_for_dcon:
            out_path=runs_dir+'/'+run_dir
            break
        else: 
            print("Invalid, cycling...")
    
    if not debug and valid_for_dcon:
        print("Validity confirmed, running physics suite:")
        run_dcon_on_equilibria(out_path,dcon_executable,rdcon_executable,
            gse_err_logtol=gse_err_logtol,
            psihigh=psihigh,
            nmax=nmax,
            grid_type_diagnose=grid_type_diagnose,
            mpsi_diagnose=mpsi_diagnose,
            mpsi=mpsi,
            qlow=qlow,
            verbose=debug,
            mtheta=mtheta
            )
        continue
    elif debug:
        print("Example good out equilibrium: ,",out_path)
        run_dcon_on_equilibria(outpathdb,dcon_executable,rdcon_executable,
            gse_err_logtol=gse_err_logtol,
            psihigh=psihigh,
            nmax=nmax,
            grid_type_diagnose=grid_type_diagnose,
            mpsi_diagnose=mpsi_diagnose,
            mpsi=mpsi,
            qlow=qlow,
            verbose=debug,
            mtheta=mtheta
            )

    valid_dirs=0

