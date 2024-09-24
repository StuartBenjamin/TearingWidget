import os
import numpy as np

#g_eqdk_filenames=[]
#eqdsk_dirstot=[]
#get_dcon_outfiles=[]
#Susdirs:
#   v1005_run3__proc10, v1005_run3__proc15, lost 237 cases between these two...
def count_cases():
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[i for i in os.listdir(runs_dir) if "_run" in i]
    print("Run ders = ",run_dirs[0]+", "+run_dirs[1]+", ",run_dirs[2]+"...")
    #print("Total run_dirs = ",len(run_dirs))
    tokamaks=0
    num_validated_dcon_outfiles=0
    num_dcon_xrs=0
    num_dcon_fails=0
    run_dirs_w_xrs=0
    dcon_completed_run_dirs=0
    g_eqdks_num=0
    for run_dir in run_dirs:
        print("Counting int ",run_dir)
        loc_dcon_xrs=0
        num_validated_gfiles=0
        in_run=os.listdir(run_dir)
        eqdsk_dirs=[i for i in in_run if "eqdsks_b" in i]
        dcon_dirs=[i for i in in_run if "dcon_xrs_b" in i]
        tokamak_xrs=[i for i in in_run if "tokamak_xr_" in i]
        if len(tokamak_xrs)>0:
            run_dirs_w_xrs+=1
        #eqdsk_dirstot.append(eqdsk_dirs)
        xr_filenames=['tokamak_xr_'+j[8:] for j in eqdsk_dirs]
        #print(xr_filenames)
        #print(eqdsk_dirs)
        for i in range(len(eqdsk_dirs)):
            if os.path.isfile(run_dir+"/"+xr_filenames[i]):
                mc_indexes=[]
                g_filenames=os.listdir(run_dir+"/"+eqdsk_dirs[i])
                g_eqdks_num+=len(g_filenames)
                for k in g_filenames:
                    num_validated_gfiles+=1
                    #g_eqdk_filenames.append(k)
                    k=k.split('_')
                    mc_indexes.append(k[2][3:])
                tokamaks+=len(set(mc_indexes))
        for i in dcon_dirs:
            dcon_filenames=os.listdir(runs_dir+'/'+run_dir+"/"+i)
            num_tot_dcon_files=len(dcon_filenames)
            num_fails=len([b for b in dcon_filenames if "xrF" in b])
            num_dcon_fails+=num_fails
            num_dcon_xrs+=num_tot_dcon_files-num_fails
            loc_dcon_xrs+=num_tot_dcon_files
            num_validated_dcon_outfiles+=num_tot_dcon_files
        #occupied=os.path.isfile(run_dir+"/doccupied")
        #valid_for_dcon = (not occupied) and (len(tokamak_xrs)>0) and (num_validated_dcon_outfiles<num_validated_gfiles)
        if loc_dcon_xrs==num_validated_gfiles:
            dcon_completed_run_dirs+=1
        print("     Num uniqe tokamaks = ",tokamaks)
        print("     Num equilibria = ",g_eqdks_num)
        print("     Num dcon runs = ",num_validated_dcon_outfiles,f" ({num_dcon_fails} failed, {num_dcon_xrs} successful)")
    print("Total run_dirs = ",run_dirs_w_xrs,f" ({dcon_completed_run_dirs} with dcon scans complete)")
    #print("Total equilibria directories = ",len(eqdsk_dirstot))
    print("Total unique tokamaks = ",tokamaks)
    print("Total equilibria = ",g_eqdks_num)
    print("Total dcon runs = ",num_validated_dcon_outfiles,f" ({num_dcon_fails} failed, {num_dcon_xrs} successful)")

import os
import shutil
def add_executables(occupiedTest=True,remove_files=False,remove_fails=False):
    dcon_executable="/home/sbenjamin/TearingMacroFolder/GPEC/dcon/dcon"
    rdcon_executable="/home/sbenjamin/TearingMacroFolder/GPEC/rdcon/rdcon"
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[runs_dir+'/'+i for i in os.listdir(runs_dir) if "_run" in i]
    for i in run_dirs:
        try:
            if not os.path.isdir(i+'/dcon_working_dir'):
                os.path.mkdir(i+'/dcon_working_dir')
            if (occupiedTest and (not os.path.isfile(i+'/doccupied'))) or (not occupiedTest):
                if remove_files and os.path.isfile(i+'/dcon_working_dir/dcon'):
                    os.remove(i+'/dcon_working_dir/dcon')
                if remove_files and os.path.isfile(i+'/dcon_working_dir/rdcon'):
                    os.remove(i+'/dcon_working_dir/rdcon') 
                shutil.copy(dcon_executable,i+'/dcon_working_dir/dcon')
                shutil.copy(rdcon_executable,i+'/dcon_working_dir/rdcon')
                if remove_fails:
                    tokdirs=[i+'/'+k for k in os.listdir(i) if "dcon_xrs_b" in k]
                    for k in tokdirs:
                        tokxrs=[k+'/'+j for j in os.listdir(k) if "_dcon_xrF" in j]
                        for z in tokxrs:
                            os.remove(z)
        except:
            print('skipping ',i)
            raise
            #if (i+'/dcon_working_dir')!='/home/sbenjamin/TearingMacroFolder/runs/v1001_run1__proc10/dcon_working_dir':
            #    raise

def add_executables_remove_fails(occupiedTest=True):
    dcon_executable="/home/sbenjamin/TearingMacroFolder/GPEC/dcon/dcon"
    rdcon_executable="/home/sbenjamin/TearingMacroFolder/GPEC/rdcon/rdcon"
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[runs_dir+'/'+i for i in os.listdir(runs_dir) if "_run" in i]
    for i in run_dirs:
        if not os.path.isdir(i+'/dcon_working_dir'):
            os.path.mkdir(i+'/dcon_working_dir')
        if not os.path.isfile(i+'/dcon_working_dir/dcon'):
            shutil.copy(dcon_executable,i+'/dcon_working_dir/dcon')
            if not os.path.isfile(i+'/dcon_working_dir/rdcon'):
                shutil.copy(rdcon_executable,i+'/dcon_working_dir/rdcon')
            tokdirs=[i+'/'+k for k in os.listdir(i) if "dcon_xrs_b" in k]
            for k in tokdirs:
                tokxrs=[k+'/'+j for j in os.listdir(k) if "_dcon_xrF" in j]
                for z in tokxrs:
                    os.remove(z)
            if os.path.isfile(i+'/doccupied'):
                os.remove(i+'/doccupied')

def remove_fails():
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[runs_dir+'/'+i for i in os.listdir(runs_dir) if "_run" in i]
    for i in run_dirs:
        tokdirs=[i+'/'+k for k in os.listdir(i) if "dcon_xrs_b" in k]
        for k in tokdirs:
            tokxrs=[k+'/'+j for j in os.listdir(k) if "_dcon_xrF" in j]
            for z in tokxrs:
                os.remove(z)
        if os.path.isfile(i+'/doccupied'):
            os.remove(i+'/doccupied')


"""
def clean_dcon_dirs():
    runs_dir='/home/sbenjamin/TearingMacroFolder/runs'
    os.chdir(runs_dir)
    run_dirs=[runs_dir+'/'+i for i in os.listdir(runs_dir) if "_run" in i]
    print("Run ders = ",run_dirs[0]+", "+run_dirs[1]+", ",run_dirs[2]+"...")
    print("Total run_dirs = ",len(run_dirs))
    for run_dir in run_dirs:
        if os.path.isdir(
        dcon_dirs=[run_dir+'/'+i for i in os.listdir(run_dir) if "dcon_xrs_b" in i]
        for i in dcon_dirs:
            dconfiles=os.listdir(i)
            for k in dconfiles:
                
                os.remove(i+'/'+k)

        if os.path.isfile(runs_dir+'/'+run_dirs+'/doccupied'):
            os.remove(runs_dir+'/'+run_dirs+'/doccupied')


tokamaks=0
for run_dir in run_dirs:
"""

"""GREAT DEBUG SESSION OF 2024
3 Leaks in dcon...
1: Non-GPEC Leaks
    1kb lost in internal processes 
        Part of this is inability of ZHETRS to omp_get_num_procs
            Going to try and define num threads
            OKAY IT WORKED
        Other part an issue with the package (I'm going to leave, clearly doesn't break anything)

2: Non fatal leaks
    q_find fix 24 bytes


3: Fatal leaks
    94kb leak during scan

DEBUG:
    [done] dcon ode/mer scan only [DONE]
        - singp reallocated without deallocating, but each time its linking to another location so its fine
            ^992 bytes, sing.f:308
    [done] dcon w free (no scan)
        - eliminated cond jump depending on ww... set it all to zero (does it break things?)
            ^vacuum_vac.f:636, 702/3/4
        - ignoring points overwriting allocation (assumed), 4 bytes per
            ^vacuum_global.f:137
        - ignoring repeated losses during printing of matrix (pestthingo, 512 per)
            ^vacuum_ma.f:71
        - got the singp overwrinting again, 992 bytes (ignoring)

    dcon w scan
    rdcon...    
        ode 
        gal 

"""
