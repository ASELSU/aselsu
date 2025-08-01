import os
import sys
from subprocess import check_call
import requests
import matplotlib as mpl
import gdown

def config_plot():
    nice_fonts = {
      "font.family": "serif",
      #"font.weight":"bold",
      "figure.titleweight":"bold",
      "figure.titlesize":20,
      "axes.labelsize": 14,
      "font.size": 16,
      "legend.fontsize": 10,
      "xtick.labelsize": 14,
      "ytick.labelsize": 14,
      }
    
    mpl.rcParams.update(nice_fonts)

def run_command(command):
    with open("log.txt", "a") as log_file:
        check_call(command, shell=True, stdout=log_file, stderr=log_file)

def load_environment_WP85():
    print('Environment loading....')
    run_command("pip install numpy==2.0.2")
    run_command("pip install xarray==2024.11.0")
    run_command("pip install matplotlib==3.9.0")
    run_command("pip install netCDF4 gdown palettable")
    run_command("pip install -q condacolab")
    import condacolab
    condacolab.install()
    run_command("conda install -c conda-forge esmpy -y")
    print('....loading....')
    run_command("conda install -c conda-forge xesmf -y")
    
    # Clone the repository and install lenapy
    run_command("git clone https://github.com/CNES/lenapy.git")
    run_command("pip install lenapy/.")
    print('.... Done')

def load_environment_WP83():
    print('Environment loading....')
    run_command("pip install netCDF4 gdown palettable")
    print('.... Done')
    
def load_data_WP85():
    print("Load GMSL timeserie\n")
    gmsl_file_id = "1sWWo6zlh3qYB13zOKneiklBWhsgT8XWW"
    gmsl_output = "MSL_aviso.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={gmsl_file_id}", gmsl_output, quiet=False)
    gmsl_file = gmsl_output

    print("\nLoad TOPEX-A correction\n")
    tpa_corr_file_id = "1e_r15fM16UwzmkqcxS4OUhDvrQ3U21fl"
    tpa_corr_output = "MSL_Aviso_Correction_GMSL_TPA.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={tpa_corr_file_id}", tpa_corr_output, quiet=False)
    tpa_corr_file = tpa_corr_output

    print("\nLoad Jason-3 correction\n")
    j3_corr_file_id = "1HQq52w2NrM8Xsm0Nsye4Q7BEHAJhUa07"
    j3_corr_output = "j3_wtc_drift_correction_cdr_al_s3a.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={j3_corr_file_id}", j3_corr_output, quiet=False)
    j3_corr_file = j3_corr_output

    print("\nLoad table of budget errors\n")
    error_budget_url = 'https://drive.google.com/uc?export=download&id=110SsJUTu3wBKhc6OHuun5bNDz08tm3eJ'
    error_prescription = 'error_budget_table.yaml'
    r = requests.get(error_budget_url)
    with open(error_prescription, 'wb') as f:
        f.write(r.content)
    
    return gmsl_file, tpa_corr_file, j3_corr_file, error_prescription

def load_environment_lenapy():
    print('Environment loading....')
    run_command("pip install numpy==2.0.2")
    run_command("pip install xarray==2024.11.0")
    run_command("pip install matplotlib==3.9.0")
    run_command("pip install netCDF4 gdown palettable")
    run_command("pip install -q condacolab")
    import condacolab
    condacolab.install()
    run_command("conda install -c conda-forge esmpy -y")
    print('....loading....')
    run_command("conda install -c conda-forge xesmf -y")
    
    # Clone the repository and install lenapy
    run_command("git clone https://github.com/CNES/lenapy.git")
    run_command("pip install lenapy/.")
   

def load_environment_lenapy_and_others():
	load_environment_lenapy()
	run_command("pip install dask==2025.5.1")
	run_command("pip install distributed==2025.5.1")
	run_command("pip install cartopy==0.24.1")

def load_environment_WP_WTC_CDR_global():
	load_environment_lenapy()
	print('.... Done')

def load_environment_WP_WTC_CDR_regional():
	load_environment_lenapy_and_others()
	print('.... Done')
 
def load_environment_WP81():
    load_environment_lenapy_and_others()
    print('.... Done')

def load_data_WTC_CDR_global():
    print("Load global WTC timeseries\n")
    wtc_global_file_id = "1KI28Gaj7Iql1VcZgLqHeOOZq5QmBudmx"
    wtc_global_output = "wtc_global.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={wtc_global_file_id}", wtc_global_output, quiet=False)
    wtc_global_file = wtc_global_output
    
    return wtc_global_file

def load_data_WTC_CDR_regional():
    print("Load regional WTC timeseries\n")
    wtc_regional_file_id = "1e_TLR6umCGBUGxx-MOPBhnGyt--NtLvk"
    wtc_regional_output = "wtc_regional.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={wtc_regional_file_id}", wtc_regional_output, quiet=False)
    wtc_regional_file = wtc_regional_output
    
    print("Load computed WTC trend uncertainties\n")
    wtc_trend_results_file_id = "1_LotsUbrNLdZOCS-iwElheOapOdfUoKC"
    wtc_trend_results_output = "wtc_regional_trend_results.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={wtc_trend_results_file_id}", wtc_trend_results_output, quiet=False)
    wtc_trend_results_file = wtc_trend_results_output
    
    return wtc_regional_file, wtc_trend_results_file

def load_data_WP81():
    print("Load regional SWH timeseries\n")
    ssb_regional_file_id = "1YQ9B0YvAaWcGi5QmqVuFpq8o9sOui0nM"
    ssb_regional_output = "ssb_regional_J3_1Hz.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={ssb_regional_file_id}", ssb_regional_output, quiet=False)
    ssb_regional_file = ssb_regional_output
    
    print("Load computed SSB trend uncertainties\n")
    # V1: constant global alpha and alpha uncertainty
    file_id = "1tAidKV-bUAmVAeozXUw-nasZ5cfMtLR8"
    file_output = "ssb_trend_results_alpha_global.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", file_output, quiet=False)
    ssb_regional_trend_results_fileV1 = file_output
    
    # V2: constant alpha and alpha uncertainty from alpha std
    file_id = "1fZlh6Vd5CVN3gixJzxUcZN7TsGyNNe34"
    file_output = "ssb_trend_results_alpha_meanV1.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", file_output, quiet=False)
    ssb_regional_trend_results_fileV2 = file_output
    
    # V3: constant alpha and alpha uncertainty from mean regression error
    file_id = "1nPwm4L20QVKpO97cyJMOOfbtg_Xz88TV"
    file_output = "ssb_trend_results_alpha_meanV2.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", file_output, quiet=False)
    ssb_regional_trend_results_fileV3 = file_output
    
    # V4 time varying alpha and alpha uncertainty, alpha uncertainty from regression error
    file_id = "1AIM35vt0oC00zCB-_wfD6cRTOpej6NeZ"
    file_output = "ssb_trend_results_time_varying_alpha.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", file_output, quiet=False)
    ssb_regional_trend_results_fileV4 = file_output
        
    return ssb_regional_file, ssb_regional_trend_results_fileV1, ssb_regional_trend_results_fileV2, ssb_regional_trend_results_fileV3, ssb_regional_trend_results_fileV4

def load_data_err_budget():
    print("Load GMSL timeserie\n")
    gmsl_file_id = "1sWWo6zlh3qYB13zOKneiklBWhsgT8XWW"
    gmsl_output = "MSL_aviso.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={gmsl_file_id}", gmsl_output, quiet=False)
    gmsl_file = gmsl_output

    print("\nLoad table of budget errors\n")
    error_budget_url = 'https://drive.google.com/uc?export=download&id=110SsJUTu3wBKhc6OHuun5bNDz08tm3eJ'
    error_prescription = 'error_budget_table.yaml'
    r = requests.get(error_budget_url)
    with open(error_prescription, 'wb') as f:
        f.write(r.content)

    print("\nLoad S6NG SLR-SUB\n")
    error_budget_url = 'https://drive.google.com/uc?export=download&id=1_vVcTz_Bkw4kzh1r1Ylwh3yE2wIQxH8C'
    error_prescription_s6ng = 'error_budget_table_s6ng.yaml'
    r = requests.get(error_budget_url)
    with open(error_prescription_s6ng, 'wb') as f:
        f.write(r.content)

    print("\nLoad S6NG SLR-SUB without GIA & POD\n")
    error_budget_url = 'https://drive.google.com/uc?export=download&id=1Gstxg7Y8bEdh9eR5gLJw6x_jqRCQ4LQh'
    error_prescription_only_s6ng = 'error_budget_table_only_s6ng.yaml'
    r = requests.get(error_budget_url)
    with open(error_prescription_only_s6ng, 'wb') as f:
        f.write(r.content)

    return gmsl_file, error_prescription, error_prescription_s6ng, error_prescription_only_s6ng

def load_data_WP83(mission, dlat=1):
    if mission=='J3-S6A':
        if dlat==1:
            print("Load J3 & S6A gridded data during tandem phase\n")
            tandem_file_id = "1rQGYWv8Gz7MTBwfk0mx7jIWGrq91V_9-"
            tandem_output = "J3-S6A_3_1_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={tandem_file_id}", tandem_output, quiet=False)
            tandem_file = tandem_output

            print("\nLoad J3 & S6A gridded data outside tandem phase\n")
            no_tandem_file_id = "1THOB-oS_zGHnnicvWGEKuEiJ9kJ_DGFQ"
            no_tandem_output = "J3-S6A_3_1_no_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={no_tandem_file_id}", no_tandem_output, quiet=False)
            no_tandem_file = no_tandem_output
        elif dlat==3:
            print("Load J3 & S6A gridded data during tandem phase\n")
            tandem_file_id = "19Vt1Y-q5jxVqObIQERygbwiNJUDfOICw"
            tandem_output = "J3-S6A_3_3_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={tandem_file_id}", tandem_output, quiet=False)
            tandem_file = tandem_output

            print("\nLoad J3 & S6A gridded data outside tandem phase\n")
            no_tandem_file_id = "1I0HNO5DzdQ0wGIPCyWwjOOK9Rzi5MZVN"
            no_tandem_output = "J3-S6A_3_3_no_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={no_tandem_file_id}", no_tandem_output, quiet=False)
            no_tandem_file = no_tandem_output
        else:
            print("\n*Error* Resolution incorrect, select dlat=1 or dlat=3\n")

    elif mission=='J2-J3':
        if dlat==1:
            print("Load J2 & J3 gridded data during tandem phase\n")
            tandem_file_id = "1k71IPlval_2_F0sTG-zVH1-1_rS-KHp5"
            tandem_output = "J2-J3_3_1_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={tandem_file_id}", tandem_output, quiet=False)
            tandem_file = tandem_output

            print("\nLoad J2 & J3 gridded data outside tandem phase\n")
            no_tandem_file_id = "1rUkbMzFrEIHUiHvD9SVDMX1hxHU0RKma"
            no_tandem_output = "J2-J3_3_1_no_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={no_tandem_file_id}", no_tandem_output, quiet=False)
            no_tandem_file = no_tandem_output
        elif dlat==3:
            print("Load J2 & J3 gridded data during tandem phase\n")
            tandem_file_id = "1ic8SPTyYLRQlraJxP-DU2SGsJp2-nYqw"
            tandem_output = "J2-J3_3_3_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={tandem_file_id}", tandem_output, quiet=False)
            tandem_file = tandem_output

            print("\nLoad J2 & J3 gridded data outside tandem phase\n")
            no_tandem_file_id = "1ZNA577nOxsHm_JMy1gED96bKcX1vikC1"
            no_tandem_output = "J2-J3_3_3_no_tandem.nc"
            gdown.download(f"https://drive.google.com/uc?export=download&id={no_tandem_file_id}", no_tandem_output, quiet=False)
            no_tandem_file = no_tandem_output
        else:
            print("\n*Error* Resolution incorrect, select dlat=1 or dlat=3\n")

    else:
        print("\n*Error* Missions name incorrect, select 'J2-J3' or 'J3-S6A'\n")

    return tandem_file, no_tandem_file

def load_data_S3A(mission,dlon=3,dlat=1):
    if mission=='J3-S6A':
        miss_comp = 'S3A-S6A'
        if dlat==1:
            print("Load S3A & S6A gridded data\n")
            file_id = "1AJzdwadFbXm1eKayRT-RNSms5O7yXT8L"
            no_tandem_output = "S3A-S6A_%i_%i_tandem.nc"%(dlon,dlat)
            gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", no_tandem_output, quiet=False)
        elif dlat==3:
            print("\nLoad S3A & S6A gridded data\n")
            file_id = "1S7qkXtdH5-ADPONndfefer98ZU3mDpEB"
            no_tandem_output = "S3A-S6A_%i_%i_no_tandem.nc"%(dlon,dlat)
            gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", no_tandem_output, quiet=False)
        else:
            print("\n*Error* Resolution incorrect, select dlat=1 or dlat=3\n")

    elif mission=='J2-J3':
        miss_comp = 'S3A-J3'
        if dlat==1:
            print("Load S3A & J3 gridded data\n")
            file_id = "13zk-4r5FgUAoAVuwMPCdlqZ9Iz_2TtN4"
            no_tandem_output = "S3A-J3_%i_%i_tandem.nc"%(dlon,dlat)
            gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", no_tandem_output, quiet=False)
        elif dlat==3:
            print("\nLoad S3A & J3 gridded data\n")
            file_id = "1l589Ezt0Bk81OKpQ01XCFfdmf_Ge_7eS"
            no_tandem_output = "S3A-J3_%i_%i_no_tandem.nc"%(dlon,dlat)
            gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", no_tandem_output, quiet=False)
        else:
            print("\n*Error* Resolution incorrect, select dlat=1 or dlat=3\n")
    else:
        print("\n*Error* Missions name incorrect, select 'J2-J3' or 'J3-S6A'\n")

    return no_tandem_output, miss_comp

def load_data_short_term():
    print("Load GMSL timeserie\n")
    gmsl_file_id = "1tHepuMtQz3uEB0alEdn9yck1TXC5_quk"
    gmsl_file = "gmsl_03_00_wtcradio.nc"
    gdown.download(f"https://drive.google.com/uc?export=download&id={gmsl_file_id}", gmsl_file, quiet=False)
  
    return gmsl_file
