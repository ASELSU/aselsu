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


def load_data_WP83(mission):
    if mission=='J3-S6A':
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

    elif mission=='J2-J3':
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