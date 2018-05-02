import analysis

path_to_data = "/global/cscratch1/sd/zarija/4096/z05.h5"
path_to_catalog = '/global/cscratch1/sd/zarija/4096/catalog_z05_iso138.txt'
output_mass_frac = "./4096z05/mass_fraction.txt"
output_WHIM_data = "./4096z05/WHIM_data.txt"

analysis.main(path_to_data,path_to_catalog,output_mass_frac,output_WHIM_data)
