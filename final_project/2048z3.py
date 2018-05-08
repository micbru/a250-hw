import analysis

path_to_data = "/global/cscratch1/sd/zarija/Bispectrum/z3_2048.h5"
path_to_catalog = "./catalogs/catalog.txt"
output_mass_frac = "./2048z3/mass_fraction.txt"
output_WHIM_data = "./2048z3/WHIM_data.txt"

analysis.main(path_to_data,path_to_catalog,output_mass_frac,output_WHIM_data)
