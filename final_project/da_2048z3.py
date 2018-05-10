import da_analysis as da

path_to_data = "/global/cscratch1/sd/zarija/Bispectrum/z3_2048.h5"
path_to_catalog = "./catalogs/catalog.txt"
output_mass_frac = "./2048z3/da_mass_fraction.txt"
output_WHIM_data = "./2048z3/da_WHIM_data.txt"

da.main(path_to_data,path_to_catalog,output_mass_frac,output_WHIM_data)
