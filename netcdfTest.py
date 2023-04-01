import netCDF4 as nc 
from scipy.io import netcdf

file2read = nc.Dataset('COSC3000\\sst.mon.ltm.1981-2010.nc','r')
print(file2read.__dict__)