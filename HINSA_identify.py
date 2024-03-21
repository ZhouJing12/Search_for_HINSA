#!/usr/bin/env python
# coding: utf-8

# In[1]:


typical_size=3
typical_velocity_width=10

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.utils import data
from spectral_cube import SpectralCube
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
import copy
import os
import sys

choose_source=int(sys.argv[1])
source_name='sun_17_%d'%(choose_source)
info=np.loadtxt('sun_17_eq',dtype=str)
info_target=info[np.where(info[:,0]==str(choose_source))[0]][0]
hdul=fits.open(info_target[3])
cube = SpectralCube.read(hdul)
pixelsize=abs(hdul[0].header['CDELT1'])


# In[11]:

center=info_target[1:3].astype(float)
print(center)
#c=SkyCoord(ra=center[0]*u.deg,dec=center[1]*u.degree,frame='galactic')
ra_range = [center[0]-0.3, center[0]+0.3] * u.deg
dec_range = [center[1]-0.3,center[1]+0.3] * u.deg
#print(lon_range)
sub_cube = cube.subcube(xlo=ra_range[0], xhi=ra_range[1], ylo=dec_range[0], yhi=dec_range[1])
sub_cube_slab = sub_cube.spectral_slab((float(info_target[4])-30) *u.km / u.s,(float(info_target[4])+20) *u.km / u.s)
w = WCS(sub_cube_slab.header)
dz=sub_cube_slab.header['CDELT3']
nz=sub_cube_slab.header['NAXIS3']
nra=sub_cube_slab.header['NAXIS1']
ndec=sub_cube_slab.header['NAXIS2']
z_start=sub_cube_slab.header['CRVAL3']-(sub_cube_slab.header['CRPIX3']-1)*dz
vel=np.linspace(min(z_start+dz*(nz-1),z_start),max(z_start+dz*(nz-1),z_start),nz)
chan_center=(float(info_target[4])*1000-z_start)/dz
chan_width=max(abs(float(info_target[5])*1000/dz),8)
print(chan_width)
size=3.5

# In[38]:


from scipy.optimize import curve_fit
def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def triplegauss(x,a0,x0,sigma0,a1,x1,sigma1,a2,x2,sigma2):
    return gaussian(x,a0,x0,sigma0)+gaussian(x,a1,x1,sigma1)+gaussian(x,a2,x2,sigma2)
p0                   = ([50 , -60000 , 300   , 40   , -50000  , 300   ,60   ,-40000    , 300])   # peak , mean , width in Lsun/Hz , GHz , GHz
param_bounds         = ([40 , -100000 ,   0   , 20    ,  -100000 , 0     ,  20 ,  -100000  ,    0],
                        [100, -30000 , 10000 , 100  ,-30000   , 10000 ,100  ,-30000    , 10000])
shape=np.shape(sub_cube_slab)
center_pixel=[int((shape[1]-1)/2),int((shape[2]-1)/2)]
r=size/60/pixelsize

def get_region(center,r):
    position=[]
    background=[]
    for i in range(int(np.floor(center[0]-r)),int(np.floor(center[0]+r))+1):
        for j in range(int(np.floor(center[1]-r)),int(np.floor(center[1]+r))+1):
            if(np.sqrt((i-center[0])**2+(j-center[1])**2)<=r):
                position.append([i,j])
            if((np.sqrt((i-center[0])**2+(j-center[1])**2)>r)*(np.sqrt((i-center[0])**2+(j-center[1])**2)<1.5*r)):
                background.append([i,j])
    return position,background
region,bkg=get_region(center_pixel,r)
small_region,s_bkg=get_region(center_pixel,r/2.)

T = np.array(sub_cube_slab)

for point in region:
    T[round(chan_center-chan_width):round(chan_center+chan_width),point[0],point[1]]=np.nan

sub_cube_slab[:,5,9].quicklook()
result=np.zeros([9,shape[1],shape[2]])
gauss_1st=np.zeros(shape)


def real(spec):
    data_temp=np.zeros([len(spec),2])
    data_temp[:,0],data_temp[:,1]=vel,spec
    real_data=np.delete(data_temp,np.where(np.isnan(data_temp))[0],axis=0)
    return real_data
for i in range(shape[1]):
    for j in range(shape[2]):
        real_data=real(T[:,i,j])
        try:
            popt,pcov = curve_fit(triplegauss,real_data[:,0],real_data[:,1], p0=p0)
        except:
            popt=np.zeros(9)
            popt[:]=np.nan
        result[:,i,j]=popt
        gauss_1st[:,i,j]=gaussian(vel,result[0,i,j],result[1,i,j],result[2,i,j])

print("popt",popt)
#print("pcov",pcov)
def show_fit(spec):
    try:
        popt,pcov = curve_fit(triplegauss,vel, spec,p0=p0, bounds=param_bounds)
    except:
        return 0
    plt.clf()
    fig, ax_f = plt.subplots(1, sharex=True, sharey=False)
    ax_f.plot(vel/1000, spec, linewidth=1,  label=r'data')
    ax_f.plot(vel/1000, triplegauss(vel,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8]),color='red',linewidth=2,label='Overall fit',alpha=0.6)
    ax_f.plot(vel/1000, gaussian(vel,popt[0],popt[1],popt[2]),color='red',linewidth=0.8,label='Individual fit',alpha=0.6)
    ax_f.plot(vel/1000, gaussian(vel,popt[3],popt[4],popt[5]),color='red',linewidth=0.8,alpha=0.6)
    ax_f.plot(vel/1000, gaussian(vel,popt[6],popt[7],popt[8]),color='red',linewidth=0.8,alpha=0.6)
    plt.xlabel('km/s')
    plt.ylabel('Tb(K)')


# In[39]:


def draw(data):
    plt.figure()
    plt.subplot(projection=w[1,:,:])
    plt.imshow(data,cmap='jet',origin='lower')
    plt.colorbar()

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve
kernel = Gaussian2DKernel(x_stddev=2,y_stddev=2)
result_nontarget=copy.copy(result)
for point in region:
    result_nontarget[:,point[0],point[1]]=np.nan
for chan in range(9):
    result_nontarget[chan,:,:]=interpolate_replace_nans(result_nontarget[chan,:,:],kernel)
model_fit=np.zeros_like(T)
a=result_nontarget
for i in range(shape[1]):
    for j in range(shape[2]):
        model_fit[:,i,j]=triplegauss(vel,a[0,i,j],a[1,i,j],a[2,i,j],a[3,i,j],a[4,i,j],a[5,i,j],a[6,i,j],a[7,i,j],a[8,i,j])
os.system('rm fit_gauss/*%s_*.fits'%(source_name))
os.system('mkdir fit_gauss')
hduout=fits.PrimaryHDU(model_fit,header=sub_cube_slab.header)
hduout.writeto('fit_gauss/%s_model_fitg.fits'%(source_name))
origin_sub=np.array(sub_cube_slab)
residual_fitg=origin_sub-model_fit
hduout=fits.PrimaryHDU(residual_fitg,header=sub_cube_slab.header)
hduout.writeto('fit_gauss/%s_residual_fitg.fits'%(source_name))



# In[36]:


#initial substract

spectrum=np.zeros([shape[0],len(region)])
for i in range(len(region)):
    spectrum[:,i]=sub_cube_slab[:,region[i][0],region[i][1]]
target_spec=np.mean(spectrum,axis=1)

spectrum_bkg=np.zeros([shape[0],len(bkg)])
for i in range(len(bkg)):
    spectrum_bkg[:,i]=sub_cube_slab[:,bkg[i][0],bkg[i][1]]
bkg_spec=np.mean(spectrum_bkg,axis=1)


# In[26]:


model_image=(sub_cube_slab[round(chan_center-chan_width),:,:]+sub_cube_slab[round(chan_center+chan_width),:,:])/2

os.system('rm -r avg_model/*%s_*.fits'%(source_name))
os.system('mkdir avg_model')
hduout=fits.PrimaryHDU(np.array(model_image),header=sub_cube_slab.header)
hduout.writeto('avg_model/%s_model_origin.fits'%(source_name))
index=[]
data=[]
model_data=[]

cube_model=np.array(sub_cube_slab)
for chan in range(round(chan_center-chan_width),round(chan_center+chan_width)):
    target_data=sub_cube_slab[chan,:,:]
    for i in range(shape[1]):
        for j in range(shape[2]):
            if([i,j] not in region):
                index.append([i,j])
                data.append(target_data[i,j])
                model_data.append(model_image[i,j].value)
    param_bounds=([-50],[50])
    a=np.mean(np.array(data)-np.array(model_data))
    cube_model[chan,:,:]=model_image+a
origin_sub=np.array(sub_cube_slab)
residual=origin_sub-cube_model
draw(residual[round(chan_center),:,:])
os.system('rm absoption_map/%s_residual.pdf'%(source_name))
os.system('mkdir absoption_map')
plt.savefig('absoption_map/%s_residual.pdf'%(source_name))

# In[27]:


hdul[0].data=residual

hduout=fits.PrimaryHDU(cube_model,header=sub_cube_slab.header)
hduout.writeto('avg_model/%s_model.fits'%(source_name))
hduout=fits.PrimaryHDU(residual,header=sub_cube_slab.header)
hduout.writeto('avg_model/%s_residual.fits'%(source_name))


# In[29]:


from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
dec = linspace(dec_range[0] ,dec_range[1],ndec)
ra = linspace(ra_range[0],ra_range[1],nra)
T = np.array(sub_cube_slab)
print(chan_center,chan_width)

for point in region:
    T[round(chan_center-chan_width):round(chan_center+chan_width),point[0],point[1]]=np.nan
os.system('rm interpolate_model/*%s_*.fits'%(source_name))
os.system('mkdir interpolate_model')
hduout=fits.PrimaryHDU(T,header=sub_cube_slab.header)
hduout.writeto('interpolate_model/%s_nontarget.fits'%(source_name))
fn = RegularGridInterpolator((vel,ra,dec), T,method='nearest')

pts = array([[284,5,284],[-55000,5.1,284]])
#fn(pts)
cube_fit=np.zeros([round(chan_center+chan_width)-round(chan_center-chan_width),3])
for point in region:
    cube_fit[:,0]=vel[round(chan_center-chan_width):round(chan_center+chan_width)]
    cube_fit[:,1],cube_fit[:,2]=ra[point[0]],dec[point[1]]
    T[round(chan_center-chan_width):round(chan_center+chan_width),point[0],point[1]]=fn(array(cube_fit))
hduout=fits.PrimaryHDU(T,header=sub_cube_slab.header)
hduout.writeto('interpolate_model/%s_model_interp.fits'%(source_name))


# In[31]:



T = np.array(sub_cube_slab)
os.system('rm origin_sub/*%s_*.fits'%(source_name))
os.system('mkdir origin_sub')
hduout=fits.PrimaryHDU(T,header=sub_cube_slab.header)
hduout.writeto('origin_sub/%s_origin.fits'%(source_name))
for point in region:
    T[:,point[0],point[1]]=np.nan
for chan in range(nz):
    T[chan,:,:]=interpolate_replace_nans(T[chan,:,:],kernel)
os.system('rm convolve_model/*%s_*.fits'%(source_name))
os.system('mkdir convolve_model')
hduout=fits.PrimaryHDU(T,header=sub_cube_slab.header)
hduout.writeto('convolve_model/%s_model_con.fits'%(source_name))
residual_con=origin_sub-T
hduout=fits.PrimaryHDU(residual_con,header=sub_cube_slab.header)
hduout.writeto('convolve_model/%s_residual_con.fits'%(source_name))

os.system('rm 2Dinterpolate_spec/*%s_*.pdf'%(source_name))
os.system('mkdir 2Dinterpolate_spec')
plt.clf()
fig, ax_f = plt.subplots(1, sharex=True, sharey=False)
ax_f.step(vel/1000, residual_con[:,center_pixel[0],center_pixel[1]], linewidth=1,  label=r'data')
ax_f.axvline(float(info_target[4]),color='r')
ax_f.axvline(float(info_target[4])-float(info_target[5]),color='r',linestyle='dashed')
ax_f.axvline(float(info_target[4])+float(info_target[5]),color='r',linestyle='dashed')
ax_f.axhline(0,color='g')
plt.xlabel('v(km/s)')
plt.ylabel('Tb(K)')
plt.savefig('2Dinterpolate_spec/%s_con_spec.pdf'%(source_name))

os.system('rm 2Dinterpolate_spec_in_region/*%s_*.pdf'%(source_name))
os.system('mkdir 2Dinterpolate_spec_in_region')
plt.clf()
fig, ax_f = plt.subplots(1, sharex=True, sharey=False)
sum_spec=np.zeros(len(residual_con[:,0,0]))
for point in small_region:
    sum_spec+=residual_con[:,point[0],point[1]]
avg_spec=sum_spec/len(region)
ax_f.step(vel/1000, avg_spec, linewidth=1,  label=r'data')
ax_f.axvline(float(info_target[4]),color='r')
ax_f.axvline(float(info_target[4])-float(info_target[5]),color='r',linestyle='dashed')
ax_f.axvline(float(info_target[4])+float(info_target[5]),color='r',linestyle='dashed')
ax_f.axhline(0,color='g')
plt.xlabel('v(km/s)')
plt.ylabel('Tb(K)')
plt.savefig('2Dinterpolate_spec_in_region/%s_con_spec.pdf'%(source_name))

from astropy.convolution.kernels import CustomKernel
def Gaussian3DKernel(x_stddev,y_stddev=None,z_stddev=None):
    if y_stddev is None:
        y_stddev = x_stddev
    if z_stddev is None:
        z_stddev = x_stddev
    size=8*np.array([x_stddev,y_stddev,z_stddev])
    shape=2*size+1
    array=np.zeros([shape[0],shape[1],shape[2]])
    print(size,shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                array[i,j,k]=1./(2*np.pi * x_stddev * y_stddev * z_stddev)**1.5 * np.exp(-(((i-size[0])/x_stddev)**2 + ((j-size[1])/y_stddev)**2 + ((k-size[2])/z_stddev)**2)/2)
    kernel=CustomKernel(array)
    return kernel
kernel3D=Gaussian3DKernel(10,3,3)
T = np.array(sub_cube_slab)
for point in region:
    T[round(chan_center-chan_width):round(chan_center+chan_width),point[0],point[1]]=np.nan
T_inter=convolve(T,kernel3D)
os.system('rm con_inter_model/*%s_*.fits'%(source_name))
os.system('mkdir con_inter_model')
hduout=fits.PrimaryHDU(T_inter,header=sub_cube_slab.header)
hduout.writeto('con_inter_model/%s_model.fits'%(source_name))
residual_3D=origin_sub-T_inter
os.system('rm 3Dinterpolate_spec_in_region/*%s_*.pdf'%(source_name))
os.system('mkdir 3Dinterpolate_spec_in_region')
plt.clf()
fig, ax_f = plt.subplots(1, sharex=True, sharey=False)
sum_spec=np.zeros(len(residual_3D[:,0,0]))
for point in small_region:
    sum_spec+=residual_3D[:,point[0],point[1]]
avg_spec=sum_spec/len(region)
ax_f.step(vel[11:-11]/1000, avg_spec[11:-11], linewidth=1,  label=r'data')
ax_f.axvline(float(info_target[4]),color='r')
ax_f.axvline(float(info_target[4])-float(info_target[5]),color='r',linestyle='dashed')
ax_f.axvline(float(info_target[4])+float(info_target[5]),color='r',linestyle='dashed')
ax_f.axhline(0,color='g')
plt.xlabel('v(km/s)')
plt.ylabel('Tb(K)')
plt.savefig('3Dinterpolate_spec_in_region/%s_con_spec.pdf'%(source_name))

plt.clf()
fig, ax_f = plt.subplots(1, sharex=True, sharey=False)
avg_spec=sum_spec/len(region)
ax_f.step(vel[11:-11]/1000, residual_3D[11:-11,center_pixel[0],center_pixel[1]], linewidth=1,  label=r'data')
ax_f.axvline(float(info_target[4]),color='r')
ax_f.axvline(float(info_target[4])-float(info_target[5]),color='r',linestyle='dashed')
ax_f.axvline(float(info_target[4])+float(info_target[5]),color='r',linestyle='dashed')
ax_f.axhline(0,color='g')
plt.xlabel('v(km/s)')
plt.ylabel('Tb(K)')
os.system('rm 3Dinterpolate_spec/*%s_*.pdf'%(source_name))
os.system('mkdir 3Dinterpolate_spec')
plt.savefig('3Dinterpolate_spec/%s_con_spec.pdf'%(source_name))

draw(residual_3D[round(chan_center),4:-4,4:-4])
os.system('rm 3D_map/%s_residual.pdf'%(source_name))
os.system('mkdir 3D_map')
plt.savefig('3D_map/%s_residual.pdf'%(source_name))

