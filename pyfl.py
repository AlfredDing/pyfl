#---------------------------------------------
#
#   Usefull tools for DNS data analysis
#
#   Created by Guang-Yu Alfred Ding
#   
#   Modification record:
#   Date                Change log
#   2016-11-23          Delete old version readfield_nas3d and readfield_nas3d_mean, 
#                       use new version readfield_nas3d and readfield_nas3d_mean functions,
#                       which import data in its original form (without shifting grid).
from pylab import *
from scipy import *
from numpy import *
import numpy as np
def cal_volume(dx,dy,dz):
    vol = zeros((len(dx),len(dy),len(dz)))
    vx  = np.matrix(dx)
    vy  = np.matrix(dy)
    for k in range(0,len(dz)):
        vol[:,:,k] = np.array(np.dot(vx.T,vy))*dz[k]
    return vol

def cal_area(dx,dy):
    vx  = np.matrix(dx)
    vy  = np.matrix(dy)
    area = np.array(np.dot(vx.T,vy))
    return area

def test_avail_memory(limit):
    #limit is in unit of GB
    import psutil
    mem = psutil.virtual_memory()
    if mem.available/1024**3 < limit:
        raise Exception('Warning: low available memory')
    else:
        print('Passed memory test.')
        return()

def read_avail_memory():
    # in unit of GB
    import psutil
    mem = np.array(psutil.virtual_memory().available/1024.0**3)
    return(mem)

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc
    else:
        print "Toc: start time not set"
        return()

def ilocalmax(f):
    f = np.array(f)
    if len(f)>3:
        df  = (f[2:]-f[1:-1])*(f[1:-1]-f[:-2])
        ddf = f[2:]+f[:-2]-2.0*f[1:-1]
        ind = arange(len(f)-2)[(df<0) & (ddf<0)]
        return ind+1
    else:
      raise NameError('Input data should have length larger than 3.')    
      return(False)
      
def localmax(f):
    f = np.array(f)
    if len(f)>3:
        df  = (f[2:]-f[1:-1])*(f[1:-1]-f[:-2])
        ddf = f[2:]+f[:-2]-2.0*f[1:-1]
        ind = f[1:-1][(df<0) & (ddf<0)]
        return ind
    else:
      raise NameError('Input data should have length larger than 3.')    
      return(False)
    
def gaussian2darray(r,n,l=1.0):
    x=np.linspace(-l/2,l/2,n)
#    y1=(6.0/pi/r**2)**1.5*exp(-6.0*(x*1.0/r)**2)
    y1=-np.sqrt(6.0/np.pi)/r*np.exp(-6.0*x**2/r**2)
    y=np.zeros([n,n])
    for i in range(0,n):
        for j in range(0,n):
            y[i,j]=y1[i]*y1[j]
    return y

def gaussian1darray(r,n,l=1.0):
    x=np.linspace(-l/2,l/2,n)
#    y1=(6.0/pi/r**2)**1.5*exp(-6.0*(x*1.0/r)**2)
    y=np.sqrt(6.0/np.pi)/r*np.exp(-6.0*x**2/r**2)
    return y
    
def gaussian3darray(r,n,l=1.0):
    import numpy as np
    x  = np.linspace(-l/2,l/2,n)
    y1 = np.matrix(np.sqrt(6.0/np.pi)/r*np.exp(-6.0*x**2/r**2))
    y0 = np.array(y1.T*y1)
    y  = np.zeros((n,n,n))
    y1 = np.array(y1)
    for j in range(0,n):
        y[:,:,j] = y1[:,j]*y0
    return y

def convolve_fft_ova(f1,f2,x):
    from scipy.signal import fftconvolve
    fx = np.zeros(np.shape(x))
    xt = np.zeros(np.shape(x))
    for i in range(0,np.size(x,axis=-1)):
        xt[:,:,i] = fftconvolve(f2,x[:,:,i],mode='same')
    for i in range(0,np.size(x,axis=0)):
        for j in range(0,np.size(x,axis=1)):
            fx[i,j,:] = fftconvolve(f1,xt[i,j,:],mode='same')
    return fx    
    
def plot_sst_HarISST1(lon,lat,sst,colormap='seismic',cm_range=[-5,35],dpara=20,dmeri=40):
    figure()
    if shape(shape(lon))!=(2,) or shape(shape(lat))!=(2,):
        lons,lats = meshgrid(lon,lat)
    else:
        lons=lon
        lats=lat
    m=Basemap(lat_1=lat[0],lat_2=lat[-1],lon_1=lon[0],lon_2=lon[-1],projection='cyl')
    m.drawcoastlines()
    #levels=linspace(-5,30,14)
    #f=m.contourf(lons,lats,sst[100,:,:],shading='flat',levels=levels,cmap='coolwarm',vmin=-5,vmax=30)
    m.pcolormesh(lons,lats,sst,shading='flat',cmap=colormap,vmin=cm_range[0],vmax=cm_range[1])
    m.fillcontinents(color='white',lake_color='None')
    parallel = arange(-80,100,dpara)
    m.drawparallels(parallel,labels=[True,True,False,False])
    meridian = arange(-180,180,dmeri)
    m.drawmeridians(meridian,labels=[1,0,0,1])

def plot_sic_HarISST1(lon,lat,sic,colormap='GnBu',cm_range=[0,1],dpara=20,dmeri=40):
    figure()
    if shape(shape(lon))!=(2,) or shape(shape(lat))!=(2,):
        lons,lats = meshgrid(lon,lat)
    else:
        lons=lon
        lats=lat
    m=Basemap(lat_1=lat[0],lat_2=lat[-1],lon_1=lon[0],lon_2=lon[-1],projection='cyl')
    m.drawcoastlines()
    #levels=linspace(-5,30,14)
    #f=m.contourf(lons,lats,sst[100,:,:],shading='flat',levels=levels,cmap='coolwarm',vmin=-5,vmax=30)
    m.pcolormesh(lons,lats,sic,shading='flat',cmap=colormap,vmin=cm_range[0],vmax=cm_range[1])
    m.fillcontinents(color='white',lake_color='None')
    parallel = arange(-80,100,dpara)
    m.drawparallels(parallel,labels=[True,True,False,False])
    meridian = arange(-180,180,dmeri)
    m.drawmeridians(meridian,labels=[1,0,0,1])

def cal_corrcoef(x,y,delta,ncal=[]):
    from scipy.stats import pearsonr
    from numpy import zeros
    y1   = x
    if ncal==[]:
        ncal=len(x)
    corr = zeros((ncal,))
    dt   = zeros((ncal,))
    y2   = zeros((len(x),))
    corr[0]=pearsonr(x,y)[0]
    for i in range(1,ncal):
        dt[i]=delta*i
        y2[:-i]=y[i:]
        y2[-i:]=y[:i]
        corr[i]=pearsonr(y1,y2)[0]
    return dt,corr

def first(x,op,val):
    x=np.array(x)
    filter_x=x[op(x,val)]
    if filter_x==[]:
        fx=[]
    else:
        fx=filter_x[0]
    return fx
    
def firstn(x,op,val):
    x=np.array(x)
    n=np.arange(0,len(x))
    filter_n=n[op(x,val)]
    if filter_n==[]:
        nx=[]
    else:
        nx=filter_n[0]
    return nx

def cal_vort_2d(u,w,xp,zp,lostagger=True):
    if shape(u) == shape(w):
        if lostagger:
            uc = (u[1:,1:]+u[:-1,1:])/2
            wc = (w[1:,1:]+w[1:,:-1])/2
            x  = xp[1:]
            z  = zp[1:]
        else:
            wc = w
            uc = u
            x  = xp
            z  = zp
        uz=caldf2d(uc,z,1)
        wx=caldf2d(wc,x,0)
        vort = wx-uz
    return vort

def cal_vort(u,w,v,xp,zp,yp,lostagger=True):
    if shape(u) == shape(v) and shape(v) == shape(w):
        if lostagger:
            uc = (u[1:,1:,1:]+u[:-1,1:,1:])/2
            wc = (w[1:,1:,1:]+w[1:,:-1,1:])/2
            vc = (v[1:,1:,1:]+v[1:,1:,:-1])/2
            x  = xp[1:]
            z  = zp[1:]
            y  = yp[1:]
        else:
            wc = w
            uc = u
            vc = v
            x  = xp
            y  = yp
            z  = zp
        uz = caldf3d(uc,z,1)
        uy = caldf3d(uc,y,2)
        vx = caldf3d(vc,x,0)
        vz = caldf3d(vc,z,1)
        wx = caldf3d(wc,x,0)
        wy = caldf3d(wc,y,2)
        vortz = uy-vx
        vorty = wx-uz
        vortx = vz-wy
        vort  = {'vortx':vortx, \
                 'vorty':vorty, \
                 'vortz':vortz}
    return vort


def cal_thermal_dissp(th,x,z,y):
    thx=caldf3d(th,x,0)
    thz=caldf3d(th,z,1)
    thy=caldf3d(th,y,2)
    dissp=thz*thz+thx*thx+thy*thy
    return dissp

def cal_thermal_dissp_vol(th,xp,zp,yp,xu,zw,yv):
    thx=caldf3d_volavg(th,xp[1:-1],xu[1:-2],0)
    thz=caldf3d_volavg(th,zp[1:-1],zw[1:-2],1)
    thy=caldf3d_volavg(th,yp[1:-1],yv[1:-2],2)
    dissp=thz*thz+thx*thx+thy*thy
    return dissp

def cal_thermal_dissp_2d(th,x,z):
    thx=caldf2d(th,x,0)
    thz=caldf2d(th,z,1)
    dissp=thz*thz+thx*thx
    return dissp

def cal_viscous_dissp_vol(u,w,v,xp,zp,yp,xu,zw,yv,lo_autofill=True):
    if shape(u) == shape(v) and shape(v) == shape(w):
        if lo_autofill:
            ue = np.zeros((u.shape[0]+1,u.shape[1],u.shape[2]))
            ue[1:,:,:] = u
            we = np.zeros((w.shape[0],w.shape[1]+1,w.shape[2]))
            we[:,1:,:] = w
            ve = np.zeros((v.shape[0],v.shape[1],v.shape[2]+1))
            ve[:,:,1:] = v
        else:
            ue = u
            we = w
            ve = v
        uc = (ue[1:,:,:]+ue[:-1,:,:])/2.0
        wc = (we[:,1:,:]+we[:,:-1,:])/2.0
        vc = (ve[:,:,1:]+ve[:,:,:-1])/2.0

        ux=caldf3d_volavg(ue,xp[1:-1],xu[ :-1],0,lo_stagger=True)
        uz=caldf3d_volavg(uc,zp[1:-1],zw[1:-2],1)
        uy=caldf3d_volavg(uc,yp[1:-1],yv[1:-2],2)
        vx=caldf3d_volavg(vc,xp[1:-1],xu[1:-2],0)
        vz=caldf3d_volavg(vc,zp[1:-1],zw[1:-2],1)
        vy=caldf3d_volavg(ve,yp[1:-1],yv[ :-1],2,lo_stagger=True)
        wx=caldf3d_volavg(wc,xp[1:-1],xu[1:-2],0)
        wz=caldf3d_volavg(we,zp[1:-1],zw[ :-1],1,lo_stagger=True)
        wy=caldf3d_volavg(wc,yp[1:-1],yv[1:-2],2)
        dissp = 2*(ux**2+vy**2+wz**2) \
                 +(uy+vx)**2 \
                 +(uz+wx)**2 \
                 +(vz+wy)**2
        #dissp=  ux**2+vx**2+wx**2 \
        #      + uy**2+vy**2+wy**2 \
        #      + uz**2+vz**2+wz**2
        return dissp


def cal_viscous_dissp(u,w,v,xp,zp,yp,lostagger=True,lo_autofill=True):
    if shape(u) == shape(v) and shape(v) == shape(w):
        if lostagger:
            if lo_autofill:
                ue = np.zeros((u.shape[0]+1,u.shape[1],u.shape[2]))
                ue[1:,:,:] = u
                we = np.zeros((w.shape[0],w.shape[1]+1,w.shape[2]))
                we[:,1:,:] = w
                ve = np.zeros((v.shape[0],v.shape[1],v.shape[2]+1))
                ve[:,:,1:] = v
                uc = (ue[1:,:,:]+ue[:-1,:,:])/2.0
                wc = (we[:,1:,:]+we[:,:-1,:])/2.0
                vc = (ve[:,:,1:]+ve[:,:,:-1])/2.0
                x  = xp
                y  = yp
                z  = zp
            else:
                uc = (u[1:,1:,1:]+u[:-1,1:,1:])/2
                wc = (w[1:,1:,1:]+w[1:,:-1,1:])/2
                vc = (v[1:,1:,1:]+v[1:,1:,:-1])/2
                x  = xp[1:]
                z  = zp[1:]
                y  = yp[1:]
        else:
            wc = w
            uc = u
            vc = v
            x  = xp
            y  = yp
            z  = zp
        ux=caldf3d(uc,x,0)
        uz=caldf3d(uc,z,1)
        uy=caldf3d(uc,y,2)
        vx=caldf3d(vc,x,0)
        vz=caldf3d(vc,z,1)
        vy=caldf3d(vc,y,2)
        wx=caldf3d(wc,x,0)
        wz=caldf3d(wc,z,1)
        wy=caldf3d(wc,y,2)
        dissp = 2*(ux**2+vy**2+wz**2) \
                 +(uy+vx)**2 \
                 +(uz+wx)**2 \
                 +(vz+wy)**2
        #dissp=  ux**2+vx**2+wx**2 \
        #      + uy**2+vy**2+wy**2 \
        #      + uz**2+vz**2+wz**2
        return dissp

def cal_viscous_dissp_2d(u,w,xp,zp,lostagger=True):
    if shape(u) == shape(w):
        if lostagger:
            uc = (u[1:,1:]+u[:-1,1:])/2
            wc = (w[1:,1:]+w[1:,:-1])/2
            x  = xp[1:]
            z  = zp[1:]
        else:
            wc = w
            uc = u
            x  = xp
            z  = zp
        ux=caldf2d(uc,x,0)
        uz=caldf2d(uc,z,1)
        wx=caldf2d(wc,x,0)
        wz=caldf2d(wc,z,1)
        dissp=  ux**2+wx**2 \
              + uz**2+wz**2
    return dissp

def caldf3d_volavg(f,xp,xu,ax=0,lo_stagger=False):
    from numpy import zeros
    xp = xp[:]
    xu = xu[:]
    if lo_stagger and len(xu)==len(xp)+1 and size(f,axis=ax)==len(xu):
        idx,fdx=cal_fd4o_vol(xp,xu)
        if ax==0:
            df = zeros((f.shape[0]-1,f.shape[1],f.shape[2]))
            for i in range(0,len(xp)):
                df[i,:,:]=fdx[i,0]*f[idx[i,0],:,:]+ \
                          fdx[i,1]*f[idx[i,1],:,:]+ \
                          fdx[i,2]*f[idx[i,2],:,:]+ \
                          fdx[i,3]*f[idx[i,3],:,:]
            return df
        if ax==1:
            df = zeros((f.shape[0],f.shape[1]-1,f.shape[2]))
            for i in range(0,len(xp)):
                df[:,i,:]=fdx[i,0]*f[:,idx[i,0],:]+ \
                          fdx[i,1]*f[:,idx[i,1],:]+ \
                          fdx[i,2]*f[:,idx[i,2],:]+ \
                          fdx[i,3]*f[:,idx[i,3],:]
            return df
        if ax==2:
            df = zeros((f.shape[0],f.shape[1],f.shape[2]-1))
            for i in range(0,len(xp)):
                df[:,:,i]=fdx[i,0]*f[:,:,idx[i,0]]+ \
                          fdx[i,1]*f[:,:,idx[i,1]]+ \
                          fdx[i,2]*f[:,:,idx[i,2]]+ \
                          fdx[i,3]*f[:,:,idx[i,3]]
            return df
        else:
            print('index error')
            return False
    elif len(xu)==len(xp)-1 and size(f,axis=ax)==len(xp):
        idxb,fdxb = cal_fd4o(xp)
        idx ,fdx  = cal_fd4o_vol(xu,xp)
        df = zeros((f.shape))
        if ax==0:
            i=0
            df[i,:,:]=fdxb[i,0]*f[idxb[i,0],:,:]+ \
                      fdxb[i,1]*f[idxb[i,1],:,:]+ \
                      fdxb[i,2]*f[idxb[i,2],:,:]+ \
                      fdxb[i,3]*f[idxb[i,3],:,:]
            i=-1
            df[i,:,:]=fdxb[i,0]*f[idxb[i,0],:,:]+ \
                      fdxb[i,1]*f[idxb[i,1],:,:]+ \
                      fdxb[i,2]*f[idxb[i,2],:,:]+ \
                      fdxb[i,3]*f[idxb[i,3],:,:]
            for i in range(1,size(f,axis=ax)-1):
                df[i,:,:]=   (fdx[i,0]*f[idx[i,0],:,:]+ \
                              fdx[i,1]*f[idx[i,1],:,:]+ \
                              fdx[i,2]*f[idx[i,2],:,:]+ \
                              fdx[i,3]*f[idx[i,3],:,:])*0.5 \
                           + (fdx[i-1,0]*f[idx[i-1,0],:,:]+ \
                              fdx[i-1,1]*f[idx[i-1,1],:,:]+ \
                              fdx[i-1,2]*f[idx[i-1,2],:,:]+ \
                              fdx[i-1,3]*f[idx[i-1,3],:,:])*0.5 
            return df
        if ax==1:
            i=0
            df[:,i,:]=fdxb[i,0]*f[:,idxb[i,0],:]+ \
                      fdxb[i,1]*f[:,idxb[i,1],:]+ \
                      fdxb[i,2]*f[:,idxb[i,2],:]+ \
                      fdxb[i,3]*f[:,idxb[i,3],:]
            i=-1
            df[:,i,:]=fdxb[i,0]*f[:,idxb[i,0],:]+ \
                      fdxb[i,1]*f[:,idxb[i,1],:]+ \
                      fdxb[i,2]*f[:,idxb[i,2],:]+ \
                      fdxb[i,3]*f[:,idxb[i,3],:]
            for i in range(1,size(f,axis=ax)-1):
                df[:,i,:]=   (fdx[i,0]*f[:,idx[i,0],:]+ \
                              fdx[i,1]*f[:,idx[i,1],:]+ \
                              fdx[i,2]*f[:,idx[i,2],:]+ \
                              fdx[i,3]*f[:,idx[i,3],:])*0.5 \
                           + (fdx[i-1,0]*f[:,idx[i-1,0],:]+ \
                              fdx[i-1,1]*f[:,idx[i-1,1],:]+ \
                              fdx[i-1,2]*f[:,idx[i-1,2],:]+ \
                              fdx[i-1,3]*f[:,idx[i-1,3],:])*0.5 
            return df
        if ax==2:
            i=0
            df[:,:,i]=fdxb[i,0]*f[:,:,idxb[i,0]]+ \
                      fdxb[i,1]*f[:,:,idxb[i,1]]+ \
                      fdxb[i,2]*f[:,:,idxb[i,2]]+ \
                      fdxb[i,3]*f[:,:,idxb[i,3]]
            i=-1
            df[:,:,i]=fdxb[i,0]*f[:,:,idxb[i,0]]+ \
                      fdxb[i,1]*f[:,:,idxb[i,1]]+ \
                      fdxb[i,2]*f[:,:,idxb[i,2]]+ \
                      fdxb[i,3]*f[:,:,idxb[i,3]]
            for i in range(1,size(f,axis=ax)-1):
                df[:,:,i]=   (fdx[i,0]*f[:,:,idx[i,0]]+ \
                              fdx[i,1]*f[:,:,idx[i,1]]+ \
                              fdx[i,2]*f[:,:,idx[i,2]]+ \
                              fdx[i,3]*f[:,:,idx[i,3]])*0.5 \
                           + (fdx[i-1,0]*f[:,:,idx[i-1,0]]+ \
                              fdx[i-1,1]*f[:,:,idx[i-1,1]]+ \
                              fdx[i-1,2]*f[:,:,idx[i-1,2]]+ \
                              fdx[i-1,3]*f[:,:,idx[i-1,3]])*0.5
            return df
        else:
            print('index error')
            return False
    else:
        raise Exception('Wrong input.')

def caldf3d(f,x,ax=0):
    from numpy import zeros
    x = x[:]
    df=zeros(shape(f))
    idx,fdx=cal_fd4o(x)
    if ax==0:
        for i in range(0,size(f,axis=ax)):
            df[i,:,:]=fdx[i,0]*f[idx[i,0],:,:]+ \
                      fdx[i,1]*f[idx[i,1],:,:]+ \
                      fdx[i,2]*f[idx[i,2],:,:]+ \
                      fdx[i,3]*f[idx[i,3],:,:]
        return df
    if ax==1:
        for i in range(0,size(f,axis=ax)):
            df[:,i,:]=fdx[i,0]*f[:,idx[i,0],:]+ \
                      fdx[i,1]*f[:,idx[i,1],:]+ \
                      fdx[i,2]*f[:,idx[i,2],:]+ \
                      fdx[i,3]*f[:,idx[i,3],:]
        return df
    if ax==2:
        for i in range(0,size(f,axis=ax)):
            df[:,:,i]=fdx[i,0]*f[:,:,idx[i,0]]+ \
                      fdx[i,1]*f[:,:,idx[i,1]]+ \
                      fdx[i,2]*f[:,:,idx[i,2]]+ \
                      fdx[i,3]*f[:,:,idx[i,3]]
        return df
    else:
        print('index error')
        return False

def caldf2d(f,x,ax=0):
    from numpy import zeros
    df=zeros(shape(f))
    idx,fdx=cal_fd4o(x)
    if ax==0:
        for i in range(0,size(f,axis=ax)):
            df[i,:]=fdx[i,0]*f[idx[i,0],:]+ \
                    fdx[i,1]*f[idx[i,1],:]+ \
                    fdx[i,2]*f[idx[i,2],:]+ \
                    fdx[i,3]*f[idx[i,3],:]
        return df
    if ax==1:
        for i in range(0,size(f,axis=ax)):
            df[:,i]=fdx[i,0]*f[:,idx[i,0]]+ \
                    fdx[i,1]*f[:,idx[i,1]]+ \
                    fdx[i,2]*f[:,idx[i,2]]+ \
                    fdx[i,3]*f[:,idx[i,3]]
        return df
    else:
        print('index error')
        return False
        
def caldf1d(f,x,ax=0):
    from numpy import zeros
    df=zeros(shape(f))
    idx,fdx=cal_fd4o(x)
    if ax==0:
        for i in range(0,size(f,axis=ax)):
            df[i]=fdx[i,0]*f[idx[i,0]]+ \
                  fdx[i,1]*f[idx[i,1]]+ \
                  fdx[i,2]*f[idx[i,2]]+ \
                  fdx[i,3]*f[idx[i,3]]
        return df
    else:
        print('index error')
        return False

def dev1(x1,x2,x3,y1,y2,y3,xp):
    import numpy as np
    from scipy.linalg import solve
    A=np.array([[x1**2,x1,1],[x2**2,x2,1],[x3**2,x3,1]])
    Y=np.array([y1,y2,y3])
    a,b,c=solve(A,Y)
    dy=2*a*xp+b
    return dy

def cal_fd4o_vol(xp,xu):
    from numpy import zeros
    from numpy import matrix
    if len(xu)==len(xp)+1:
        xu  = xu[:]
        xp  = xp[:]
        dxp = xu[1:]-xu[:-1]
        fdx=zeros((len(xp),4))
        idx=zeros((len(xp),4),int)
        #-- i = 0
        i = 0
        idx[i,:]   =range(0,4)
        x1,x2,x3,x4=xu[idx[i,:]]
        x0         =xp[i]
        dx         =dxp[i]
        X =matrix([[x1**3,x1**2,x1,1],
                   [x2**3,x2**2,x2,1],
                   [x3**3,x3**2,x3,1],
                   [x4**3,x4**2,x4,1]])
        Xd=1.0/dx*matrix([(x0+dx/2.0)**3-(x0-dx/2.0)**3,
                          (x0+dx/2.0)**2-(x0-dx/2.0)**2, 
                           dx, 
                           0])
        fdx[i,:]=array(Xd*X**-1)
        #-- i = -1
        i = -1
        idx[i,:]   =range(-4,0)
        x1,x2,x3,x4=xu[idx[i,:]]
        x0         =xp[i]
        dx         =dxp[i]
        X =matrix([[x1**3,x1**2,x1,1],
                   [x2**3,x2**2,x2,1],
                   [x3**3,x3**2,x3,1],
                   [x4**3,x4**2,x4,1]])
        Xd=1.0/dx*matrix([(x0+dx/2.0)**3-(x0-dx/2.0)**3,
                          (x0+dx/2.0)**2-(x0-dx/2.0)**2,
                           dx,
                           0])
        fdx[i,:]=array(Xd*X**-1)
        #-- i = -2
        #i = -2
        #idx[i,:]   =range(-4,0)
        #x1,x2,x3,x4=xu[idx[i,:]]
        #x0         =xp[i]
        #dx         =dxp[i]
        #X =matrix([[x1**3,x1**2,x1,1],
        #           [x2**3,x2**2,x2,1],
        #           [x3**3,x3**2,x3,1],
        #           [x4**3,x4**2,x4,1]])
        #Xd=1.0/dx*matrix([(x0+dx/2.0)**3-(x0-dx/2.0)**3,
        #                  (x0+dx/2.0)**2-(x0-dx/2.0)**2,
        #                   dx,
        #                   0])
        #fdx[i,:]=array(Xd*X**-1)
        #-- others
        for i in range(1,len(xp)-1):
            idx[i,:]   =range(i-1,i+3)
            x1,x2,x3,x4=xu[idx[i,:]]
            x0         =xp[i]
            dx         =dxp[i]
            X =matrix([[x1**3,x1**2,x1,1],
                       [x2**3,x2**2,x2,1],
                       [x3**3,x3**2,x3,1],
                       [x4**3,x4**2,x4,1]])
            Xd=1.0/dx*matrix([(x0+dx/2.0)**3-(x0-dx/2.0)**3,
                              (x0+dx/2.0)**2-(x0-dx/2.0)**2,
                               dx,
                               0])
            fdx[i,:]=array(Xd*X**-1)
        return idx,fdx
    else:
        raise Exception('please check input Xu and Xp. \
                         len(Xu)=len(Xp)+1 when lostagger=True.')

def cal_fd4o(x):
    x = x[:]
    from numpy import zeros
    from numpy import matrix
    fdx=zeros((len(x),4))
#    fcx=zeros((len(x),4))
    idx=zeros((len(x),4),int)
    #-- i = 0
    i = 0
    idx[i,:]   =range(0,4)
    x1,x2,x3,x4=x[idx[i,:]]
    X =matrix([[x1**3,x1**2,x1,1],
               [x2**3,x2**2,x2,1],
               [x3**3,x3**2,x3,1],
               [x4**3,x4**2,x4,1]])
    xd=matrix([3*x[i]**2, 2*x[i],  1,    0])
    fdx[i,:]=array(xd*X**-1)
    #-- i = 1
    i = 1
    idx[i,:]   =range(0,4)
    x1,x2,x3,x4=x[idx[i,:]]
    X =matrix([[x1**3,x1**2,x1,1],
               [x2**3,x2**2,x2,1],
               [x3**3,x3**2,x3,1],
               [x4**3,x4**2,x4,1]])
    xd=matrix([3*x[i]**2, 2*x[i],  1,    0])
    fdx[i,:]=array(xd*X**-1)
    #-- i = len(x)
    i = -1
    idx[i,:]   =range(-4,0)
    x1,x2,x3,x4=x[idx[i,:]]
    X =matrix([[x1**3,x1**2,x1,1],
               [x2**3,x2**2,x2,1],
               [x3**3,x3**2,x3,1],
               [x4**3,x4**2,x4,1]])
    xd=matrix([3*x[i]**2, 2*x[i],  1,    0])
    fdx[i,:]=array(xd*X**-1)
    #-- i = len(x)-1
    i = -2
    idx[i,:]   =range(-4,0)
    x1,x2,x3,x4=x[idx[i,:]]
    X =matrix([[x1**3,x1**2,x1,1],
               [x2**3,x2**2,x2,1],
               [x3**3,x3**2,x3,1],
               [x4**3,x4**2,x4,1]])
    xd=matrix([3*x[i]**2, 2*x[i],  1,    0])
    fdx[i,:]=array(xd*X**-1)
    #-- others
    for i in range(2,len(x)-2):
        idx[i,:]   =range(i-1,i+3)
        x1,x2,x3,x4=x[idx[i,:]]
        X =matrix([[x1**3,x1**2,x1,1],
                   [x2**3,x2**2,x2,1],
                   [x3**3,x3**2,x3,1],
                   [x4**3,x4**2,x4,1]])
#        xc=matrix([x[i]**3,   x[i]**2, x[i], 1])
        xd=matrix([3*x[i]**2, 2*x[i],  1,    0])
        fdx[i,:]=array(xd*X**-1)
#        fcx[i,:]=array(xc*X**-1)
    return idx,fdx
        
def plot_block(blk, nfig = None, color = 'w', fill = False, axes = None):
    import numpy as np
#    import matplotlib as plt
    if axes==None:
        fig = plt.figure(num = nfig)
        plt.axes()
    else:
        fig= axes
    for i in range(0,len(blk)):
        po = plt.Polygon(np.array(blk[i]).T, fc=color, fill = fill, axes = axes)
        if axes==None:
            plt.gca().add_patch(po)
        else:
            axes.add_patch(po)
    return fig

def read_block(geopath):
    import h5py
    fsimgeo = geopath + '/SIMGEO.h5'
    f       = h5py.File(fsimgeo, 'r')
    nb      = f['NB'][()]
    blk     = []
    for i in range(0,nb):
        blk = blk + [[f['BLK_'+'{0:0>2}'.format(i)+'/Zb'][()],
                      f['BLK_'+'{0:0>2}'.format(i)+'/Xb'][()]]]
    f.close()
    return blk

def plot_field(x,y,f,nfig=None,colormap='seismic',cm_range=[],\
               lo_contour=False,\
               level=8,line_tp='dashed',lv_array=[],\
               lo_tight=True,lo_colorbar=True, lo_label=False,\
               resxy=[],\
               old_axes = None, lo_fill = True, lo_sym=False):
    import numpy as np
    from scipy import integrate
    import matplotlib.pyplot as plt
    if len(cm_range)==0:
        if lo_contour:
            if lo_sym:
                minv = -np.amax([abs(np.amin(f)),np.amax(f)])
                maxv =  np.amax([abs(np.amin(f)),np.amax(f)])
                levels = np.linspace(minv,maxv,level)
            else:
                minv=None
                maxv=None
                levels = np.linspace(np.amin(f),np.amax(f),level)
        else:
            if lo_sym:
                minv = -np.amax([abs(np.amin(f)),np.amax(f)])
                maxv =  np.amax([abs(np.amin(f)),np.amax(f)])
            else:
                minv=None
                maxv=None
    else:
        minv=cm_range[0]
        maxv=cm_range[1]
        if lo_contour:
            levels = np.linspace(minv,maxv,level)
    
    xmax=np.amax(x)
    xmin=np.amin(x)
    ymax=np.amax(y)
    ymin=np.amin(y)
    lx=xmax-xmin
    ly=ymax-ymin
    
    if resxy!=[]:
        x,y,f=unify_xy(x,y,f,resxy[0],resxy[1])
    
    if lo_contour:
        if old_axes==None:
            if 5*lx/ly<15:
                if lo_colorbar:
                    fig = plt.figure(num=nfig,figsize=(5*lx/ly+1,5))
                else:
                    fig = plt.figure(num=nfig,figsize=(5*lx/ly,5))
            else:
                if lo_colorbar:
                    fig = plt.figure(num=nfig,figsize=(15+1,15*ly/lx))
                else:
                    fig = plt.figure(num=nfig,figsize=(15,15*ly/lx))
            axes = fig.add_subplot(111)
        #    fig = plt.figure(num=nfig,figsize=(6.5,5))
            axes = fig.add_subplot(111)
            axes.set_xticks([xmin,(xmin+xmax)/2.0,xmax])
            axes.set_yticks([ymin,(ymin+ymax)/2.0,ymax])
            axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if not(lo_label):
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            axes.set_ylim([ymin,ymax])
            axes.set_xlim([xmin,xmax])
            if lo_colorbar:
                colorbar()
            else:
                if lo_fill:
                    contourf(x,y,f,alpha=.75,cmap=colormap,levels=levels,axes=axes)
            contour(x,y,f,colors='black',linewidth=.5,linestyles=line_tp,levels=levels,axes=axes)       
        else:
            axes = old_axes
            if lo_colorbar:
                colorbar()
            else:
                if lo_fill:
                    contourf(x,y,f,alpha=.75,cmap=colormap,levels=levels,axes=old_axes)
            contour(x,y,f,colors='black',linewidth=.5,linestyles=line_tp,levels=levels,axes=old_axes)          
    else:
        if old_axes==None:
            if 5*lx/ly<15:
                if lo_colorbar:
                    fig = plt.figure(num=nfig,figsize=(5*lx/ly+1,5))
                else:
                    fig = plt.figure(num=nfig,figsize=(5*lx/ly,5))
            else:
                if lo_colorbar:
                    fig = plt.figure(num=nfig,figsize=(15+1,15*ly/lx))
                else:
                    fig = plt.figure(num=nfig,figsize=(15,15*ly/lx))
            axes = fig.add_subplot(111)
            axes = fig.add_subplot(111)
            axes.set_xticks([xmin,(xmin+xmax)/2.0,xmax])
            axes.set_yticks([ymin,(ymin+ymax)/2.0,ymax])
            axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if not(lo_label):
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            axes.set_ylim([ymin,ymax])
            axes.set_xlim([xmin,xmax])
            pcolor(x,y,f,cmap=colormap,vmin=minv,vmax=maxv,axes=axes)
        else:
            pcolor(x,y,f,cmap=colormap,vmin=minv,vmax=maxv,axes=old_axes)
        if lo_colorbar:
            colorbar()
    if lo_tight and axes==None:
        fig.tight_layout()
    if old_axes==None:
        return fig
    else:
        return axes
        
def plot_field_polar(r, phi, f, \
                     nfig=None,colormap='seismic',cm_range=[],\
                     lo_tight=True,lo_colorbar=True,resxy=[],\
                     old_axes = None, lo_mesh=False):
    """
    Plots a 2D array in polar coordinates.

    :param r: array of length n containing r coordinates
    :param phi: array of length m containing phi coordinates
    :param data: array of shape (n, m) containing the data to be plotted
    """
    if len(cm_range)==0:
        minv=None
        maxv=None
    else:
        minv=cm_range[0]
        maxv=cm_range[1]
    # Generate the mesh
    if resxy!=[]:
        r,phi,f=unify_xy(r,phi,f,resxy[0],resxy[1])
    phi_grid, r_grid = np.meshgrid(phi, r)
    x, y = r_grid*np.cos(phi_grid), r_grid*np.sin(phi_grid)
    if old_axes==None:
        if lo_colorbar:
            fig = plt.figure(num=nfig,figsize=(9,8))
        else:
            fig = plt.figure(num=nfig,figsize=(8,8))
        axes = fig.add_subplot(111,aspect='equal')
        axes.set_xticklabels([])
        axes.set_yticklabels([])
        plt.pcolormesh(x, y, f, cmap=colormap, vmin=minv,vmax=maxv,axes=axes)
    else:
        plt.pcolormesh(x, y, f, cmap=colormap, vmin=minv,vmax=maxv,axes=axes)
    if lo_colorbar:
                colorbar()
    if lo_tight and old_axes==None:
        fig.tight_layout()   
    if old_axes==None:
        return fig
    else:
        return axes    
        
def plot_uv(x,y,u,v,th=None,resx=50,resy=50,nfig=None,colormap='seismic', \
            lo_stream=False, arrowstyle='->', strm_den=1.5, 
            scale=None, lotick = True, width=None):
    xmax=np.amax(x)
    xmin=np.amin(x)
    ymax=np.amax(y)
    ymin=np.amin(y)
    lx=xmax-xmin
    ly=ymax-ymin

    if 5*lx/ly<15:
        fig = plt.figure(num=nfig,figsize=(5*lx/ly,5))
    else:
        fig = plt.figure(num=nfig,figsize=(15,15*ly/lx))
    axes = fig.add_subplot(111)
    if not(lotick):
        axes.set_xticklabels([])
        axes.set_yticklabels([])
    axes.set_ylim([ymin,ymax])
    axes.set_xlim([xmin,xmax])
    if lo_stream:
        ix,iy,iu=unify_xy(x,y,u,len(x),len(y))
        ix,iy,iv=unify_xy(x,y,v,len(x),len(y))
        streamplot(ix, iy, iu, iv, linewidth=0.5, color='black', density= strm_den, arrowsize=0.8,arrowstyle='->')
    else:
        ix,iy,iu=unify_xy(x,y,u,resx,resy)
        ix,iy,iv=unify_xy(x,y,v,resx,resy)
        if th==None:
            if scale==None:
                quiver(ix,iy,iu,iv,cmap=colormap, units='x')
            else:
                quiver(ix, iy, iu, iv, cmap=colormap,scale=scale, \
                       width = width, headwidth = 2.5, headlength = 5, units='width')
        else:
            ix,iy,ith=unify_xy(x,y,th,resx,resy)
            if scale==None:
                quiver(ix,iy,iu,iv,ith, cmap=colormap, units='x')
            else:
                quiver(ix, iy, iu, iv, ith, cmap=colormap,scale=scale, \
                       width = width, headwidth = 2.5, headlength = 5, units='width')
        
    return fig

def plot_line(f, axis, colors='k', linestyles='dashed'):
    ax = plt.gca()
    if axis=='x':
        ylim = ax.get_ylim()
        ax.vlines(f, ylim[0], ylim[1], colors=colors, linestyles=linestyles)
    else:
        xlim = ax.get_xlim()
        ax.hlines(f, xlim[0], xlim[1], colors=colors, linestyles=linestyles)
    

def unify_xy(x,y,p,nix,niy,minx=[],maxx=[],miny=[],maxy=[]):
    from pylab import linspace
    from scipy.interpolate import interp2d
    lenx=len(x)
    leny=len(y)
    if minx==[]:
        minx=amin(x)
    if maxx==[]:
        maxx=amax(x)
    if miny==[]:
        miny=amin(y)
    if maxy==[]:
        maxy=amax(y)
    ix=linspace(minx,maxx,nix)
    iy=linspace(miny,maxy,niy)
    
    if shape(p)[1]==lenx and shape(p)[0]==leny:
        f  = interp2d(x,y,p)
        ip = f(ix,iy)

        #ip=zeros((nix,niy))
        #ir=zeros((lenx,niy))
        #
        #for i in range(0,lenx):
        #    ir[i,:]=interp(iy,y,p[i,:])
        #
        #for i in range(0,niy):
        #    ip[:,i]=interp(ix,x,ir[:,i])
            
    elif shape(p)[1]==leny and shape(p)[0]==lenx:
        f  = interp2d(y,x,p)
        ip = f(iy,ix)
        #ip=zeros((niy,nix))
        #ir=zeros((leny,nix))
        #
        #for i in range(0,leny):
        #    ir[i,:]=interp(ix,x,p[i,:])
        #
        #for i in range(0,nix):
        #    ip[:,i]=interp(iy,y,ir[:,i])
    
    else:
        ix=[]
        iy=[]
        ip=[]
        disp('Dimensions of p are not agree with the length of x,y')
    
    return ix,iy,ip
    
def unify_xyz(z,x,y,p,niz,nix,niy,minz=[],maxz=[],minx=[],maxx=[],maxy=[],miny=[]):
    from pylab import linspace
    from scipy.interpolate import interp1d
#    lenx=len(x)
#    leny=len(y)
    if minx==[]:
        minx=amin(x)
    if maxx==[]:
        maxx=amax(x)
    if miny==[]:
        miny=amin(y)
    if maxy==[]:
        maxy=amax(y)
    if minz==[]:
        minz=amin(z)
    if maxz==[]:
        maxz=amax(z)
    ix=linspace(minx,maxx,nix)
    iy=linspace(miny,maxy,niy)
    iz=linspace(minz,maxz,niz)
    fz=interp1d(z,p,axis=0,kind='cubic',fill_value='extrapolate')
    ir1=fz(iz)
    fx=interp1d(x,ir1,axis=1,kind='cubic',fill_value='extrapolate')
    ir2=fx(ix)
    fy=interp1d(y,ir2,axis=2,kind='cubic',fill_value='extrapolate')
    ip=fy(iy)
    
    return iz,ix,iy,ip
    
def intp_3d(p,x,z,y,xs,zs,ys):
    from pylab import linspace
    from scipy.interpolate import interp1d
    fx=interp1d(x,p,axis=0,kind='cubic',fill_value='extrapolate')
    ir1=fx(xs)
    fz=interp1d(z,ir1,axis=1,kind='cubic',fill_value='extrapolate')
    ir2=fz(zs)
    fy=interp1d(y,ir2,axis=2,kind='cubic',fill_value='extrapolate')
    ps=fy(ys)
    return ps

def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        print(path+' mkdir successed')
        os.makedirs(path)
        return True
    else:
        print(path+' document exists')
        return False
        
#def parameters(directory,filename):
#      import simio
#      import h5py
##      filename=directory+'OUT0000.h5'
#      file = h5py.File(filename,'r')
#      imap2= simio.h5dims(file,'IMAP2')
#      kmap2= simio.h5dims(file,'KMAP2')
#
#      Xu = simio.h5grid(file,'Xu',imap2)
#      dXp= simio.h5grid(file,'dXp',imap2)
#      Zw = simio.h5grid(file,'Zw',kmap2)
#      dZp= simio.h5grid(file,'dZp',kmap2)
#      file.close()
#      return Xu,dXp,Zw,dZp
#
#def parameters_mean(directory):
#      import simio
#      import h5py
#      import glob
#      import os
#      filename=directory+'OUT0000.h5'
#      if not os.path.exists(filename):
#          filename  = sorted(glob.glob(directory+'OUT*.*.h5'))[0]
#      file = h5py.File(filename,'r')
#      imap2= simio.h5dims(file,'IMAP2')
#      kmap2= simio.h5dims(file,'KMAP2')
#
#      Xu = simio.h5grid(file,'Xu',imap2)
#      dXp= simio.h5grid(file,'dXp',imap2)
#      Zw = simio.h5grid(file,'Zw',kmap2)
#      dZp= simio.h5grid(file,'dZp',kmap2)
#      file.close()
#      return Xu,dXp,Zw,dZp
#     
#
#def parameters3d(directory,filename):
#      import simio
#      import h5py
##      filename=directory+'OUT0000.h5'
#      file = h5py.File(filename,'r')
#      imap2= simio.h5dims(file,'IMAP2')
#      jmap2= simio.h5dims(file,'JMAP2')
#      kmap2= simio.h5dims(file,'KMAP2')
#
#      Xu = simio.h5grid(file,'Xu',imap2)
#      dXp= simio.h5grid(file,'dXp',imap2)
#      Yv = simio.h5grid(file,'Yv',jmap2)
#      dYp= simio.h5grid(file,'dYp',jmap2)
#      Zw = simio.h5grid(file,'Zw',kmap2)
#      dZp= simio.h5grid(file,'dZp',kmap2)
#      file.close()
#      return Xu,dXp,Yv,dYp,Zw,dZp
#
#def parameters3d_mean(directory, multi_res = False):
#      import simio
#      import h5py
#      import glob
#      import os
#      filename=directory+'OUT0000.h5'
#      if not os.path.exists(filename):
#          filename  = sorted(glob.glob(directory+'OUT*.*.h5'))[0]
#      file = h5py.File(filename,'r')
#      imap2= simio.h5dims(file,'IMAP2')
#      jmap2= simio.h5dims(file,'JMAP2')
#      kmap2= simio.h5dims(file,'KMAP2')
#
#      Xu = simio.h5grid(file,'Xu',imap2)
#      dXp= simio.h5grid(file,'dXp',imap2)
#      Yv = simio.h5grid(file,'Yv',jmap2)
#      dYp= simio.h5grid(file,'dYp',jmap2)
#      Zw = simio.h5grid(file,'Zw',kmap2)
#      dZp= simio.h5grid(file,'dZp',kmap2)
#      file.close()
#      return Xu,dXp,Yv,dYp,Zw,dZp
      
def readXZ(filename):
      import h5py
      file = h5py.File(filename,'r')
      U = file['U2']#(file,'U'+name)
      W = file['W2']#h5readxz(file,'W'+name)
      T = file['T2']#h5readxz(file,'T'+name)
      return U,W,T
      file.close()
      del file


def readXYZ(filename):
      import h5py
      file = h5py.File(filename,'r')
      U = file['U2']#(file,'U'+name)
      W = file['W2']#h5readxz(file,'W'+name)
      V = file['V2']#h5readxz(file,'W'+name)
      T = file['T2']#h5readxz(file,'T'+name)
      return U,W,V,T
      file.close()
      del file
      
def readXYZ_mean(filename):
      import h5py
      file = h5py.File(filename,'r')
      U = file['Umean']#(file,'U'+name)
      W = file['Wmean']#h5readxz(file,'W'+name)
      V = file['Vmean']#h5readxz(file,'W'+name)
      T = file['Tmean']#h5readxz(file,'T'+name)
      return U,W,V,T
      file.close()
      del file

#def readfield_nas(geopath,filename, \
#                  pressure_flg = False):
#    import simio
#    flowfield = {'X':[],   'Z':[], \
#             'U':[],   'W':[], \
#             'T':[],   'P':[]}
#    Xu,dXp,Zw,dZp = parameters(geopath,filename)
#    Xp,dXu = simio.calgeo(Xu,dXp)
#    Zp,dZw = simio.calgeo(Zw,dZp)
#    Us,Ws,T = readXZ(filename)
#    U=array(Us)
#    W=array(Ws)
#    U[0,:]=0
#    U[-1,:]=0
#    U[:,0]=0
#    U[:,-1]=0
#    W[0,:]=0
#    W[-1,:]=0
#    W[:,0]=0
#    W[:,-1]=0
#    UC=(U[1:-1,:]+U[:-2,:])/2
#    WC=(W[:,1:-1]+W[:,:-2])/2  
#    flowfield['X']=Xp[1:-1]
#    flowfield['Z']=Zp[1:-1]
#    flowfield['U']=UC[:,1:-1]
#    flowfield['W']=WC[1:-1,:]
#    flowfield['T']=T[1:-1,1:-1]
#    return flowfield

def read_probe(filename, grpname, varname):
    import h5py
    varful = grpname+'/'+varname
    file = h5py.File(filename,'r')
    var  = file[varful]
    return var

def read_slice(filename, \
              pressure_flg = False, \
              slice_flg = 'xz', \
              grp = '/', \
              multi_res = False, \
              lo_2d  = True, \
              lo_ibm  = False, \
              lo_time = False, \
              cylinder = False, geopath = []):
    import h5py
    file = h5py.File(filename,'r')
    if geopath != []:
        simgeo = h5py.File(geopath+'/SIMGEO.h5','r')
    else:
        simgeo = file
    flowfield = {'Xp':[],  'Zp':[],  'Yp':[], \
                 'Xu':[],  'Zw':[],  'Yv':[], \
                 'dXp':[], 'dZp':[], 'dYp':[], \
                 'dXu':[], 'dZw':[], 'dYv':[], \
                 'Xs':[],  'Zs':[],  'Ys':[], \
                 'Xus':[], 'Zws':[], 'Yvs':[],\
                 'dXs':[], 'dZs':[], 'dYs':[],\
                 'dXus':[],'dZws':[],'dYvs':[],\
                 'U':[],   'W':[], 'V':[],\
                 'T':[],   'P':[] }
    if slice_flg == 'xz':
        if multi_res:
            flowfield['Xs']  = simgeo['Xs'] 
            flowfield['Zs']  = simgeo['Zs'] 
            flowfield['dXs'] = simgeo['dXs'] 
            flowfield['dZs'] = simgeo['dZs'] 
            flowfield['Xus'] = simgeo['Xus'] 
            flowfield['Zws'] = simgeo['Zws'] 
            flowfield['dXus']= simgeo['dXus'] 
            flowfield['dZws']= simgeo['dZws']
        flowfield['Xp'] = simgeo['Xp'] 
        flowfield['Zp'] = simgeo['Zp'] 
        flowfield['dXp']= simgeo['dXp'] 
        flowfield['dZp']= simgeo['dZp'] 
        flowfield['Xu'] = simgeo['Xu'] 
        flowfield['Zw'] = simgeo['Zw'] 
        flowfield['dXu']= simgeo['dXu'] 
        flowfield['dZw']= simgeo['dZw'] 
        flowfield['U']  = file[grp+'U2'] 
        flowfield['W']  = file[grp+'W2'] 
        if not lo_2d:
            flowfield['V']  = file[grp+'V2']
        flowfield['T']  = file[grp+'T2'] 
        if lo_ibm:
            flowfield['VFU'] = file['VFU']
            flowfield['VFW'] = file['VFW']
            flowfield['VFT'] = file['VFT']
        if lo_time:
            flowfield['Time'] = file['TPROBL']
        return flowfield
    if slice_flg == 'xy':
        if multi_res:
            flowfield['Xs']  = simgeo['Xs']
            flowfield['dXs'] = simgeo['dXs']
            flowfield['Xus'] = simgeo['Xus']
            flowfield['dXus']= simgeo['dXus']
            if not(cylinder):
                flowfield['Ys']  = simgeo['Ys']
                flowfield['dYs'] = simgeo['dYs']
                flowfield['Yvs'] = simgeo['Yvs']
                flowfield['dYvs']= simgeo['dYvs']
            else:
                flowfield['dYs'] = simgeo['dYs']
        flowfield['Xp'] = simgeo['Xp']
        flowfield['dXp']= simgeo['dXp']
        flowfield['Xu'] = simgeo['Xu']
        flowfield['dXu']= simgeo['dXu']
        if not(cylinder):
            flowfield['Yp']  = simgeo['Yp']
            flowfield['dYp'] = simgeo['dYp']
            flowfield['Yv']  = simgeo['Yv']
            flowfield['dYv'] = simgeo['dYv']
        else:
            flowfield['dY']  = simgeo['dY']
        flowfield['U']  = file[grp+'U2']
        flowfield['W']  = file[grp+'W2']
        flowfield['V']  = file[grp+'V2']
        flowfield['T']  = file[grp+'T2']
        if lo_ibm:
            flowfield['VFU'] = file['VFU']
            flowfield['VFW'] = file['VFW']
            flowfield['VFT'] = file['VFT']
        if lo_time:
            flowfield['Time'] = file['TPROBL']
        return flowfield
    if slice_flg == 'yz':
        if multi_res:
            flowfield['Zs']  = simgeo['Zs']
            flowfield['dZs'] = simgeo['dZs']
            flowfield['Zws'] = simgeo['Zws']
            flowfield['dZws']= simgeo['dZws']
            if not(cylinder):
                flowfield['Ys']  = simgeo['Ys']
                flowfield['dYs'] = simgeo['dYs']
                flowfield['Yvs'] = simgeo['Yvs']
                flowfield['dYvs']= simgeo['dYvs']
            else:
                flowfield['dYs'] = simgeo['dYs']
        flowfield['Zp'] = simgeo['Zp']
        flowfield['dZp']= simgeo['dZp']
        flowfield['Zw'] = simgeo['Zw']
        flowfield['dZw']= simgeo['dZw']
        if not(cylinder):
            flowfield['Yp']  = simgeo['Yp']
            flowfield['dYp'] = simgeo['dYp']
            flowfield['Yv']  = simgeo['Yv']
            flowfield['dYv'] = simgeo['dYv']
        else:
            flowfield['dY']  = simgeo['dY']
        flowfield['U']  = file[grp+'U2']
        flowfield['W']  = file[grp+'W2']
        flowfield['V']  = file[grp+'V2']
        flowfield['T']  = file[grp+'T2']
        if lo_ibm:
            flowfield['VFU'] = file['VFU']
            flowfield['VFW'] = file['VFW']
            flowfield['VFT'] = file['VFT']
        if lo_time:
            flowfield['Time'] = file['TPROBL']
        return flowfield
       
def read_slice_mean(geopath, \
                   dissp_flg = False, pressure_flg = False, \
                   ut_flg    = False, u2_flg       = False, u3_flg  = False, \
                   multi_res = False):
    import simio
    import h5py
    import numpy as np
    filename = geopath+'MEAN.h5'
    file = h5py.File(filename,'r')
    flowfield = {'Xp':[],  'Zp':[],\
                 'Xu':[],  'Zw':[],\
                 'dXp':[],'dZp':[],\
                 'dXu':[],'dZw':[],\
                 'U':[],   'W':[], \
                 'T':[],   'P':[], \
                 'epu':[],  'ept':[], \
                 'U2':[],   'W2':[],\
                 'U3':[],   'W3':[],\
                 'UT':[],   'WT':[],}
    flowfield['Xp'] = file['Xp']
    flowfield['Zp'] = file['Zp']
    flowfield['dXp']= file['dXp']
    flowfield['dZp']= file['dZp']
    flowfield['Xu'] = file['Xu']
    flowfield['Zw'] = file['Zw']
    flowfield['dXu']= file['dXu']
    flowfield['dZw']= file['dZw']
    if dissp_flg: 
        flowfield['ept'] = file['epstmean']
        flowfield['epu'] = file['epsvmean']
    if pressure_flg:
        flowfield['P']   = file['Pmean']
    if ut_flg:
        flowfield['UT']  = file['UTmean']
        flowfield['WT']  = file['WTmean']
    if u2_flg:
        flowfield['U2']  = file['U2mean']
        flowfield['W2']  = file['W2mean']
        flowfield['T2']  = file['T2mean']
    if u3_flg:
        flowfield['U3']  = file['U3mean']
        flowfield['W3']  = file['W3mean']
        flowfield['T3']  = file['T3mean']
    flowfield['U']=file['Umean']
    flowfield['W']=file['Wmean']
    flowfield['T']=file['Tmean']
    if multi_res:
        flowfield['Xs']  = file['Xs']  
        flowfield['Zs']  = file['Zs']  
        flowfield['dXs'] = file['dXs'] 
        flowfield['dZs'] = file['dZs'] 
        flowfield['Xus'] = file['Xus'] 
        flowfield['Zws'] = file['Zws'] 
        flowfield['dXus']= file['dXus']
        flowfield['dZws']= file['dZws']
    return flowfield
    
def plot_mesh(Arr_xi,Arr_yi, ns = 1):
    Arr_x = Arr_xi[:]
    Arr_y = Arr_yi[:]
    #Arr_y: y-direction data, 1D numpy array or list.
    for j in range(0,len(Arr_y),ns):
            plt.plot([Arr_x.min(), Arr_x.max()],[Arr_y[j], Arr_y[j]], color = 'black')
    
    #Arr_x: x-direction data, 1D numpy array or list.
    for i in range(0,len(Arr_x),ns):
            plt.plot([Arr_x[i],Arr_x[i]], [Arr_y.min(), Arr_y.max()], color = 'black')
    
    plt.xlim(Arr_x.min(), Arr_x.max())
    plt.ylim(Arr_y.min(), Arr_y.max())  
    
def read_field(geopath,filename,\
                    lo_pressure = False, lo_multires = False,\
                    lo_cylinder = False,field_only = False, lo_tc=False):
    #import simio
    import h5py
    import pandas as pd
    file   = h5py.File(filename,'r')
    simgeo = h5py.File(geopath+'/SIMGEO.h5','r')
    flowfield = {}
    if not(field_only):
        if lo_multires:
            flowfield['Xs']  = simgeo['Xs']
            flowfield['Zs']  = simgeo['Zs']
            flowfield['dXs'] = simgeo['dXs']
            flowfield['dZs'] = simgeo['dZs']
            flowfield['Xus'] = simgeo['Xus']
            flowfield['Zws'] = simgeo['Zws']
            flowfield['dXus']= simgeo['dXus']
            flowfield['dZws']= simgeo['dZws']
            if not(lo_cylinder):
                flowfield['Ys']  = simgeo['Ys']
                flowfield['dYs'] = simgeo['dYs']
                flowfield['Yvs'] = simgeo['Yvs']
                flowfield['dYvs']= simgeo['dYvs']
            else:
                flowfield['dYs']  = simgeo['dYs']
                flowfield['Ys']  = arange(-2, len(simgeo['JMAS'])+2)*flowfield['dYs'].value
            if lo_tc:
                flowfield['Tc'] = file['Tc']
#            flowfield['Us']  = np.array(file['Us'])
#            flowfield['Vs']  = np.array(file['Vs'])
#            flowfield['Ws']  = np.array(file['Ws'])
#            flowfield['Tc']  = np.array(file['Tc'])
        flowfield['Xp'] = simgeo['Xp']
        flowfield['Zp'] = simgeo['Zp']
        flowfield['dXp']= simgeo['dXp']
        flowfield['dZp']= simgeo['dZp']
        flowfield['Xu'] = simgeo['Xu']
        flowfield['Zw'] = simgeo['Zw']
        flowfield['dXu']= simgeo['dXu']
        flowfield['dZw']= simgeo['dZw']
        if not(lo_cylinder):
            flowfield['Yp'] = simgeo['Yp']
            flowfield['dYp']= simgeo['dYp']
            flowfield['Yv'] = simgeo['Yv']
            flowfield['dYv']= simgeo['dYv']
        else:
            flowfield['dY'] = simgeo['dY']
            flowfield['Y']  = arange(-2, len(simgeo['JMA'])+2)*flowfield['dY'].value
    flowfield['U']  = file['U2']
    flowfield['W']  = file['W2']
    flowfield['V']  = file['V2']
    flowfield['T']  = file['T2']
    # the valuable return are z,x,y,w,u,v,th
    return flowfield
    
def read_field_mean(geopath, \
                         lo_dissp = False, lo_pressure = False, \
                         lo_ut    = False, lo_u2       = False, lo_u3  = False, \
                         lo_multires = False, lo_cylinder = False):
    import simio
    import h5py
    import numpy as np
    filename = geopath+'MEAN.h5'
    file = h5py.File(filename,'r')
    simgeo = h5py.File(geopath+'/SIMGEO.h5','r')
    if lo_multires:
        flowfield = {'Xp':[],  'Zp':[], 'Yp':[],\
                     'Xu':[],  'Zw':[], 'Yv':[],\
                     'dXp':[],'dZp':[], 'dYp':[],\
                     'dXu':[],'dZw':[], 'dYv':[],\
                     'Xs':[],  'Zs':[],  'Ys':[],\
                     'Xus':[], 'Zws':[], 'Yvs':[],\
                     'dXs':[], 'dZs':[], 'dYs':[],\
                     'dXus':[],'dZws':[],'dYvs':[],\
                     'U':[],   'W':[], 'V':[],\
                     'T':[],   'P':[], \
                     'epu':[],  'ept':[], \
                     'U2':[],   'W2':[],  'V2':[],\
                     'U3':[],   'W3':[],  'V3':[],\
                     'UT':[],   'WT':[],  'VT':[]}
        flowfield['Xs']  = simgeo['Xs']
        flowfield['Zs']  = simgeo['Zs']  
        flowfield['dXs'] = simgeo['dXs'] 
        flowfield['dZs'] = simgeo['dZs'] 
        flowfield['Xus'] = simgeo['Xus'] 
        flowfield['Zws'] = simgeo['Zws'] 
        flowfield['dXus']= simgeo['dXus']
        flowfield['dZws']= simgeo['dZws']
        if not(lo_cylinder):
            flowfield['Ys']  = simgeo['Ys']
            flowfield['dYs'] = simgeo['dYs']
            flowfield['Yvs'] = simgeo['Yvs']
            flowfield['dYvs']= simgeo['dYvs']
        else:
            flowfield['dYs']  = simgeo['dYs']
            flowfield['Ys']  = arange(-2, len(simgeo['JMAS'])+2)*flowfield['dYs'].value
    else:
        flowfield = {'Xp':[],  'Zp':[],'Yp':[],\
                     'Xu':[],  'Zw':[],'Yv':[],\
                     'U':[],   'W':[], 'V':[],\
                     'dXp':[],'dZp':[],'dYp':[],\
                     'dXu':[],'dZw':[],'dYv':[],\
                     'T':[],   'P':[], \
                     'epu':[],  'ept':[], \
                     'U2':[],   'W2':[],  'V2':[],\
                     'U3':[],   'W3':[],  'V3':[],\
                     'UT':[],   'WT':[],  'VT':[]}
    flowfield['Xp'] = simgeo['Xp']
    flowfield['dXp']= simgeo['dXp']
    flowfield['Zp'] = simgeo['Zp']
    flowfield['dZp']= simgeo['dZp']
    flowfield['Xu'] = simgeo['Xu']
    flowfield['Zw'] = simgeo['Zw']
    flowfield['dXu']= simgeo['dXu']
    flowfield['dZw']= simgeo['dZw']
    if not(lo_cylinder):
        flowfield['Yp'] = simgeo['Yp']
        flowfield['dYp']= simgeo['dYp']
        flowfield['Yv'] = simgeo['Yv']
        flowfield['dYv']= simgeo['dYv']
    else:
        flowfield['dY'] = simgeo['dY']
        flowfield['Y']  = arange(-2, len(simgeo['JMA'])+2)*flowfield['dY'].value

    if lo_dissp: 
        flowfield['ept'] = file['ditmean']
        flowfield['epu'] = file['dikmean']
    if lo_pressure:
        flowfield['P']   = file['Pmean']
    if lo_ut:
        flowfield['UT']  = file['UTmean']
        flowfield['WT']  = file['WTmean']
        flowfield['VT']  = file['VTmean']
    if lo_u2:
        flowfield['U2']  = file['U2mean']
        flowfield['W2']  = file['W2mean']
        flowfield['V2']  = file['V2mean']
        flowfield['T2']  = file['T2mean']
    if lo_u3:
        flowfield['U3']  = file['U3mean']
        flowfield['W3']  = file['W3mean']
        flowfield['V3']  = file['V3mean']
        flowfield['T3']  = file['T3mean']
    flowfield['U']=file['Umean']
    flowfield['W']=file['Wmean']
    flowfield['V']=file['Vmean']
    flowfield['T']=file['Tmean']
    return flowfield

def save_var_csv(filepath, var, varname=[], flg_index=False):
    import numpy as np
    import csv
    import pandas as pd
    if varname==[] and type(var)==dict:
        dc  = var
        df = pd.DataFrame.from_dict(dc, orient = 'index')
        df = df.transpose()
    elif len(varname)==len(var):
        dc = {varname[i]:var[i] for i in range(0,len(varname))}
        df = pd.DataFrame.from_dict(dc, orient = 'index')
        df = df.transpose()
    elif type(var)==pd.core.frame.DataFrame:
        df = var
    else:
        raise Exception('Wrong input type. Please make sure var is dictionary/DataFrame, or varname and var with same shape.')
    return df.to_csv(filepath, index=flg_index)
    
def save_var_excel(filepath, var, varname=[], flg_index=False, sheet_name='Sheet1'):
    import numpy as np
    import csv
    import pandas as pd
    if varname==[] and type(var)==dict:
        dc  = var
        df = pd.DataFrame.from_dict(dc, orient = 'index')
        df = df.transpose()
    elif len(varname)==len(var):
        dc = {varname[i]:var[i] for i in range(0,len(varname))}
        df = pd.DataFrame.from_dict(dc, orient = 'index')
        df = df.transpose()
    elif type(var)==pd.core.frame.DataFrame:
        df = var
    else:
        raise Exception('Wrong input type. Please make sure var is dictionary/DataFrame, or varname and var with same shape.')
    return df.to_excel(filepath, index=flg_index, sheet_name=sheet_name)

def save_var_pickle(filepath, var):
    import pickle
    import pandas
    f  = open(filepath, 'wb')
    if type(var)==pandas.core.frame.DataFrame:
        var = {col_name : var[col_name].values for col_name in var.columns.values}
    pickle.dump(var, f)
    f.close()
    return

def load_var_pickle(filepath):
    import pickle
    f   = open(filepath, 'rb')
    var = pickle.load(f)
    return var 

def save_var_hdf5(filepath, var):
    import h5py
    import pandas
    import os
    if os.path.exists((filepath)):
        os.remove(filepath)
    f  = h5py.File(filepath, 'a')
    if type(var)==pandas.core.frame.DataFrame:
        var = {col_name : var[col_name].values for col_name in var.columns.values}
    for varitem in var.items():
        f.create_dataset(varitem[0], data = varitem[1])
    f.close()

def load_var_hdf5(filepath):
    import h5py
    f = h5py.File(filepath, 'r')
    return(f)

def save_var_mat(filepath, var):
    from scipy.io import savemat
    import pandas
    f = open(filepath, 'wb')
    if type(var)==pandas.core.frame.DataFrame:
        var = {col_name : var[col_name].values for col_name in var.columns.values}
    savemat(f,var)
    f.close()
    return

def load_var_mat(filepath):
    from scipy.io import loadmat
    f = open(filepath, 'rb')
    var = loadmat(f)
    f.close()
    return var
    
def readfield_tc(geopath, filenames):
    import h5py
    simgeo = h5py.File(geopath+'/continua_master.h5','r')
    flowfield = {}
    flowfield['R'] = simgeo['ym']
    flowfield['Azim'] = simgeo['xm']
    flowfield['Z']    = simgeo['zm']
    q1 = h5py.File(filenames[0], 'r')
    flowfield['Uazim'] = q1['var']
    q2 = h5py.File(filenames[1], 'r')
    flowfield['Ur']    = q2['var']
    q3 = h5py.File(filenames[2], 'r')
    flowfield['Uz']    = q3['var']
    return flowfield

def readfield_tcmean(geopath, statpath):
    import h5py
    simgeo = h5py.File(geopath+'/stafield_master.h5','r')
    flowfield = {}
    flowfield['R'] = simgeo['R_cordin']
    flowfield['Azim'] = simgeo['Azim_cordin']
    flowfield['Z']    = simgeo['Z_cordin']
    flowfield['Navg'] = simgeo['averaging_time']
    dissp  = h5py.File(statpath+'/disste.h5','r')
    flowfield['dissp'] = dissp['disste']
    prmean  = h5py.File(statpath+'/pr_mean.h5','r')
    flowfield['prmean'] = prmean['pr_mean']
    prrms  = h5py.File(statpath+'/pr_rms.h5','r')
    flowfield['prrms']  = prrms['pr_rms']
    q1mean = h5py.File(statpath+'/q1_mean.h5','r')
    flowfield['Uazimmean'] = q1mean['q1_mean']
    q1q2mean = h5py.File(statpath+'/q1q2_mean.h5','r')
    flowfield['UazimUrmean'] = q1q2mean['q1q2_mean']
    q1rms = h5py.File(statpath+'/q1_rms.h5')
    flowfield['Uazimrms'] = q1rms['q1_rms']
    q2mean = h5py.File(statpath+'/q2_mean.h5','r')
    flowfield['Urmean'] = q2mean['q2_mean']
    q2rms = h5py.File(statpath+'/q2_rms.h5','r')
    flowfield['Urrms'] = q2rms['q2_rms']
    q3mean = h5py.File(statpath+'/q3_mean.h5','r')
    flowfield['Uzmean'] = q3mean['q3_mean']
    q3rms = h5py.File(statpath+'/q3_rms.h5','r')
    flowfield['Uzrms'] = q3rms['q3_rms']
    return flowfield
    
def vectorize(x):
    if x.ndim == 1:
        return x
    else:
        return x.reshape((-1,))

def combine_vecs(th,u,w,v=[]):
    from numpy import array
    comb = []
    comb.extend(vectorize(th))
    comb.extend(vectorize(u))
    comb.extend(vectorize(w))
    if v is not None:
        comb.extend(vectorize(v))
    return array(comb)
    
def seperate_vecs(comb,tshape,ushape):
    from numpy import prod
    if comb.size == 2*prod(ushape)+prod(tshape):
        th = comb[:prod(tshape)].T.reshape(tshape)
        u  = comb[prod(tshape):prod(tshape)+prod(ushape)].T.reshape(ushape)
        w  = comb[prod(tshape)+prod(ushape):].T.reshape(ushape)
        return th,u,w
    elif comb.size == 3*prod(ushape)+prod(tshape): 
        th = comb[:prod(tshape)].T.reshape(tshape)
        u  = comb[prod(tshape):prod(tshape)+prod(ushape)].T.reshape(ushape)
        w  = comb[prod(tshape)+prod(ushape):prod(tshape)+2*prod(ushape)].T.reshape(ushape)
        v  = comb[prod(tshape)+2*prod(ushape):].T.reshape(ushape)
        return th,u,w,v
    else:
        raise Exception('Wrong array shape')

def prettify(elem):
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent=' ')

def write_xdmf(fieldpath, filename, geopath=[]):
    if geopath == []:
        geopath=fieldpath
    geofile = h5py.File(geopath,'r')
    lenx = str(geofile['Xp'].shape[0])
    leny = str(geofile['Yp'].shape[0])
    lenz = str(geofile['Zp'].shape[0])
    dims_topol = "{0} {1} {2}".format(lenx,lenz,leny)
    # --- Create Element ---
    root   = Element('Xdmf')
    root.set('version', '2.0')
    domain = SubElement(root, 'Domain')
    ## --- Define Geometry ---
    grid   = SubElement(domain, 'Grid', {'Name':'Structured Grid',
                                         'GridType':'Uniform'})
    topo   = SubElement(grid, 'Topology', {'Type':'3DRectMesh',
                                           'Dimensions':dims_topol})
    geo    = SubElement(grid, 'Geometry', {'Type':"VXVYVZ"})
    x_data = SubElement(geo, 'DataItem', {'Name':"Xp",
                                          'Dimensions':lenx,
                                          'NumberType':"Float",
                                          'Precision':"4",
                                          'Format':"HDF"})
    x_data.text = geopath+":/Xp"
    y_data = SubElement(geo, 'DataItem', {'Name':"Zp",
                                          'Dimensions':lenz,
                                          'NumberType':'Float',
                                          'Precision':'4',
                                          'Format':'HDF'})
    y_data.text = geopath+':/Zp'
    z_data = SubElement(geo, 'DataItem', {'Name':"Yp",
                                          'Dimensions':leny,
                                          'NumberType':'Float',
                                          'Precision':'4',
                                          'Format':'HDF'})
    z_data.text = geopath+':/Yp'
    # --- Define Fields ---
    scal   = SubElement(grid, 'Attribute', {'Name':"Temperature",
                                             'AttributeType':"Scalar",
                                             'Center':"Node"})
    temp   = SubElement(scal, 'DataItem', {'Dimensions':dims_topol,
                                           'Numbertype':'Float',
                                           'Precision':'4',
                                           'Format':'HDF'})
    temp.text = fieldpath+':/T2'
    # --- Write Xdmf ---
    with open(filename,'w') as f:
        f.write(prettify(root))

