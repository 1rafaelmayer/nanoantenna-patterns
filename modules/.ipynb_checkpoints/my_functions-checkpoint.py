import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.optimize import minimize

def epsilon_(w,e_inf_II,w_TO_II,gama_II,w_LO_II,e_inf_L,w_TO_L,gama_L,w_LO_L):
    """This function calculates the dielectric function of hBN using drude-Lorentz oscillators.

        Ab inition Calc. (Phys. Rev. B, 2001,63,115207)

        Exp. ( Nat Commun 2014, 5 , 5221 ) and ( Solid State Commun. 2011, 141, 262 )"""
    
    Eps_II = e_inf_II*( 1 + (w_LO_II**2 - w_TO_II**2)/(w_TO_II**2 - w**2 - 1j*w*gama_II))
    
    Eps_L = e_inf_L*( 1 + (w_LO_L**2 - w_TO_L**2)/(w_TO_L**2 - w**2 - 1j*w*gama_L))
    
    return Eps_II,Eps_L

def dispersion_(ea,es,e_i,e_ii,d): 
    """Dispersion of hyperbolic phonon polaritons from Tunable Phonon Polaritons in 'Atomically Thin van der Waals' """
    psi = np.sqrt(e_ii)/(1j*np.sqrt(e_i))
    
    return -(psi/d)*(np.arctan(ea/(e_i*psi)) + np.arctan(es/(e_i*psi)))

"""Rod antenna geometry functions"""

def create_vert(L,w):
    """Create array of vertices for this specific antenna"""
    n_of_vertices = 4

    vert = np.zeros(n_of_vertices*2).reshape(n_of_vertices,2)

    vert[0,0] = -L/2
    vert[0,1] = -w/2

    vert[1,0] = L/2
    vert[1,1] = -w/2

    vert[2,0] = L/2
    vert[2,1] = w/2

    vert[3,0] = -L/2
    vert[3,1] = w/2
    
    return vert

def create_path(vert):
    """Create vector which link the vertices"""

    path = np.empty_like(vert)

    for i in range(len(vert)):
        if (i==len(vert)-1):
            path[i] = vert[0] - vert[i]
        else:
            path[i] = vert[i+1] - vert[i]
            
    return path

def create_normal(path):
    """Create normalized vectors normal to the path vector"""
    normal = np.empty_like(path)

    for i in range(len(path)):
        normal[i,0] = path [i,1]
        normal[i,1] = -path [i,0]
        normal[i] = normal[i]/np.linalg.norm(path[i])
    return normal

def perimiter(path):
    """From the path vector, calculates the perimeter of antenna"""
    d=0
    for i in range (len(path)):
        d = d + np.linalg.norm(path[i])
    return d

def find_nearest(array, value):
    """Finds the index of vector corresponding to a given value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_2D(array, value):
    """Finds the index of 2D vector corresponding to a given value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def path_resolution_(x,y):
    """Finds the lowest mesh size used in the simulation."""
    dx = np.zeros(len(x)-1)
    dy = np.zeros(len(y)-1)
    
    for i in range(len(x)-1):
        dx[i] = x[i+1]-x[i]
        
    for i in range(len(y)-1):
        dy[i] = y[i+1]-y[i]
        
    return np.min ( [np.min(dx),np.min(dy)])

def gen_dir(v,p,n,path_res,ang_res):
    """Generates list with coordinates, direction of analysis and vertices positions given the vertices,path,normal, path resolution and the angle resolution (of analysis direction).

It first generates an array with ranging from zero to the perimiter and it calculates the its vertices. Than, for each position of this array, this coordinate is appended. Furthermore, if it is an vertice, it will generate a list of directions in which direction is given by vectors rotated between both normals of the sides of these vertices. If it is not a vertice, than it will simply save the normal vector of this side."""
    coor = []
    anal_dir = []
    vert_pos = [] #generate f(x)
    
    weg = np.arange(0,perimiter(p),path_res) #put all perimiter in an array with the desired resolution
    vert_mask = find_v_in_path(p,weg) # find the position of the vertices
    j=0
    point = v[0]        # set initial position and direction
    direc = p[0]/np.linalg.norm(p[0])
    
    for i in range(len(weg)):
        coor.append(point) #creates coordinates
        
        if(vert_mask[i]==1):
            vert_pos.append(i)
            if(j+1<(len(v))):
                last = (n[j])
                now = (n[j+1])
                j = j+1
            else:
                last = (n[-1])
                now = (n[0])
                
            l_a = angle_(last)
            n_a = angle_(now)
            
            #this creates the angles in which the initial direction will be rotated
            
            if (n_a-l_a>=0): #check if angle is being calculated in certain quadrants
                theta = np.arange(0,n_a-l_a,ang_res)
            else: #c
                theta = np.arange(0,2*np.pi-l_a+n_a,ang_res)
            rot_dir = []
            for t in range(len(theta)):
                vector = rotation_(last,theta[t])
                rot_dir.append(vector)               

            anal_dir.append(rot_dir)
        else:
            anal_dir.append(n[j])
        
        
        direc =  p[j]/np.linalg.norm(p[j])
        point = point + path_res*direc #goes to next point
        
    return coor,anal_dir, vert_pos

def find_v_in_path(p,caminho):
    """Find the vertices in the perimeter array"""
    result = np.zeros(len(caminho))
    d = 0
    for i in range(len(p)):
        d = d + np.linalg.norm(p[i])
        j = find_nearest(caminho,d)
        result[j] = 1
    return result

def angle_(v):
    """Calculates the angle of the vector considering its quadrant ( 0 -> $2\pi$ )"""
    x = v[0]
    y = v[1]
    
    if (x==0):
        return np.sign(y)*np.pi/2
    if (x>=0 and y >= 0):
        return np.arctan(y/x)
    if (x<0 and y>=0):
        return np.arctan(y/x) + np.pi
    if (x<0 and y<0):
        return np.arctan(y/x) + np.pi
    if (x>=0 and y<0):
        return np.arctan(y/x) + 2*np.pi
    
def rotation_(v,phi):
    """Rotates vector by $\phi$"""
    r = np.zeros(4).reshape(2,2)
    r[0,0] = np.cos(phi)
    r[0,1] = - np.sin(phi)
    r[1,0] = np.sin(phi)
    r[1,1] = np.cos(phi)
    
    return r @ v

def extract_profile_v2(x,y,E,initial,dir_,path_res):
    """Extract profile from arbitrary direction (linear interpolation is used). Check if the direction is parallel. If it is, then its easy.
If its not, than pick ups profiles along a desired path."""

    
    # find closest initial point to desired
    
    ini_x = find_nearest(array=x,value=initial[0])
    ini_y = find_nearest(array=y,value=initial[1])
    
    # Take profiles for directions parallel to the grid
    
    # Horizontal direction
    
    if (dir_[1]==0):
        if(dir_[0]<0):
            profile_path_dist = (abs(x[:ini_x])-abs(x[ini_x]))
            idx = np.argsort(profile_path_dist)
            profile_path_dist = profile_path_dist[idx]
            profile = E[:ini_x,ini_y][idx]
            return profile_path_dist , profile
        if(dir_[0]>0):
            profile_path_dist = abs(x[ini_x:])-abs(x[ini_x])
            idx = np.argsort(profile_path_dist)
            profile = E[ini_x:,ini_y][idx]
            return profile_path_dist , profile
    
    # Vertical direction

    if (dir_[0]==0):
        if(dir_[1]<0):
            profile_path_dist = abs(y[:ini_y])-abs(y[ini_y])
            idx = np.argsort(profile_path_dist)
            profile_path_dist = profile_path_dist[idx]
            profile = E[ini_x,:ini_y][idx]
            return profile_path_dist , profile
        if(dir_[1]>0):
            profile_path_dist = abs(y[ini_y:])-abs(y[ini_y])
            idx = np.argsort(profile_path_dist)
            profile_path_dist = profile_path_dist[idx]
            profile = E[ini_x,ini_y:][idx]
            return profile_path_dist , profile
        
    # In the case that direction is not parallel to the grid
    signal_map = np.zeros(E.size).reshape(E.shape)
    
    profile_path_dist = []
    profile = []
    
    loc = [x[ini_x],y[ini_y]]
    dist = 0
    i = ini_x
    j = ini_y
    
    while( np.min(x) < loc[0] < np.max(x) and np.min(y) < loc[1] < np.max(y)):
        # Save this position and field
        profile_path_dist.append(dist)
        profile.append(E[i,j])
        
        #update signal map and indexes
        signal_map[i,j] = 1
        i_old = i
        j_old = j

        # Prepare next step
        while(signal_map[i,j] == 1 and ( np.min(x) < loc[0] < np.max(x) and np.min(y) < loc[1] < np.max(y))):
            loc = loc + dir_*path_res
            i = find_nearest(array=x,value=loc[0])
            j = find_nearest(array=y,value=loc[1])
            
        # Calculates distance of next step
        dist = np.hypot(x[i]-x[ini_x],y[j]-y[ini_y])
        
    profile_path_dist = np.array(profile_path_dist)
    profile = np.array(profile)
    
    return profile_path_dist , profile

def fit_(x,A,a,k,phi,B,k_2,phi_2,f):
    """Fitting function for the extracted Ez profile"""
    return A*np.exp(-a*(x))*np.sin(k*(x)+phi)/x**f + B*np.sin(k_2*x+phi_2)/np.sqrt(x)

def params_border(x,y,coordinates,directions, vert_position,path_res,Ez_r,beg_value,q,kapa):
    
    # Making new coordinates sweep (downsampling)
    coord_sweep = np.arange(0,len(coordinates),2)
    only_border = np.array([i for i in coord_sweep if i not in vert_position])
    coord_sweep_new = np.sort(np.concatenate((only_border,np.array(vert_position)), axis=None))
    coord_sweep_new = np.concatenate((coord_sweep_new, [(coord_sweep_new[0])]), axis = None)
    
    # Parameters
    
    param = []
    coor_an = []
    dir_an = []

    signal = np.zeros(x.size*y.size).reshape(x.size,y.size) # let the code skip checked coordinates from sides

    for i in coord_sweep_new:
        if i in only_border:
            x0 =coordinates[i][0]
            y0 = coordinates[i][1]
            ini = [find_nearest(x,x0),find_nearest(y,y0)]

            if (x0<=np.max(x) and y0<=np.max(y)): #Bounds to values only inside analysis region
                if(signal[ini[0],ini[1]]==0):
                    signal[ini[0],ini[1]] = 1
                    dir_ = directions[i]
                    a,b = extract_profile_v2(x,y,Ez_r,[x0,y0],dir_,path_res)
                    beg = find_nearest(array=a,value=beg_value)


                    try:
                        bounds_1 = ((-np.inf,kapa-0.001,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0), (np.inf, kapa+0.001 ,np.inf,np.inf,np.inf,np.inf,np.inf,0.5))
                        popt,pcov = curve_fit(fit_,a[beg:],b[beg:],p0=[b[0],kapa,q,0,0.1,1,0,0.25],bounds = bounds_1)
                        if(abs(popt[0])>0.5):
                            param.append(popt)
                            coor_an.append([x0,y0])
                            dir_an.append(dir_)
                    except:
                        pass

        if i in vert_position:
            x0 =coordinates[i][0]
            y0 = coordinates[i][1]
            ini = [find_nearest(x,x0),find_nearest(y,y0)]
            for j in range(0,len(directions[i]),2):
                dir_ = directions[i][j]
                a,b = extract_profile_v2(x,y,Ez_r,[x0,y0],dir_,path_res)
                beg = find_nearest(array=a,value=beg_value)

                try:
                    bounds_1 = ((-np.inf,kapa-0.001,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0), (np.inf, kapa+0.001 ,np.inf,np.inf,np.inf,np.inf,np.inf,0.5))
                    popt,pcov = curve_fit(fit_,a[beg:],b[beg:],p0=[b[0],kapa,q,0,0.1,1,0,0.25],bounds = bounds_1)
                    if(abs(popt[0])>0.5):
                        param.append(popt)
                        coor_an.append([x0,y0])
                        dir_an.append(dir_)


                except:
                    pass
      
    # Filtering outliers using Z-function
    z_amp = np.abs(stats.zscore(abs(np.array(param)[:,0])))
    idx_amp = np.where(z_amp>3)[0]
    z_dec = np.abs(stats.zscore(abs(np.array(param)[:,1])))
    idx_dec = np.where(z_dec>3)[0]
    
    idx = np.union1d(idx_amp,idx_dec)

    for i in sorted(idx, reverse=True):
        del param[i]
        del coor_an[i]
        del dir_an[i]
                
    return param,coor_an,dir_an

def params_border_circular(x,y,coordinates,directions,path_res,Ez_r,beg_value,q,kapa):
    
    # Making new coordinates sweep (downsampling)
    coord_sweep = np.arange(len(coordinates))
    coord_sweep = np.concatenate((coord_sweep, [(coord_sweep[0])]), axis = None)
    
    # Parameters
    
    param = []
    coor_an = []
    dir_an = []

    signal = np.zeros(x.size*y.size).reshape(x.size,y.size) # let the code skip checked coordinates from sides

    for i in coord_sweep:
        x0 =coordinates[i][0]
        y0 = coordinates[i][1]
        ini = [find_nearest(x,x0),find_nearest(y,y0)]

        if (x0<=np.max(x) and y0<=np.max(y)): #Bounds to values only inside analysis region
            if(signal[ini[0],ini[1]]==0):
                signal[ini[0],ini[1]] = 1
                dir_ = directions[i]
                a,b = extract_profile_v2(x,y,Ez_r,[x0,y0],dir_,path_res)
                beg = find_nearest(array=a,value=beg_value)

                try:
                    bounds_1 = ((-np.inf,kapa-0.001,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0), (np.inf, kapa+0.001 ,np.inf,np.inf,np.inf,np.inf,np.inf,0.5))
                    popt,pcov = curve_fit(fit_,a[beg:],b[beg:],p0=[b[0],kapa,q,0,0.1,1,0,0.25],bounds = bounds_1)
                    if(abs(popt[0])>0.1):
                        param.append(popt)
                        coor_an.append([x0,y0])
                        dir_an.append(dir_)
                except:
                    pass
    # Filtering outliers using Z-function
    z_amp = np.abs(stats.zscore(abs(np.array(param)[:,0])))
    idx_amp = np.where(z_amp>3)[0]
    z_dec = np.abs(stats.zscore(abs(np.array(param)[:,1])))
    idx_dec = np.where(z_dec>3)[0]
    
    idx = np.union1d(idx_amp,idx_dec)

    for i in sorted(idx, reverse=True):
        del param[i]
        del coor_an[i]
        del dir_an[i]
                
    return param,coor_an,dir_an

def prop_length(coor_an,param,detect_factor):
    # Use detect factor = 0.5 for now
    
    prop_l = []
    for i in range(len(coor_an)):
        if (abs(param[i][0])<=0.5 or abs(param[i][1])<=0.05):
            prop_l.append(0)
        else:
            propagation = minimize(lambda x: prop_length_equation(x, abs(param[i][0]),abs(param[i][1]),param[i][-1],detect_factor), x0=0.001)
            #propagation = -np.log(detect_factor/abs(param[i][0]))/abs(param[i][1])
            #print(propagation.x[0])
            if (propagation.x[0]>=0):
                prop_l.append(propagation.x[0])
            else:
                prop_l.append(0)
                
    return prop_l

def prop_length_v2(coor_an,param,detect_factor):
    # Use detect factor = 0.5 for now
    
    prop_l = []
    for i in range(len(coor_an)):

        propagation = minimize(lambda x: prop_length_equation(x, abs(param[i][0]),abs(param[i][1]),param[i][-1],detect_factor), x0=0.0001)

        prop_l.append(propagation.x[0])

                
    return prop_l

def prop_length_equation(x,A,dec,f,detect_factor):
    return (detect_factor - A*np.exp(-x*dec)/x**f)**2

def plot_func_1(coordinates,coor_an,dir_an,prop_l,param,f,f_i,L,p):
    fig = plt.figure(dpi=150)
    ax =  fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    #plt.title("Dec., " + "$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")


    # Antenna Geometry
    plot_geo = plt.plot(np.array(coordinates)[:,0],np.array(coordinates)[:,1],color="black")

    # Propagation Length

    arrow_size = np.array(prop_l)
    plot_arrows = plt.quiver(np.array(coor_an)[:,0],np.array(coor_an)[:,1],arrow_size*np.array(dir_an)[:,0],arrow_size*np.array(dir_an)[:,1],color="gray",alpha=0.5,scale=1, scale_units='xy')
    plot_ref = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1]+ arrow_size*np.array(dir_an)[:,1], marker=".", color="white",alpha=0)


     # Checking if arrow will be too small    
    if(np.mean(prop_l[:])<1):
        factor = 1
    else:
        factor = np.mean(prop_l[:])

     #Decay    
    arrow_size = factor*np.array(param)[:,1]/(max(np.array(param)[:,1]))
    cor = np.array(param)[:,1]
    sc_dec = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Blues", marker=".",label="decay",s=15)
    sc_plot = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    #Amp

    arrow_size = factor*abs(np.array(param)[:,0])/(2*max(np.array(param)[:,0]))
    cor = abs(np.array(param)[:,0])
    sc_amp = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Reds", marker=".",label="amplitude")
    plot_amp = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    ax.set_xlabel("$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")



    divider = make_axes_locatable(ax)
    cax_dec = divider.append_axes("right", size="5%", pad=0.05)
    clb_dec = plt.colorbar(sc_dec, cax = cax_dec)


    cax_amp = divider.append_axes("top", size="5%",pad=0.05)
    clb_amp = plt.colorbar(sc_amp, cax = cax_amp, orientation = "horizontal")
    cax_amp.xaxis.set_ticks_position("top")
    
def plot_func2(coordinates,coor_an,dir_an,prop_l,param,f,f_i,L,p):
    fig = plt.figure(dpi=150)
    ax =  fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    #plt.title("Dec., " + "$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")


    # Antenna Geometry
    plot_geo = plt.plot(np.array(coordinates)[:,0],np.array(coordinates)[:,1],color="black")

    # Propagation Length

    arrow_size = np.array(prop_l)
    plot_arrows = plt.quiver(np.array(coor_an)[:,0],np.array(coor_an)[:,1],arrow_size*np.array(dir_an)[:,0],arrow_size*np.array(dir_an)[:,1],color="gray",alpha=0.5,scale=1, scale_units='xy')
    plot_ref = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1]+ arrow_size*np.array(dir_an)[:,1], marker=".", color="white",alpha=0)


     # Checking if arrow will be too small    
    if(np.mean(prop_l[:])<1):
        factor = 1
    else:
        factor = np.mean(prop_l[:])

     #f factor (geometrical decay)   
    arrow_size = factor*np.array(param)[:,-1]/(max(np.array(param)[:,-1]))
    cor = np.array(param)[:,-1]
    sc_dec = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Greens", marker=".",label="f",s=15)
    sc_plot = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    #Decay

    arrow_size = factor*abs(np.array(param)[:,1])/(2*max(np.array(param)[:,1]))
    cor = abs(np.array(param)[:,1])
    sc_amp = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Blues", marker=".",label="amplitude")
    plot_amp = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    ax.set_xlabel("$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")



    divider = make_axes_locatable(ax)
    cax_dec = divider.append_axes("right", size="5%", pad=0.05)
    clb_dec = plt.colorbar(sc_dec, cax = cax_dec)


    cax_amp = divider.append_axes("top", size="5%",pad=0.05)
    clb_amp = plt.colorbar(sc_amp, cax = cax_amp, orientation = "horizontal")
    cax_amp.xaxis.set_ticks_position("top")

def plot_func3(coordinates,coor_an,dir_an,prop_l,param,f,f_i,L,p):
    fig = plt.figure(dpi=150)
    ax =  fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    

    #plt.title("Dec., " + "$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")


    # Antenna Geometry
    plot_geo = plt.plot(np.array(coordinates)[:,0],np.array(coordinates)[:,1],color="black")

    # Propagation Length

    arrow_size = np.array(prop_l)
    plot_arrows = plt.quiver(np.array(coor_an)[:,0],np.array(coor_an)[:,1],arrow_size*np.array(dir_an)[:,0],arrow_size*np.array(dir_an)[:,1],color="gray",alpha=0.5,scale=1, scale_units='xy')
    plot_ref = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1]+ arrow_size*np.array(dir_an)[:,1], marker=".", color="white",alpha=0)


     # Checking if arrow will be too small    
    if(np.mean(prop_l[:])<1):
        factor = 1
    else:
        factor = np.mean(prop_l[:])

     #f geometric decay factor 
    
    arrow_size = factor*np.array(param)[:,-1]/(max(np.array(param)[:,-1]))
    cor = np.array(param)[:,-1]
    sc_dec = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Greens", marker=".",label="decay",s=15,vmin=0,vmax=0.5)
    sc_plot = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    #Amp
    
    amp_suave = np.convolve(abs(np.array(param)[:,0]), np.ones((3,))/3, mode='same')

    arrow_size = factor*abs(amp_suave)/(2*max(amp_suave))
    cor = amp_suave
    sc_amp = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Reds", marker=".",label="amplitude")
    plot_amp = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    ax.set_xlabel("$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")



    divider = make_axes_locatable(ax)
    cax_dec = divider.append_axes("right", size="5%", pad=0.05)
    clb_dec = plt.colorbar(sc_dec, cax = cax_dec)


    cax_amp = divider.append_axes("top", size="5%",pad=0.05)
    clb_amp = plt.colorbar(sc_amp, cax = cax_amp, orientation = "horizontal")
    cax_amp.xaxis.set_ticks_position("top")
    
def plot_func5(coordinates,coor_an,dir_an,prop_l,param,f,f_i,L,p,Avmin,Avmax):
    fig = plt.figure(dpi=150)
    ax =  fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    

    #plt.title("Dec., " + "$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")


    # Antenna Geometry
    plot_geo = plt.plot(np.array(coordinates)[:,0],np.array(coordinates)[:,1],color="black",linewidth=1,alpha=0.5)

    # Propagation Length

    arrow_size = np.array(prop_l)
    plot_arrows = plt.quiver(np.array(coor_an)[:,0],np.array(coor_an)[:,1],arrow_size*np.array(dir_an)[:,0],arrow_size*np.array(dir_an)[:,1],color="gray",alpha=0.5,scale=1, scale_units='xy')
    plot_ref = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1]+ arrow_size*np.array(dir_an)[:,1], marker=".", color="white",alpha=0)


     # Checking if arrow will be too small    
    if(np.mean(prop_l[:])<1):
        factor = 1
    else:
        factor = np.mean(prop_l[:])

     #f geometric decay factor 
    f_suave = np.convolve(np.array(param)[:,-1], np.ones((5,))/5, mode='same')
    
    arrow_size = 0
    cor = f_suave
    sc_dec = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="seismic", marker=".",label="decay",s=15,vmin=0,vmax=0.5)
    #sc_plot = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    #Amp
    
    arrow_size = factor*abs(np.array(param)[:,0])/(2*max(np.array(param)[:,0]))
    cor = abs(np.array(param)[:,0])
    sc_amp = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="afmhot_r", marker=".",label="amplitude",vmin = Avmin, vmax = Avmax)
    #plot_amp = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    ax.set_xlabel("$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")



    divider = make_axes_locatable(ax)
    cax_dec = divider.append_axes("right", size="5%", pad=0.05)
    clb_dec = plt.colorbar(sc_dec, cax = cax_dec)


    cax_amp = divider.append_axes("top", size="5%",pad=0.05)
    clb_amp = plt.colorbar(sc_amp, cax = cax_amp, orientation = "horizontal")
    cax_amp.xaxis.set_ticks_position("top")
    
def plot_func4(coordinates,coor_an,dir_an,prop_l,param,f,f_i,L,p):
    fig = plt.figure(dpi=150)
    ax =  fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    #plt.title("Dec., " + "$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")


    # Antenna Geometry
    plot_geo = plt.plot(np.array(coordinates)[:,0],np.array(coordinates)[:,1],color="black")

    # Propagation Length

    arrow_size = np.array(prop_l)
    plot_arrows = plt.quiver(np.array(coor_an)[:,0],np.array(coor_an)[:,1],arrow_size*np.array(dir_an)[:,0],arrow_size*np.array(dir_an)[:,1],color="gray",alpha=0.5,scale=1, scale_units='xy')
    plot_ref = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1]+ arrow_size*np.array(dir_an)[:,1], marker=".", color="white",alpha=0)


     # Checking if arrow will be too small    
    if(np.mean(prop_l[:])<1):
        factor = 1
    else:
        factor = np.mean(prop_l[:])

     #f geometric decay factor
    
    arrow_size = factor/20
    cor = np.array(param)[:,-1]
    sc_dec = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Greens", marker=".",label="decay",s=15)
    #sc_plot = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    #Amp

    arrow_size = 2*factor/20
    cor = abs(np.array(param)[:,0])
    sc_amp = plt.scatter(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],c = cor, cmap="Reds", marker=".",label="amplitude")
    #plot_amp = plt.plot(np.array(coor_an)[:,0] + arrow_size*np.array(dir_an)[:,0] ,np.array(coor_an)[:,1] + arrow_size*np.array(dir_an)[:,1],"--",color="black",alpha=0.5,linewidth=0.5)

    ax.set_xlabel("$\omega$ = "+ str(round(f[f_i])) + " cm$^{-1}$ , L = " + str(round(L[p],2)) + " $\mu m$")



    divider = make_axes_locatable(ax)
    cax_dec = divider.append_axes("right", size="5%", pad=0.05)
    clb_dec = plt.colorbar(sc_dec, cax = cax_dec)


    cax_amp = divider.append_axes("top", size="5%",pad=0.05)
    clb_amp = plt.colorbar(sc_amp, cax = cax_amp, orientation = "horizontal")
    cax_amp.xaxis.set_ticks_position("top")
    
def suav_mov_avarge(param, coor_an , dir_an,N,mode_):
    new_param = []
    new_coor_an = []
    new_dir_an = []
    
    for i in range(np.array(param).shape[1]):
        if (i==0):
            new_param.append(np.convolve(abs(np.array(param)[:,i]), np.ones((N,))/N, mode=mode_))
        else:
            new_param.append(np.convolve((np.array(param)[:,i]), np.ones((N,))/N, mode=mode_))
   
    new_coor_an.append(np.convolve((np.array(coor_an)[:,0]), np.ones((N,))/N, mode=mode_))
    new_coor_an.append(np.convolve((np.array(coor_an)[:,1]), np.ones((N,))/N, mode=mode_))

    new_dir_an.append(np.convolve((np.array(dir_an)[:,0]), np.ones((N,))/N, mode=mode_))
    new_dir_an.append(np.convolve((np.array(dir_an)[:,1]), np.ones((N,))/N, mode=mode_))
    
    return np.array(new_param).T,np.array(new_coor_an).T,np.array(new_dir_an).T

def find_nearest_2D(array, value):
    array = np.array(array)
    dist_ = (array-value)[:,0]**2 + (array-value)[:,1]**2
    idx = dist_.argmin()
    return idx

def fresnel_N(L,lamb,dist):
    return L**2/(lamb*dist)