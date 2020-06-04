#!/home/hcho/venv/bin/python
# coding: utf-8
# Radial profiles
# downloaded from
# https://bitbucket.org/ngoldbaum/galaxy_analysis/src/default/annularAnalysis.py
# 06/01/2020
# Hyerin Cho

from yt.mods import *
import matplotlib.pyplot as plt
from sfrFromParticles import *

   
G=6.67384e-8
gPerMsun = 1.9891e33
cmperpc = 3.08567758e18
cmperkpc = cmperpc*1000
speryear = 3.16e7

# a decorator from yt
@derived_field(name='logDensity',take_log=False)
def logDensity(field,data):
    return np.log10(data['Density'])


def fH2KMT(col,Z):
    ''' Produce the Krumholz McKee and Tumlinson (2009) estimate for the molecular hydrogen fraction given
        col, the column density in solar masses per square parsec
        Z, the absolute metallicity.
        '''
    ZSol = 0.02
    Z0 = Z/ZSol
    clumping = 5.0
    Sig0 = col * gPerMsun / cmperpc**2
    ch = 3.1 * (1.0 + 3.1*np.power(Z0,0.365))/4.1
    tauc = 320.0 * clumping * Sig0 * Z0
    ss = np.log(1.0 + 0.6 * ch + .01*ch*ch)/(0.6*tauc)
    val = 1.0 - 0.75 * ss/(1.0+0.25*ss)
    val = np.clip(val, 0.001, 1.0)
    return val

def axisSwitch(axIn):
    ''' A convenience function which does a little bit of logic at the beginning of every plotting function below.
	    thisgrid->BaryonField[DensNum][index] = HaloDensity;
    If a matplotlib axis is provided to the function, then we assume that the figure is being set up elsewhere, and
    the function only needs to draw a plot on the provided axes. 
    If an axis is not provided, create one.
    In both cases, return fig,ax. If we didn't have to create a figure, fig is just a string.'''
    if axIn is None:
        fig,ax = plt.subplots(figsize=(8,8))
    else:
        ax = axIn
        fig = "No figure specified inside the plotting function"
    return fig,ax
    
def returnSwitch(fig,ax,basename,filetype,pfName):
    ''' This provides the return value for the plotting functions below. It's always the filename, but if no
    filetype is provided or if the filetype is explicitly 'screen' as in an iPython notebook, we don't save the plot anywhere.
    However if filetype is something else (hopefully something reasonable like png or pdf), save the figure. '''
    filename = pfName+"_"+basename+'.'+filetype
    if filetype=='screen' or filetype is None or fig=="No figure specified inside the plotting function":
        return filename
    plt.savefig(filename)
    return filename
    
    
class annularAnalysis:
    ''' Analyze a sequence of annuli to compute various radial profiles. 
        WARNING: This analysis class assumes the cylindrical z-axis of your annuli is 
        aligned with the z-axis of the simulation box. It's presumably not very hard to generalize this.
        ADDITIONAL WARNING: The analysis also assumes that newly-formed particles (i.e. particles which were formed
          over the course of the simulation) are of particle type 4. It may be better to use creation_time>0.
        '''

    def __init__(self, inner, outer, Nsteps, height, center, fluxHeights):
        '''
        Arguments:
        inner - the innermost radius in kpc - I usually set this to be a little larger than the finest resolution
        outer - the outermost radius in kpc
        Nsteps - number of annuli to use
        height - the distance above and below the midplane to which our cylinders will extend
        center - the center of every annulus. Note that the code assumes the angular momentum is aligned with +z.
        fluxHeights - a list of heights above or below the disk in kpc at which to compute the mass flux. This is a particularly
                        time-consuming part of the code, so if you don't care about mass loading factors, pass in a short array.'''
        
        #  Save the inputs for later use
        self.inner = inner
        self.outer = outer
        self.Nsteps = Nsteps
        self.height = height
        self.center = center
        self.fluxHeights = fluxHeights
        
        # Set up arrays to store the quantities we're going to compute.
        self.r=np.zeros(Nsteps) # radii of boundaries between annuli (kpc)
        self.spMassEnclosed=np.zeros(Nsteps) # mass enclosed in a sphere of radius r[i] (Msun)
        self.cylMassEnclosed=np.zeros(Nsteps) # mass enclosed in a cylinder of radius r[i] and half-height self.height (Msun)
        self.spVcirc=np.zeros(Nsteps) # circular velocity calculated from spherical mass enclosed (cm/s)
        self.annMass=np.zeros(Nsteps) # mass in an annulus (Msun)
        self.annArea=np.zeros(Nsteps) # area of an annulus (kpc^2)
        self.col=np.zeros(Nsteps) # column density in an annulus (Msun/pc^2)
        self.colGas=np.zeros(Nsteps) # column density of gas in an annulus (Msun/pc^2)
        self.colSt=np.zeros(Nsteps) # column density of stars in an annulus (Msun/pc^2)
        self.colDM=np.zeros(Nsteps) # column density in dark matter in an annulus (Msun/pc^2)
        self.sigR=np.zeros(Nsteps) # non-thermal velocity dispersion of gas along cylindrical rHat in a given radius (cm/s)
        self.sigPhi=np.zeros(Nsteps) # same but along cylindrical thetaHat (cm/s)
        self.sigZ=np.zeros(Nsteps) # same but along zHat (cm/s)
        self.sigStR=np.zeros(Nsteps) # The next three are the same as those above, but for star particles.
        self.sigStPhi=np.zeros(Nsteps)
        self.sigStZ=np.zeros(Nsteps)
        self.vR=np.zeros(Nsteps) # The average velocity of gas in cylindrical rHat in the annulus (cm/s)
        self.vPhi=np.zeros(Nsteps) # Same but along cylindrical thetaHat (cm/s)
        self.vZ=np.zeros(Nsteps) # Same but along zHat (cm/s)
        self.vStR=np.zeros(Nsteps) # The next three are the same, except using star particles.
        self.vStPhi=np.zeros(Nsteps)
        self.vStZ=np.zeros(Nsteps)
        self.beta=np.zeros(Nsteps) # an estimate of the powerlaw slope of the rotation curve dln v_phi / dln r (dimensionless)
        self.QWS=np.zeros(Nsteps) # the Wang-Silk estimate of 2-component axisymmetric stability (dimensionless)
        self.QRW=np.zeros(Nsteps) # the Romeo-Wiegert estimate of the same (dimensionless)
        self.Qgas=np.zeros(Nsteps) # Q for the gas only (dimensionless)
        self.Qst=np.zeros(Nsteps) # Q for the stars only (dimensionless)
        self.hGas=np.zeros(Nsteps) # The mass-weighted standard deviation of gas height. Taken to be an estimate of the scale height (kpc)
        self.hStars=np.zeros(Nsteps) # same for the stars
        self.Nstars=np.zeros(Nsteps) # Number of new star particles
        self.Ngas = np.zeros(Nsteps) # Number of resolution elements
        self.spMassDM=np.zeros(Nsteps) # Mass contained in DM in sphere of radius r[i] (Msun)
        self.spMassSt=np.zeros(Nsteps) # same for star particles
        self.spMassGas=np.zeros(Nsteps) # same for gas
        self.sigTh=np.zeros(Nsteps) # thermal velocity dispersion (cm/s)
        self.tOrb = np.zeros(Nsteps) # orbital time (yrs)
        self.Z = np.zeros(Nsteps) # metallicity of gas
        self.widthZ=np.zeros(Nsteps) # standard deviation of gas metallicity
        self.stZ = np.zeros(Nsteps) # stellar metallicity
        self.stWidthZ = np.zeros(Nsteps) # standard deviation of stellar metallicity
        self.stAge = np.zeros(Nsteps) # stellar age (years)
        self.stAgeSpread = np.zeros(Nsteps) # standard deviation of stellar age (years)
        self.tff = np.zeros(Nsteps) # freefall time -- from mean density in the annular volume(s)
        self.tff2 = np.zeros(Nsteps) # freefall time -- from the mass-weighted mean of the log in the annular volume (s)
        self.tff3 = np.zeros(Nsteps) # freefall time -- from the scale height crossing time (s)
        self.sfrs = np.zeros(Nsteps) # star formation rate (Msun/yr)
        self.colSFRs = np.zeros(Nsteps) # star formation column density (Msun/yr/pc^2)
        self.avgLogDensity = np.zeros(Nsteps) # mass-weighted average of the mean of the log of the density (log_10 of g/cm^3)
        self.logDensityWidth = np.zeros(Nsteps) # standard deviation of the above (dex)
        self.avgRadius = np.zeros(Nsteps) # mass-weighted average radius of the bin.
        self.cylUpFluxes = np.zeros((Nsteps,10)) # upward mass flux through the annulus
        self.cylDownFluxes = np.zeros((Nsteps,10)) # downward mass flux through the annulus
        
    def run(self, pf, temperatureCut=False):
        '''
        Actually run the analysis. 
        pf is the parameter file. This is an argument here rather than in the initializer because 
            we don't want to store the parameter file and keep the data in memory longer than necessary. 
        temperatureCut is an optional flag. When true it asks that the annuli be constructed only from cells below
            10^5 K. This is a vestige of how I set up this function originally, and is probably a bad idea.
        '''
        # Initialize a couple lists we'll need in the computation, but which we don't want to keep around after the computation.
        # These will store yt data objects.
        annuli=[0]*self.Nsteps
        coldAnnuli=[0]*self.Nsteps
        disks=[0]*self.Nsteps
        tallDisks=[0]*self.Nsteps
        tallAnnuli=[0]*self.Nsteps
        
        # Store the pf name so we can use it when saving plots to files later.
        self.pfName = repr(pf)
        
        # Now let's start the main loop over annuli.
        for i in range(self.Nsteps):
            print("Starting analysis for disk i=",i) # tell the user what we're up to
            
            self.r[i] = self.inner + float(i)*self.outer/float(self.Nsteps-1)
            
            # Find particles in this spherical shell. 
            sphere=pf.h.sphere(self.center, (self.r[i],'kpc'))
            dmp = sphere["particle_type"]==1 # Dark matter
            stp = np.logical_or(sphere["particle_type"]==2, sphere["particle_type"]==4) # Stars

            # Add up mass
            self.spMassEnclosed[i] = sphere.quantities["TotalMass"]()
            self.spMassDM[i] = np.sum(sphere["ParticleMassMsun"][dmp])
            self.spMassSt[i] = np.sum(sphere["ParticleMassMsun"][stp])
            self.spMassGas[i] = np.sum(sphere["CellMassMsun"])
            
            # A couple quantities derived from the spherical mass enclosed:
            self.spVcirc[i] = np.sqrt(G*self.spMassEnclosed[i]*gPerMsun/(self.r[i]*cmperkpc)) # v^2/r = GM/r^2 -- cm/s
            self.tOrb[i] = self.r[i]*cmperkpc/(speryear*self.spVcirc[i]) # years
            
            # Set up a series of concentric disks. disks is used for most of the analysis, but we need
            # tall disks, i.e. ones with larger heights, to measure the mass flux at various heights.
            disks[i] = pf.h.disk(self.center,[0,0,1],self.r[i]/pf['kpc'],self.height/pf['kpc'])
            tallDisks[i] = pf.h.disk(self.center,[0,0,1],self.r[i]/pf['kpc'],max(self.fluxHeights)*1.1/pf['kpc'])
            self.cylMassEnclosed[i] = disks[i].quantities["TotalMass"]()

#            coldDisks[i] = disk.cut_region( ["grid['Temperature'] < 10.0**5"] )
            if(i != 0):
                # Identify the annulus as this cold disk minus the previous cold disk
                annuli[i] = pf.h.boolean([disks[i], "NOT", disks[i-1]])
                tallAnnuli[i] = pf.h.boolean([tallDisks[i], "NOT", tallDisks[i-1]])
                # Identify particles in this annulus.
            else:
                # When i=0, we're dealing with a disk, not an annulus.
                annuli[i] = disks[i] # The center "annulus" is just a disk.
                tallAnnuli[i] = tallDisks[i]
                
            # The particles shouldn't care about this cut, since all the analysis for particles is done using annuli
            #    rather than coldAnnuli. This will, however, affect things like the measured velocity dispersion.
            if temperatureCut:
                coldAnnuli[i] = annuli[i].cut_region( ["grid['Temperature'] < 10.0**5"] )
            else:
                coldAnnuli[i] = annuli[i]
                
            # Compute the flux through the tops and bottoms of annuli of various heights.
            conv = 6.30321217e25 # conversion factor for g/s -> Msun/yr
            for j, h in enumerate(self.fluxHeights):
                self.cylUpFluxes[i,j]=( tallAnnuli[i].calculate_isocontour_flux('z', self.center[2]+h/pf['kpc'], 
                                                'x-velocity', 'y-velocity', 'z-velocity', 'Density')*pf['cm']**2.0/conv )
                self.cylDownFluxes[i,j]= ( -1.0 * tallAnnuli[i].calculate_isocontour_flux('z', self.center[2]-h/pf['kpc'],
                                                       'x-velocity', 'y-velocity', 'z-velocity', 'Density')*pf['cm']**2.0/conv )
               
            # Get the particle positions in kpc
            particleX = (annuli[i]["particle_position_x"] - self.center[0])*pf['kpc']
            particleY = (annuli[i]["particle_position_y"] - self.center[1])*pf['kpc']
            particleR = np.sqrt(np.power(particleX,2.0) + np.power(particleY,2.0))
            particleZ = (annuli[i]["particle_position_z"] - self.center[2])*pf['kpc']
                
            annularVolume = np.sum(coldAnnuli[i]['CellVolume'])/cmperkpc**3  # cubic kpc
            supersetAnnularVolume = np.sum(annuli[i]['CellVolume'])/cmperkpc**3
            assert annularVolume / supersetAnnularVolume <= 1.0
            # Check the ratio of the volume in coldAnnuli compared to annuli. This should be 1 without temperatureCut
            # and less than 1 with it.
            print("Ratio of cold annulus volume to annulus volume: ",\
                  annularVolume/supersetAnnularVolume)

            # I used to compute the area analytically, but I think it's better to take the measured volume and divide by the
            # prescribed height.
            #if(i != 0):
            #    self.annArea[i] = np.pi * (self.r[i]**2.0 - self.r[i-1]**2.0) # sq kpc
            #else:
            #    self.annArea[i] = np.pi * self.r[i]**2.0 # square kpc
            
            self.annArea[i] = annularVolume/(2.0*self.height) # square kpc

            # Total gas mass. I would have naively expect this to include particle mass, but I /think/ that since
            # particles have no information on 'Temperature', the coldAnnuli refer only to the gas. Any assignment
            # of particles to these annuli has to be done by hand, as with the logic on the 'particles' array above.
            self.annMass[i] = coldAnnuli[i].quantities["TotalMass"]() # solar masses
            self.col[i] = self.annMass[i]/(self.annArea[i]*1.0e6) # Msun/pc^2 
            self.colGas[i] = coldAnnuli[i].quantities["TotalQuantity"]("CellMassMsun") / (self.annArea[i]*1.0e6)
            
            # Split up the particles which are in this annulus into stars and DM. 
            star_particles = np.logical_or( annuli[i]["particle_type"]==2, annuli[i]["particle_type"]==4 )
            new_star_particles = annuli[i]["particle_type"]==4 
            dm_particles = annuli[i]["particle_type"]==1

            self.Nstars[i]=np.sum(star_particles)
            self.Ngas[i]=len(annuli[i]["x-velocity"])
            self.colSt[i] = np.sum(annuli[i]["ParticleMassMsun"][star_particles]) / (self.annArea[i]*1.0e6)
            self.colDM[i] = np.sum(annuli[i]["ParticleMassMsun"][dm_particles]) / (self.annArea[i]*1.0e6)
            
            # Manually compute the radial and tangential components of the velocity for the star particles.
            stX = particleX[star_particles]
            stY = particleY[star_particles]
            stVx = annuli[i]["particle_velocity_x"][star_particles]
            stVy = annuli[i]["particle_velocity_y"][star_particles]
            stTh = np.arctan2(stY,stX) # angular position relative to the x-axis
            stCosTh = np.cos(stTh)
            stSinTh = np.sin(stTh)
            # Radial velocity
            stVr = stVx*stCosTh + stVy*stSinTh
            # Tangentail velocity
            stVt = -stVx*stSinTh + stVy*stCosTh            
            # Vertical velocity
            stVz = annuli[i]["particle_velocity_z"][star_particles]
            # Mass for star particles and for newly-formed star particles. These arrays are used when computing 
            # various moments below.
            stM =  annuli[i]["ParticleMassMsun"][star_particles]
            newStM = annuli[i]["ParticleMassMsun"][new_star_particles]
            
            # Compute moments of the velocity, positions, and metals for stars
            stTotalMass = np.sum(stM)
            newStTotalMass = np.sum(newStM)
            # Bulk velocity:
            self.vStR[i] = np.sum(stM*stVr)/stTotalMass
            self.vStPhi[i] = np.sum(stM*stVt)/stTotalMass
            self.vStZ[i] = np.sum(stM*stVz)/stTotalMass
            # Velocity dispersion
            self.sigStR[i] = np.sqrt(np.sum(stM*np.power(stVr-self.vStR[i],2.0))/stTotalMass)
            self.sigStPhi[i] = np.sqrt(np.sum(stM*np.power(stVt-self.vStPhi[i],2.0))/stTotalMass)
            self.sigStZ[i] = np.sqrt(np.sum(stM*np.power(stVz-self.vStZ[i],2.0))/stTotalMass)
            # Metallicity:
            self.stZ[i] = np.sum(newStM * annuli[i]["metallicity_fraction"][new_star_particles])/newStTotalMass
            self.stWidthZ[i] = np.sqrt(np.sum(newStM*np.power(annuli[i]["metallicity_fraction"][new_star_particles] - self.stZ[i],2.0))/newStTotalMass)
            # vertical postion
            zStars = np.sum(stM * particleZ[star_particles])/stTotalMass
            self.hStars[i] = np.sqrt(np.sum(stM * np.power(zStars-particleZ[star_particles],2.0))/stTotalMass) # kpc
            # Stellar ages
            self.stAge[i] = np.sum(newStM * pf['years']*(pf['InitialTime']-annuli[i]['creation_time'][new_star_particles]))/newStTotalMass
            self.stAgeSpread[i] = np.sqrt(np.sum(newStM * np.power( \
                                    pf['years']*(pf['InitialTime']-annuli[i]['creation_time'][new_star_particles]),2.0))/newStTotalMass)

            # ... and for gas
            self.sigR[i],self.vR[i] = coldAnnuli[i].quantities["WeightedVariance"]("cyl_RadialVelocity","CellMassMsun") # cm/s
            self.sigPhi[i],self.vPhi[i] = coldAnnuli[i].quantities["WeightedVariance"]("TangentialVelocity","CellMassMsun") #cm/s
            self.sigZ[i],self.vZ[i] = coldAnnuli[i].quantities["WeightedVariance"]("z-velocity","CellMassMsun") # cm/s
            self.hGas[i],self.zGas = coldAnnuli[i].quantities["WeightedVariance"]("z","CellMassMsun") # code units
            self.hGas[i]*=pf['kpc']
            self.zGas*=pf['kpc']
            self.widthZ[i], self.Z[i] = coldAnnuli[i].quantities["WeightedVariance"]("Metal_Fraction","CellMassMsun")
        
            # Compute the mass-weighted thermal velocity dispersion.
            soundSpeeds = coldAnnuli[i]["SoundSpeed"]
            masses = coldAnnuli[i]["CellMassMsun"]
            cylMass = np.sum(masses)
            self.sigTh[i] = np.sum(soundSpeeds*masses)/cylMass
            
            # Compute the logarithmic derivative of the circular velocity, as found from the mass contained in spherical shells
            if(i!=0):
                self.beta[i] = np.log(self.spVcirc[i]/self.spVcirc[i-1])/np.log(self.r[i]/self.r[i-1])
            else:
                self.beta[i] = 1 #shrug - this is correct for a uniform-density sphere, hopefully not too bad as r->0
        
            # Adjust the velocity dispersions of the gas to include the thermal component. 
            # Note self.sig<whatever> is still 'turbulent' only.
            totSigR = np.sqrt(self.sigR[i]**2.0 + self.sigTh[i]**2.0)
            totSigZ = np.sqrt(self.sigZ[i]**2.0 + self.sigTh[i]**2.0)
            
            # Compute the various prerequisites for approximating the total Q.
            W = 2.0*totSigR * self.sigStR[i] / (self.sigStR[i]**2.0 + totSigR**2.0)
            Tg = 0.8 + 0.7 * totSigZ/totSigR
            Tst = 0.8 + 0.7 * self.sigStZ[i]/self.sigStR[i]
            
            # ... including the Q of individual disk components
            self.Qst[i] = self.sigStR[i] * np.sqrt(2.0*(self.beta[i]+1.0)) * self.spVcirc[i] / \
                (self.r[i] * cmperkpc * np.pi * G * self.colSt[i]*gPerMsun/cmperpc**2.0)
            self.Qgas[i] = totSigR * np.sqrt(2.0*(self.beta[i]+1.0)) * self.spVcirc[i]/ \
                (self.r[i] * cmperkpc * np.pi * G * self.colGas[i]*gPerMsun/cmperpc**2.0)
            
            self.QWS[i] = 1.0/ (1.0/self.Qst[i] + 1.0/self.Qgas[i]) # The Wang-Silk approximation
            # The Romeo-Wiegert approximation:
            if(self.Qst[i]*Tst >= self.Qgas[i]*Tg):
                self.QRW[i] = 1.0 / (  W/(self.Qst[i]*Tst) + 1.0/(self.Qgas[i]*Tg))
            else:
                self.QRW[i] = 1.0 / ( 1.0/(self.Qst[i]*Tst) + W/(self.Qgas[i]*Tg))
                
            self.logDensityWidth[i], self.avgLogDensity[i] = coldAnnuli[i].quantities["WeightedVariance"]("logDensity","CellMassMsun")

            # Various estimates of the freefall or dynamical time.
            self.tff[i] = np.sqrt(3*np.pi/32.0)/np.sqrt(G*cylMass*gPerMsun/(annularVolume*cmperkpc**3)) # seconds
            self.tff2[i] = np.sqrt(3*np.pi/32.0)/np.sqrt(G*10.0**(self.avgLogDensity[i]))# estimate from average log density -- seconds
            self.tff3[i] = 2.0*self.hGas[i]*cmperkpc / self.sigTh[i]# estimate from mass-weighted average dynamical time --seconds
            
            # Check the actual volume in the data objects being analyzed compared to 2 pi (r_out^2 - r_in^2) h.
            analyticArea = np.pi*self.r[i]**2
            if i!=0:
                analyticArea-=np.pi*self.r[i-1]**2.0
            coldVolumeFillingFactor = annularVolume/(analyticArea*self.height*2.0)
            print("actual volume / analytic volume: ",coldVolumeFillingFactor)

            # Finally, check the present star formation rate in each annulus.
            # Something fancier could be done here, e.g. looking at the average rate over some time in the past.
            times,sfrs = sfrFromParticles(pf, annuli[i], new_star_particles, 
                                          times = [pf['InitialTime']*pf['years']])
            self.sfrs[i] = sfrs[0] # Msun/yr
            self.colSFRs[i] = self.sfrs[i] / (1.0e6 * self.annArea[i]) # solar masses / yr / pc^2
            

        # Done!
        return

 
# Now we have a bunch of convenience functions for plotting the quantities we just extracted.

def plotSfLaw(annuli, axIn=None, basename="sfLaw", filetype="png"):
    fig,ax=axisSwitch(axIn)
    colPerT = np.sort(annuli.colGas/(annuli.tff2/speryear))
    ax.plot(colPerT, colPerT*0.01,ls='--',lw=2, label='0.01 efficiency per tff')
    ax.scatter(annuli.colGas/(annuli.tff2/speryear), annuli.colSFRs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Msun / yr / sq pc')
    ax.set_ylabel('Msun / yr / sq pc')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotSfGasLaw(annuli, axIn=None, basename="sfGasLaw", filetype="png"):
    fig,ax=axisSwitch(axIn)
    colGas = np.sort(annuli.colGas)
    KS = 2.5e-10 * np.power(colGas,1.4) # Kennicutt 1998 equation 4.
    ax.plot(colGas,KS,ls='--',lw=2,color='gray',label='Kennicutt 98')
    ax.plot(colGas,KS*1.8/2.5,ls='--',lw=1,color='gray')
    ax.plot(colGas,KS*3.2/2.5,ls='--',lw=1,color='gray')
    ax.scatter(annuli.colGas,annuli.colSFRs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Msun / sq pc')
    ax.set_ylabel('Msun / yr / sq pc')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotTimescales(annuli, axIn=None, basename="timescales", filetype="png"):
    fig,ax=axisSwitch(axIn)
    ax.plot(annuli.r,annuli.tff/speryear,ls='-',lw=2,label='from mean density')
    ax.scatter(annuli.r,annuli.tff/speryear)
    ax.plot(annuli.r,annuli.tff2/speryear,ls='-',lw=2,label='from median density')
    ax.scatter(annuli.r,annuli.tff2/speryear)
    ax.plot(annuli.r,annuli.tff3/speryear,ls='-',lw=2,label='h/cs')
    ax.scatter(annuli.r,annuli.tff3/speryear)
    ax.plot(annuli.r,annuli.tOrb,ls='-',lw=2,label='orbital time')
    ax.scatter(annuli.r,annuli.tOrb)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('dynamical times (yrs)')
    ax.legend()
    return returnSwitch(fig, ax, basename, filetype, annuli.pfName)
    
def plotColumnDensities(annuli, axIn=None, basename="col", filetype="png"):
    colTot = annuli.colGas+annuli.colSt+annuli.colDM

    fig,ax=axisSwitch(axIn)
    ax.scatter(annuli.r,annuli.colGas,color='b')
    ax.scatter(annuli.r,annuli.colSt,color='g')
    ax.scatter(annuli.r,annuli.colDM,color='r')
    ax.scatter(annuli.r,annuli.col,color='k')
    ax.scatter(annuli.r,colTot,color='k')
    ax.plot(annuli.r,annuli.colGas,color='b',label='gas')
    ax.plot(annuli.r,annuli.colSt,color='g',label='stars')
    ax.plot(annuli.r,annuli.colDM,color='r',label='DM')
    ax.plot(annuli.r,annuli.col,color='k',label='total')
    ax.plot(annuli.r,colTot,color='k',ls='--',label='added',lw=3)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Column Density (Msun per pc sq)')
    return returnSwitch(fig, ax, basename, filetype, annuli.pfName)

def plotSphericalMasses(annuli, axIn=None, basename="spheres", filetype="png"):
    fig,ax=axisSwitch(axIn)
    r = annuli.r 
    spTotalMass = annuli.spMassEnclosed 
    spDM = annuli.spMassDM 
    spSt = annuli.spMassSt 
    spGas = annuli.spMassSt 
    spTot = spGas+spSt+spDM

    ax.scatter(r,spGas,color='b')
    ax.scatter(r,spSt,color='g')
    ax.scatter(r,spDM,color='r')
    ax.scatter(r,spTotalMass,color='k')
    ax.scatter(r,spTot,color='k')
    ax.plot(r,spGas,color='b',label='gas')
    ax.plot(r,spSt,color='g',label='stars')
    ax.plot(r,spDM,color='r',label='DM')
    ax.plot(r,spTotalMass,color='k',label='total')
    ax.plot(r,spTot,color='k',ls='--',label='added',lw=3)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Mass contained in spherical shells (Msun)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotVelocityDispersions(annuli, axIn=None, basename="dispersions", filetype="png"):
    fig,ax=axisSwitch(axIn)

    r = annuli.r 
    sigR = annuli.sigR 
    sigPhi = annuli.sigPhi 
    sigZ = annuli.sigZ 
    sigth = annuli.sigTh 
    # Add in the thermal component to each direction
    sigR = np.sqrt(np.power(sigR,2.0)+np.power(sigth,2.0))
    sigPhi = np.sqrt(np.power(sigPhi,2.0)+np.power(sigth,2.0))
    sigZ = np.sqrt(np.power(sigZ,2.0)+np.power(sigth,2.0))

    sigStR = annuli.sigStR
    sigStPhi = annuli.sigStPhi 
    sigStZ = annuli.sigStZ 

    # Gas
    ax.scatter(r,sigR*1.0e-5,color='b')
    ax.scatter(r,sigPhi*1.0e-5,color='g')
    ax.scatter(r,sigZ*1.0e-5,color='r')
    ax.plot(r,sigth*1.0e-5,color='k')
    ax.plot(r,sigR*1.0e-5,label='R (gas)',color='b')
    ax.plot(r,sigPhi*1.0e-5,label='Phi (gas)',color='g')
    ax.plot(r,sigZ*1.0e-5,label='Z (gas)',color='r')
    ax.plot(r,sigth*1.0e-5,label="Sound speed",color='k')
    # Stars
    ax.scatter(r,sigStR*1.0e-5,color='b')
    ax.scatter(r,sigStPhi*1.0e-5,color='g')
    ax.scatter(r,sigStZ*1.0e-5,color='r')
    ax.plot(r,sigStR*1.0e-5,label='R (stars)',color='b',ls='--')
    ax.plot(r,sigStPhi*1.0e-5,label='Phi (stars)',color='g',ls='--')
    ax.plot(r,sigStZ*1.0e-5,label='Z (stars)',color='r',ls='--')
    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Velocity Dispersion (km/s)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotBulkVelocity(annuli, axIn=None, basename="bulk", filetype="png"):
    r = annuli.r 
    vR = annuli.vR 
    vPhi = annuli.vPhi 
    vZ = annuli.vZ

    vStR = annuli.vStR 
    vStPhi = annuli.vStPhi 
    vStZ = annuli.vStZ 

    fig,ax=axisSwitch(axIn)
    # Gas
    ax.scatter(r,vR*1.0e-5,color='b')
    ax.scatter(r,vPhi*1.0e-5,color='g')
    ax.scatter(r,vZ*1.0e-5,color='r')
    ax.plot(r,vR*1.0e-5,label='R (gas)',color='b')
    ax.plot(r,vPhi*1.0e-5,label='Phi (gas)',color='g')
    ax.plot(r,vZ*1.0e-5,label='Z (gas)',color='r')
    # Stars
    ax.scatter(r,vStR*1.0e-5,color='b')
    ax.scatter(r,vStPhi*1.0e-5,color='g')
    ax.scatter(r,vStZ*1.0e-5,color='r')
    ax.plot(r,vStR*1.0e-5,label='R (stars)',color='b',ls='--')
    ax.plot(r,vStPhi*1.0e-5,label='Phi (stars)',color='g',ls='--')
    ax.plot(r,vStZ*1.0e-5,label='Z (stars)',color='r',ls='--')
    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Bulk velocity (km/s)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotRotationCurve(annuli, axIn=None, basename="rotation", filetype="png"):
    fig,ax=axisSwitch(axIn)
        
    r = annuli.r 
    vPhi = annuli.vPhi 
    vStPhi = annuli.vStPhi 
    spVcirc = annuli.spVcirc 
    
    ax.scatter(r,vPhi*1.0e-5,color='b')
    ax.scatter(r,spVcirc*1.0e-5,color='r')
    ax.plot(r,vPhi*1.0e-5,label='Average in annulus (gas)',color='b')
    ax.scatter(r,vStPhi*1.0e-5,color='b')
    ax.plot(r,vStPhi*1.0e-5,label='Average in annulus (stars)',color='b',ls='--')
    ax.plot(r,spVcirc*1.0e-5,label='From total mass enclosed',color='r')
    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Rotational Velocity (km/s)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotStability(annuli, axIn=None, basename="toomre", filetype="png"):
    fig,ax=axisSwitch(axIn)
    r = annuli.r[1:] 
    QRW = annuli.QRW[1:] 
    Qgas = annuli.Qgas[1:] 
    Qst = annuli.Qst[1:] 
    QWS = annuli.QWS[1:] 

    ax.scatter(r,QWS,color='b')
    ax.plot(r,QWS,color='b',label=r'Wang-Silk')
    ax.scatter(r,QRW,color='k')
    ax.plot(r,QRW,color='k',label=r'Romeo-Wiegert')
    ax.scatter(r,Qgas,color='r')
    ax.plot(r,Qgas,color='r',label=r'Gas')
    ax.scatter(r,Qst,color='g')
    ax.plot(r,Qst,color='g',ls='--',label=r'Stars')
    ax.plot(r,r*0+1.0,label=r'Approx Marginal Stability',color='b',ls='--')
    ax.plot(r,r*0+2.0,label=r'Approx Marginal Stability w/ Finite Thickness',color='k',ls='--')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Q')
    ax.set_yscale('log')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotFluxes(annuli, axIn=None, basename="verticalFlux", filetype="png"):
    fig,ax = axisSwitch(axIn)
    for j,h in enumerate(annuli.fluxHeights):
        colorPart = float(j+5)/float(len(annuli.fluxHeights)+5)
        print(colorPart)
        theColor = (colorPart,colorPart,colorPart)
        ax.plot(annuli.r, (annuli.cylUpFluxes[:,j]+annuli.cylDownFluxes[:,j])/(4.0*annuli.annArea), 
                c=str(colorPart),lw=int(5*colorPart),label='Height '+str(h)+' kpc')
    ax.plot(annuli.r, annuli.colSFRs*1.0e6, color='r', lw=2, label='col dens SFR')
    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Vertical flux (Msun/yr/sq kpc)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotScaleheights(annuli, axIn=None, basename="scaleheights", filetype="png"):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.hGas, color='blue', label='gas')
    ax.scatter(annuli.r, annuli.hGas, color='blue')
    ax.plot(annuli.r, annuli.hStars, color='red', label='stars')
    ax.scatter(annuli.r, annuli.hStars, color='red')

    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Scaleheight (kpc)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotNResolutionElements(annuli, axIn=None, basename="nResElements", filetype="png"):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.Ngas, color='blue', label='gas')
    ax.scatter(annuli.r, annuli.Ngas, color='blue')
    ax.plot(annuli.r, annuli.Nstars, color='red', label='stars')
    ax.scatter(annuli.r, annuli.Nstars, color='red')

    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Number of Resolution Elements')
    ax.set_yscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotMetallicity(annuli, axIn=None, basename='metallicity', filetype='png'):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.Z, color='blue', label='gas')
    ax.plot(annuli.r, annuli.Z+annuli.widthZ, color='blue', ls='--')
    ax.plot(annuli.r, annuli.Z-annuli.widthZ, color='blue', ls='--')
    ax.scatter(annuli.r, annuli.Z, color='blue')
    ax.plot(annuli.r, annuli.stZ, color='red', label='stars')
    ax.plot(annuli.r, annuli.stZ+annuli.stWidthZ, color='red', ls='--')
    ax.plot(annuli.r, annuli.stZ-annuli.stWidthZ, color='red', ls='--')
    ax.scatter(annuli.r, annuli.stZ, color='red')

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Metallicity')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotStellarAges(annuli, axIn=None, basename='stellarAges', filetype='png'):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.stAge, color='blue', label='gas')
    ax.plot(annuli.r, annuli.stAge+annuli.stAgeSpread, color='blue', ls='--')
    ax.plot(annuli.r, annuli.stAge-annuli.stAgeSpread, color='blue', ls='--')
    ax.scatter(annuli.r, annuli.stAge, color='blue')

    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Stellar Ages (yrs)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)

def plotDensityDistribution(annuli, axIn=None, basename='densDist', filetype='png'):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.avgLogDensity, color='blue', label='gas')
    ax.plot(annuli.r, annuli.avgLogDensity+annuli.logDensityWidth, color='blue', ls='--')
    ax.plot(annuli.r, annuli.avgLogDensity-annuli.logDensityWidth, color='blue', ls='--')
    ax.scatter(annuli.r, annuli.avgLogDensity, color='blue')

    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Log Density (g/cc)')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName)


# A few convenience functions for quick looks at the results.
plotFunctions = [plotStability, plotRotationCurve, plotBulkVelocity, 
                 plotVelocityDispersions, plotSphericalMasses,
                 plotColumnDensities, plotTimescales, plotSfGasLaw, 
                 plotSfLaw, plotFluxes, plotScaleheights, plotNResolutionElements,
                 plotMetallicity, plotStellarAges, plotDensityDistribution ]
def showAllPlots(annuli):
    for fn in plotFunctions:
        fn(annuli,filetype='screen')
def saveAllPlots(annuli):
    for fn in plotFunctions:
        fn(annuli,filetype='png')   


if __name__ == '__main__':

    # A use case

    #pf = load('/pfs/jforbes/run-enzo-dev-jforbes/dwarf_nf35/DD0172/DD0172')
    pf = load('/data/hcho/MyAgoraGalaxy/run_200130_640pc/DD0010/DD0010')
    fluxHeights = [1.0/1000.0 * 8**i for i in range(1)]#(4)] # 1, 8, 64, 512 pc

    # Look at annuli between 30 pc and 600 pc
    # Use 20 annuli
    # Up to a height of 100 pc
    # Centered at [.5,.5,.5] in code units
    # Look at vertical fluxes at the heights allocated above.
    #annuli = annularAnalysis(.03, 0.6, 20, 0.1, [.5,.5,.5],fluxHeights) # set up
    annuli = annularAnalysis(1.2, 35., 20, 4., [.5,.5,.5],fluxHeights) # set up
    annuli.run(pf) # actually run the analysis

    PlotStability(annuli,filetype='png')

    #saveAllPlots(annuli) # save the plots as <pfName>_<plotName>.png

