#!/home/hcho/venv/bin/python

# coding: utf-8
# Radial profiles
# downloaded from
# https://bitbucket.org/ngoldbaum/galaxy_analysis/src/default/annularAnalysis.py
# 06/01/2020
# Hyerin Cho

import yt
from yt.mods import *
from yt import units
from yt import YTArray
import matplotlib.pyplot as plt
from sfrFromParticles import *
import pdb
from yt.units import G
import os
from os import path

   
#G=6.67384e-8
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
    
def returnSwitch(fig,ax,basename,filetype,pfName,pfDir):
    ''' This provides the return value for the plotting functions below. It's always the filename, but if no
    filetype is provided or if the filetype is explicitly 'screen' as in an iPython notebook, we don't save the plot anywhere.
    However if filetype is something else (hopefully something reasonable like png or pdf), save the figure. '''
    filename = './'+pfDir+'/'+pfName+"_"+basename+'.'+filetype
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
        self.inner = inner*units.kpc
        self.outer = outer*units.kpc
        self.Nsteps = Nsteps
        self.height = height*units.kpc
        self.center = center
        self.fluxHeights = fluxHeights
        
        # Set up arrays to store the quantities we're going to compute.
        self.r=YTArray(np.zeros(Nsteps),'kpc') # radii of boundaries between annuli (kpc)
        self.rFinest=YTArray(np.zeros(1),'kpc') # finest cell width (kpc)
        self.spMassEnclosed=YTArray(np.zeros(Nsteps),'Msun') # mass enclosed in a sphere of radius r[i] (Msun)
        self.cylMassEnclosed=YTArray(np.zeros(Nsteps),'Msun') # mass enclosed in a cylinder of radius r[i] and half-height self.height (Msun)
        self.cylMassEnclosedTallDisks=YTArray(np.zeros(Nsteps),'Msun') # mass enclosed in a cylinder of radius r[i] and half-height self.height (Msun)
        self.spVcirc=YTArray(np.zeros(Nsteps),'cm/s') # circular velocity calculated from spherical mass enclosed (cm/s)
        self.annMass=YTArray(np.zeros(Nsteps),'Msun') # mass in an annulus (Msun)
        self.annArea=YTArray(np.zeros(Nsteps),'kpc**2') # area of an annulus (kpc^2)
        self.col=YTArray(np.zeros(Nsteps),'Msun/pc**2') # column density in an annulus (Msun/pc^2)
        self.colGas=YTArray(np.zeros(Nsteps),'Msun/pc**2') # column density of gas in an annulus (Msun/pc^2)
        self.colGas2=YTArray(np.zeros(1),'Msun/pc**2') # column density of gas in a cell MBH is in (Msun/pc^2)
        self.colGas3=YTArray(np.zeros(1),'Msun/pc**2') # column density of gas in a cell MBH is in (Msun/pc^2)
        self.colSt=YTArray(np.zeros(Nsteps),'Msun/pc**2') # column density of stars in an annulus (Msun/pc^2)
        self.colDM=YTArray(np.zeros(Nsteps),'Msun/pc**2') # column density in dark matter in an annulus (Msun/pc^2)
        self.sigR=YTArray(np.zeros(Nsteps),'cm/s') # non-thermal velocity dispersion of gas along cylindrical rHat in a given radius (cm/s)
        self.sigPhi=YTArray(np.zeros(Nsteps),'cm/s') # same but along cylindrical thetaHat (cm/s)
        self.sigZ=YTArray(np.zeros(Nsteps),'cm/s') # same but along zHat (cm/s)
        self.sigStR=YTArray(np.zeros(Nsteps),'cm/s') # The next three are the same as those above, but for star particles.
        self.sigStPhi=YTArray(np.zeros(Nsteps),'cm/s')
        self.sigStZ=YTArray(np.zeros(Nsteps),'cm/s')
        self.vR=YTArray(np.zeros(Nsteps),'cm/s') # The average velocity of gas in cylindrical rHat in the annulus (cm/s)
        self.vPhi=YTArray(np.zeros(Nsteps),'cm/s') # Same but along cylindrical thetaHat (cm/s)
        self.vZ=YTArray(np.zeros(Nsteps),'cm/s') # Same but along zHat (cm/s)
        self.vStR=YTArray(np.zeros(Nsteps),'cm/s') # The next three are the same, except using star particles.
        self.vStPhi=YTArray(np.zeros(Nsteps),'cm/s')
        self.vStZ=YTArray(np.zeros(Nsteps),'cm/s')
        self.beta=np.zeros(Nsteps) # an estimate of the powerlaw slope of the rotation curve dln v_phi / dln r (dimensionless)
        self.QWS=np.zeros(Nsteps) # the Wang-Silk estimate of 2-component axisymmetric stability (dimensionless)
        self.QRW=np.zeros(Nsteps) # the Romeo-Wiegert estimate of the same (dimensionless)
        self.Qgas=np.zeros(Nsteps) # Q for the gas only (dimensionless)
        self.Qst=np.zeros(Nsteps) # Q for the stars only (dimensionless)
        self.QHC=np.zeros(Nsteps) # Q added by Hyerin, following Goodman(2003) (dimensionless)
        self.QHC2=np.zeros(Nsteps) # Q added by Hyerin, testing (dimensionless)
        self.hGas=YTArray(np.zeros(Nsteps),'kpc') # The mass-weighted standard deviation of gas height. Taken to be an estimate of the scale height (kpc)
        self.hGas2=YTArray(np.zeros(Nsteps),'kpc') # The mass-weighted standard deviation of gas height. Taken to be an estimate of the scale height (kpc)
        self.hStars=YTArray(np.zeros(Nsteps),'kpc') # same for the stars
        self.Nstars=np.zeros(Nsteps) # Number of new star particles
        self.Ngas = np.zeros(Nsteps) # Number of resolution elements
        self.spMassDM=YTArray(np.zeros(Nsteps),'Msun') # Mass contained in DM in sphere of radius r[i] (Msun)
        self.spMassSt=YTArray(np.zeros(Nsteps),'Msun') # same for star particles
        self.spMassGas=YTArray(np.zeros(Nsteps),'Msun') # same for gas
        self.sigTh=YTArray(np.zeros(Nsteps),'cm/s') # thermal velocity dispersion (cm/s)
        self.tOrb = np.zeros(Nsteps) # orbital time (yrs)
        self.Z = np.zeros(Nsteps) # metallicity of gas
        self.widthZ=np.zeros(Nsteps) # standard deviation of gas metallicity
        self.stZ = np.zeros(Nsteps) # stellar metallicity
        self.stWidthZ = np.zeros(Nsteps) # standard deviation of stellar metallicity
        self.stAge = np.zeros(Nsteps) # stellar age (years)
        self.stAgeSpread = np.zeros(Nsteps) # standard deviation of stellar age (years)
        self.tff = YTArray(np.zeros(Nsteps),'s') # freefall time -- from mean density in the annular volume(s)
        self.tff2 = YTArray(np.zeros(Nsteps),'s') # freefall time -- from the mass-weighted mean of the log in the annular volume (s)
        self.tff3 = YTArray(np.zeros(Nsteps),'s') # freefall time -- from the scale height crossing time (s)
        self.sfrs = YTArray(np.zeros(Nsteps),'Msun/yr') # star formation rate (Msun/yr)
        self.sfrs2 = YTArray(np.zeros(Nsteps),'Msun/yr') # star formation rate (Msun/yr)
        self.colSFRs = YTArray(np.zeros(Nsteps),'Msun/yr/pc**2') # star formation column density (Msun/yr/pc^2)
        self.avgLogDensity = np.zeros(Nsteps) # mass-weighted average of the mean of the log of the density (log_10 of g/cm^3)
        self.logDensityWidth = np.zeros(Nsteps) # standard deviation of the above (dex)
        self.avgRadius = np.zeros(Nsteps) # mass-weighted average radius of the bin.
        self.cylUpFluxes = YTArray(np.zeros((Nsteps,10)),'Msun/yr') # upward mass flux through the annulus
        self.cylDownFluxes = YTArray(np.zeros((Nsteps,10)),'Msun/yr') # downward mass flux through the annulus
        self.cylOutFluxes = YTArray(np.zeros(Nsteps),'Msun/yr') # outward mass flux through the annulus
        
    def run(self, pf, temperatureCut=False, rlogscale=True):
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
        dir_splitted = pf.directory.split('/')
        self.pfDir = dir_splitted[np.argwhere(np.array([1 if 'run' in t else 0 for t in dir_splitted])==1)[0,0]]
        if not path.exists('./'+self.pfDir): # make folder if doesn't exist
            os.mkdir('./'+self.pfDir)

        # create MBH filter
        def mbh(pfilter, data):
            filter = data[(pfilter.filtered_type, "particle_type")] == 8
            return filter
        yt.add_particle_filter("mbh", function=mbh, filtered_type='all', requires=["particle_type"])
        pf.add_particle_filter('mbh')
        # add gas density and dx field
        pf.add_mesh_sampling_particle_field(('gas', 'density'), ptype='mbh')
        pf.add_mesh_sampling_particle_field(('gas', 'cell_mass'), ptype='mbh')
        pf.add_mesh_sampling_particle_field(('gas', 'dx'), ptype='mbh')

        # Now let's start the main loop over annuli.
        for i in range(self.Nsteps):
            print("Starting analysis for disk i=",i) # tell the user what we're up to
            
            if rlogscale:
                # logscale
                self.r[i] = np.exp(np.log(self.inner)+\
                float(i)*(np.log(self.outer)-np.log(self.inner))/float(self.Nsteps-1))*units.kpc
            else:
                self.r[i] = (self.inner + float(i)*(self.outer-self.inner)/float(self.Nsteps-1)).to('kpc')
            print(self.r[i])
            
            # Find particles in this spherical shell. 
            sphere=pf.h.sphere(self.center, (self.r[i]))
            dmp = sphere["particle_type"]==1 # Dark matter
            stp = np.logical_or(sphere["particle_type"]==2, sphere["particle_type"]==4) # Stars
            bhp = sphere["particle_type"]==8 # MBH

            # Add up mass
            self.spMassEnclosed[i] = (np.sum(sphere.quantities["TotalMass"]())).to('Msun')
            self.spMassDM[i] = (np.sum(sphere["particle_mass"][dmp])).to('Msun')
            self.spMassSt[i] = (np.sum(sphere["particle_mass"][stp])).to('Msun')
            # TODO! Add Blackhole mass too
            self.spMassGas[i] = (np.sum(sphere["cell_mass"])).to('Msun')
            
            # A couple quantities derived from the spherical mass enclosed:
            self.spVcirc[i] = (np.sqrt(G*self.spMassEnclosed[i]/(self.r[i]))).to('cm/s') # v^2/r = GM/r^2 -- cm/s
            self.tOrb[i] = (self.r[i]/(self.spVcirc[i])).to('yr') # years
            
            # Set up a series of concentric disks. disks is used for most of the analysis, but we need
            # tall disks, i.e. ones with larger heights, to measure the mass flux at various heights.
            disks[i] = pf.h.disk(self.center,[0,0,1],(self.r[i]),(self.height))
            tallDisks[i] = pf.h.disk(self.center,[0,0,1],(self.r[i]),\
                                     (max(self.fluxHeights)*1.0001,'kpc'))
            self.cylMassEnclosed[i] = (np.sum(disks[i].quantities["TotalMass"]())).to('Msun')
            self.cylMassEnclosedTallDisks[i] = (np.sum(tallDisks[i].quantities["TotalMass"]())).to('Msun')

#            coldDisks[i] = disk.cut_region( ["grid['Temperature'] < 10.0**5"] )
            if(i != 0):
                # Identify the annulus as this cold disk minus the previous cold disk
                #annuli[i] = pf.h.boolean([disks[i], "NOT", disks[i-1]])
                annuli[i] = disks[i] - disks[i-1]
                #tallAnnuli[i] = pf.h.boolean([tallDisks[i], "NOT", tallDisks[i-1]])
                tallAnnuli[i] = tallDisks[i] - tallDisks[i-1]
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
            # TODO: Hyerin- I don't need this so skipping the check for now,
            # but should check this at some point
            for j, h in enumerate(self.fluxHeights):
                surf = pf.surface(tallAnnuli[i],'z',self.center[2]*pf.length_unit+h*units.kpc)
                self.cylUpFluxes[i,j] = (surf.calculate_flux("x-velocity","y-velocity","z-velocity","density")).to('Msun/yr')
                #self.cylUpFluxes[i,j]=(tallAnnuli[i].calculate_isocontour_flux('z',\
                #                    self.center[2]*pf.length_unit+h*units.kpc, 'x-velocity', 'y-velocity', 'z-velocity', 'Density')*units.cm**2.0/conv )
                surf = pf.surface(tallAnnuli[i],'z',self.center[2]*pf.length_unit-h*units.kpc)
                self.cylDownFluxes[i,j] = (surf.calculate_flux("x-velocity","y-velocity","z-velocity","density")).to('Msun/yr')
                #self.cylDownFluxes[i,j]= ( -1.0 * tallAnnuli[i].calculate_isocontour_flux('z',\
                #                          self.center[2]*pf.length_unit-h*units.kpc,'x-velocity','y-velocity','z-velocity', 'Density')*units.cm**2.0/conv )
                #self.cylOutFluxes[i,j]=(tallAnnuli[i].calculate_isocontour_flux('cylindrical_r',\
                #                          self.r[i],'x-velocity','y-velocity','z-velocity', 'Density')*units.cm**2.0/conv )
            surf = pf.surface(tallDisks[i],'cylindrical_r',self.r[i])
            self.cylOutFluxes[i] = (surf.calculate_flux("x-velocity","y-velocity","z-velocity","density")).to('Msun/yr')
               
            # Get the particle positions in kpc
            particleX = (annuli[i]["particle_position_x"] - self.center[0]*pf.length_unit).to('kpc')
            particleY = (annuli[i]["particle_position_y"] - self.center[1]*pf.length_unit).to('kpc')
            particleR = np.sqrt(np.power(particleX,2.0) + np.power(particleY,2.0))
            particleZ = (annuli[i]["particle_position_z"] - self.center[2]*pf.length_unit).to('kpc')
 
            annularVolume = (np.sum(coldAnnuli[i]['cell_volume'])).to('kpc**3')  # cubic kpc
            supersetAnnularVolume = (np.sum(annuli[i]['cell_volume'])).to('kpc**3')
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
            self.annArea[i] = (annularVolume/(2.0*self.height)).to('kpc**2') # square kpc

            # Total gas mass. I would have naively expect this to include particle mass, but I /think/ that since
            # particles have no information on 'Temperature', the coldAnnuli refer only to the gas. Any assignment
            # of particles to these annuli has to be done by hand, as with the logic on the 'particles' array above.
            self.annMass[i] = (np.sum(coldAnnuli[i].quantities["TotalMass"]())).to('Msun') # solar masses
            self.col[i] = (self.annMass[i]/(self.annArea[i])).to('Msun/pc**2') # Msun/pc^2 
            self.colGas[i] = ((coldAnnuli[i].quantities["TotalQuantity"]("cell_mass"))/(self.annArea[i])).to('Msun/pc**2') 
            # TODO! test colGas approximation at BH cell rho*R
            if i == 0:
                # only inside the innermost annulus, get rho & dx for the cell
                # that BH resides in
                self.colGas2 = (coldAnnuli[i]["mbh","cell_gas_density"]*coldAnnuli[i]["mbh","cell_gas_dx"]).to('Msun/pc**2')
                self.colGas3 = (coldAnnuli[i]["mbh","cell_gas_cell_mass"]/coldAnnuli[i]["mbh","cell_gas_dx"]**2).to('Msun/pc**2')
                self.rFinest = (coldAnnuli[i]["mbh","cell_gas_dx"]).to('kpc')

                # calculate \Omega at the MBH cell
                self.spVcirc2 = (coldAnnuli[i]["mbh","cell_gas_dx"]/2. * np.sqrt(8.*G*coldAnnuli[i]["mbh","cell_gas_density"] +\
                           G*coldAnnuli[i]["mbh","particle_mass"] / np.power(coldAnnuli[i]["mbh","cell_gas_dx"]/2.,3.0))).to('cm/s')

            
            # Split up the particles which are in this annulus into stars and DM. 
            star_particles = np.logical_or( annuli[i]["particle_type"]==2, annuli[i]["particle_type"]==4 )
            new_star_particles = np.logical_and(annuli[i]["particle_type"]==2, annuli[i]["creation_time"]>0)
            dm_particles = annuli[i]["particle_type"]==1

            self.Nstars[i]=np.sum(star_particles)
            self.Ngas[i]=len(annuli[i][("gas","cell_mass")])
            self.colSt[i] = (np.sum(annuli[i]["particle_mass"][star_particles]) / (self.annArea[i])).to('Msun/pc**2')
            self.colDM[i] = (np.sum(annuli[i]["particle_mass"][dm_particles]) / (self.annArea[i])).to('Msun/pc**2')
            
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
            stM =  annuli[i]["particle_mass"][star_particles]
            newStM = annuli[i]["particle_mass"][new_star_particles]
            
            # Compute moments of the velocity, positions, and metals for stars
            stTotalMass = np.sum(stM)
            newStTotalMass = np.sum(newStM)
            # Bulk velocity:
            self.vStR[i] = (np.sum(stM*stVr)/stTotalMass).to('cm/s')
            self.vStPhi[i] = (np.sum(stM*stVt)/stTotalMass).to('cm/s')
            self.vStZ[i] = (np.sum(stM*stVz)/stTotalMass).to('cm/s')
            # Velocity dispersion
            self.sigStR[i] = (np.sqrt(np.sum(stM*np.power(stVr-self.vStR[i],2.0))/stTotalMass)).to('cm/s')
            self.sigStPhi[i] = (np.sqrt(np.sum(stM*np.power(stVt-self.vStPhi[i],2.0))/stTotalMass)).to('cm/s')
            self.sigStZ[i] = (np.sqrt(np.sum(stM*np.power(stVz-self.vStZ[i],2.0))/stTotalMass)).to('cm/s')
            # Metallicity:
            self.stZ[i] = np.sum(newStM * annuli[i]["metallicity_fraction"][new_star_particles])/newStTotalMass
            self.stWidthZ[i] = np.sqrt(np.sum(newStM*np.power(annuli[i]["metallicity_fraction"][new_star_particles] - self.stZ[i],2.0))/newStTotalMass)
            # vertical postion
            zStars = np.sum(stM * particleZ[star_particles])/stTotalMass
            self.hStars[i] = (np.sqrt(np.sum(stM * np.power(zStars-particleZ[star_particles],2.0))/stTotalMass)).to('kpc') # kpc
            # Stellar ages
            self.stAge[i] = np.sum(newStM * (pf.current_time-annuli[i]['creation_time'][new_star_particles]).to('yr'))/newStTotalMass
            self.stAgeSpread[i] = np.sqrt(np.sum(newStM * np.power( \
                                    (pf.current_time-annuli[i]['creation_time'][new_star_particles]).to('yr'),2.0))/newStTotalMass)

            # ... and for gas
            # TODO! Try relative velocity wrt MBH for both gas and stars

            gasX = (annuli[i][("gas","x")] - self.center[0]*pf.length_unit).to('kpc')
            gasY = (annuli[i][("gas","y")] - self.center[1]*pf.length_unit).to('kpc')
            gasZ = (annuli[i][("gas","z")] - self.center[2]*pf.length_unit).to('kpc')
            gasVx = annuli[i][("gas","velocity_x")]
            gasVy = annuli[i][("gas","velocity_y")]
            gasTh = np.arctan2(gasY,gasX)
            gasCosTh = np.cos(gasTh)
            gasSinTh = np.sin(gasTh)
            gasVr = gasVx*gasCosTh + gasVy*gasSinTh
            gasVt = -gasVx*gasSinTh + gasVy*gasCosTh
            gasVz = annuli[i][("gas","velocity_z")]
            gasM = annuli[i]["cell_mass"]
            gasTotalMass = np.sum(gasM)
            
            self.vR[i]= (np.sum(gasM*gasVr)/gasTotalMass).to('cm/s')
            self.vPhi[i] = (np.sum(gasM*gasVt)/gasTotalMass).to('cm/s')
            self.vZ[i] = (np.sum(gasM*gasVz)/gasTotalMass).to('cm/s')

            self.sigR[i] = (np.sqrt(np.sum(gasM*np.power(gasVr-self.vR[i],2.0))/gasTotalMass)).to('cm/s')
            self.sigPhi[i] = (np.sqrt(np.sum(gasM*np.power(gasVt-self.vPhi[i],2.0))/gasTotalMass)).to('cm/s')
            self.sigZ[i] = (np.sqrt(np.sum(gasM*np.power(gasVz-self.vZ[i],2.0))/gasTotalMass)).to('cm/s')

            self.zGas = np.sum(gasM*gasZ)/gasTotalMass
            self.hGas[i] = (np.sqrt(np.sum(gasM*np.power(self.zGas-gasZ,2.0))/gasTotalMass)).to('kpc')
            # test by Hyerin
            self.hGas2[i]  = (np.power(self.sigR[i],2.0)/\
                              (np.pi  * G * self.colGas[i])).to('kpc')

            if 0: # velocities are not quite right.
                temp = coldAnnuli[i].quantities["WeightedVariance"]("cylindrical_radial_velocity","cell_mass")
                if str(temp[0].units) == 'dimensionless':
                    temp *= units.cm/units.s
                self.sigR[i],self.vR[i] = (temp).to('cm/s') # cm/s
                temp = coldAnnuli[i].quantities["WeightedVariance"]("cylindrical_tangential_velocity","cell_mass")
                if str(temp[0].units) == 'dimensionless':
                    temp *= units.cm/units.s
                self.sigPhi[i],self.vPhi[i] = (temp).to('cm/s') #cm/s
                temp = coldAnnuli[i].quantities["WeightedVariance"]("z-velocity","cell_mass")
                if str(temp[0].units) == 'dimensionless':
                    temp *= pf.velocity_unit
                self.sigZ[i],self.vZ[i] = (temp).to('cm/s') # cm/s
                temp = coldAnnuli[i].quantities["WeightedVariance"]("z","cell_mass")
                if str(temp[0].units) == 'dimensionless':
                    temp *= units.cm
                self.hGas[i],self.zGas = (temp).to('kpc') # *units.kpc#kpc
            self.widthZ[i], self.Z[i] = (coldAnnuli[i].quantities["WeightedVariance"]("metallicity","cell_mass")).to('') # dimensionless
        
            # Compute the mass-weighted thermal velocity dispersion.
            soundSpeeds = coldAnnuli[i]["sound_speed"]
            masses = (coldAnnuli[i]["cell_mass"]).to('Msun')
            cylMass = np.sum(masses)
            self.sigTh[i] = (np.sum(soundSpeeds*masses)/cylMass).to('cm/s')
            
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
            self.Qst[i] = (self.sigStR[i] * np.sqrt(2.0*(self.beta[i]+1.0)) * self.spVcirc[i] / \
                (self.r[i] * np.pi * G * self.colSt[i])).to('')
            self.Qgas[i] = (totSigR * np.sqrt(2.0*(self.beta[i]+1.0)) * self.spVcirc[i]/ \
                (self.r[i] * np.pi * G * self.colGas[i])).to('')
            # Q equation in Goodman (2003)
            self.QHC[i] = (self.sigTh[i]*self.spVcirc[i]/\
                           (self.r[i]*np.pi*G*self.colGas[i])).to('')
            # Q equation test
            self.QHC2[i] = (self.sigR[i] * np.sqrt(2.0*(self.beta[i]+1.0)) * self.spVcirc[i]/ \
                (self.r[i] * np.pi * G * self.colGas[i])).to('')
            
            self.QWS[i] = 1.0/ (1.0/self.Qst[i] + 1.0/self.Qgas[i]) # The Wang-Silk approximation
            # The Romeo-Wiegert approximation:
            if(self.Qst[i]*Tst >= self.Qgas[i]*Tg):
                self.QRW[i] = 1.0 / (  W/(self.Qst[i]*Tst) + 1.0/(self.Qgas[i]*Tg))
            else:
                self.QRW[i] = 1.0 / ( 1.0/(self.Qst[i]*Tst) + W/(self.Qgas[i]*Tg))
                
            #self.logDensityWidth[i], self.avgLogDensity[i] = coldAnnuli[i].quantities["WeightedVariance"]("logDensity","cell_mass")
            temp = coldAnnuli[i].quantities.weighted_variance("Density","cell_mass")
            if str(temp[0].units) == 'dimensionless':
                temp *= pf.mass_unit/pf.length_unit**3
            self.logDensityWidth[i], self.avgLogDensity[i] = np.log10((temp).to('g/cm**3'))

            # Various estimates of the freefall or dynamical time.
            self.tff[i] = (np.sqrt(3*np.pi/32.0)/np.sqrt(G*cylMass/(annularVolume))).to('s') # seconds
            self.tff2[i] = (np.sqrt(3*np.pi/32.0)/np.sqrt(G*(10.0**(self.avgLogDensity[i]))*units.g/units.cm**3)).to('s')# estimate from average log density -- seconds
            self.tff3[i] = (2.0*self.hGas[i] / self.sigTh[i]).to('s')# estimate from mass-weighted average dynamical time --seconds
            
            # Check the actual volume in the data objects being analyzed compared to 2 pi (r_out^2 - r_in^2) h.
            analyticArea = np.pi*self.r[i]**2
            if i!=0:
                analyticArea-=np.pi*self.r[i-1]**2.0
            coldVolumeFillingFactor = annularVolume/(analyticArea*self.height*2.0)
            print("actual volume / analytic volume: ",coldVolumeFillingFactor)

            # Finally, check the present star formation rate in each annulus.
            # Something fancier could be done here, e.g. looking at the average rate over some time in the past.
            if 1:
                times,sfrs,sfrs2 = sfrFromParticles(pf, annuli[i], new_star_particles, \
                                times = YTArray([(pf.current_time).to('yr')],'yr'))
                self.sfrs[i] = sfrs[0] # Msun/yr
                self.sfrs2[i] = sfrs2[0] # Msun/yr
                self.colSFRs[i] = (self.sfrs[i] / (self.annArea[i])).to('Msun/yr/pc**2') # solar masses / yr / pc^2
            

        # Done!
        return

 
# Now we have a bunch of convenience functions for plotting the quantities we just extracted.

def plotSf(annuli, axIn=None, basename="sfr", filetype="png",rlogscale=True):
    # Added by HC. SFR as a function of radius
    fig,ax=axisSwitch(axIn)
    ax.plot(annuli.r,annuli.sfrs,color='k',lw=2,label='SFR')
    ax.plot(annuli.r,annuli.sfrs2,ls='-',color='gray',lw=2,label='SFR_HC')
    #if rlogscale:
    ax.set_xscale('log')
    #if np.sum(annuli.sfrs) == 0 and np.sum(annuli.sfrs2) == 0:
    #    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel(r'SFR(<R) $M_\odot {\rm yr}^{-1}$')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotSfLaw(annuli, axIn=None, basename="sfLaw",filetype="png",rlogscale=True):
    fig,ax=axisSwitch(axIn)
    colPerT = np.sort((annuli.colGas/(annuli.tff2)).to('Msun/yr/pc**2'))
    ax.plot(colPerT, colPerT*0.01,ls='--',lw=2, label='0.01 efficiency per tff')
    ax.scatter((annuli.colGas/(annuli.tff2)).to('Msun/yr/pc**2'), annuli.colSFRs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Sigma / t_{ff}$ ($M_\odot {\rm yr}^{-1} {\rm pc}^{-2}$)')
    ax.set_ylabel(r'$\dot{\Sigma}_*$ ($M_\odot {\rm yr}^{-1} {\rm pc}^{-2}$)')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotSfGasLaw(annuli, axIn=None, basename="sfGasLaw", filetype="png",rlogscale=True):
    fig,ax=axisSwitch(axIn)
    colGas = np.sort(annuli.colGas)
    KS = 2.5e-10 * np.power(colGas,1.4) # Kennicutt 1998 equation 4.
    ax.plot(colGas,KS,ls='--',lw=2,color='gray',label='Kennicutt 98')
    ax.plot(colGas,KS*1.8/2.5,ls='--',lw=1,color='gray')
    ax.plot(colGas,KS*3.2/2.5,ls='--',lw=1,color='gray')
    ax.scatter(annuli.colGas,annuli.colSFRs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Sigma$ ($M_\odot {\rm pc}^{-2}$)')
    ax.set_ylabel(r'$\dot{\Sigma}_*$ ($M_\odot {\rm yr}^{-1} {\rm pc}^{-2}$)')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotTimescales(annuli, axIn=None, basename="timescales", filetype="png",rlogscale=True):
    fig,ax=axisSwitch(axIn)
    ax.plot(annuli.r,(annuli.tff).to('yr'),ls='-',lw=2,label='from mean density')
    ax.scatter(annuli.r,(annuli.tff).to('yr'))
    ax.plot(annuli.r,(annuli.tff2).to('yr'),ls='-',lw=2,label='from median density')
    ax.scatter(annuli.r,(annuli.tff2).to('yr'))
    ax.plot(annuli.r,(annuli.tff3).to('yr'),ls='-',lw=2,label=r'$h/c_s$')
    ax.scatter(annuli.r,(annuli.tff3).to('yr'))
    ax.plot(annuli.r,annuli.tOrb,ls='-',lw=2,label='orbital time')
    ax.scatter(annuli.r,annuli.tOrb)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('dynamical times (yrs)')
    ax.legend()
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig, ax, basename, filetype, annuli.pfName,annuli.pfDir)
    
def plotColumnDensities(annuli, axIn=None, basename="sigma", filetype="png",rlogscale=True):
    colTot = annuli.colGas+annuli.colSt+annuli.colDM

    fig,ax=axisSwitch(axIn)
    ax.scatter(annuli.r,(annuli.colGas).to('g/cm**2'),color='b')
    ax.plot(annuli.rFinest,(annuli.colGas2).to('g/cm**2'),color='m',marker="D",label=r'$\rho \Delta x$')
    #ax.plot(annuli.rFinest,(annuli.colGas3).to('g/cm**2'),color='b',marker="x",label=r'$ M/ \Delta x^2$')
    ax.scatter(annuli.r,(annuli.colSt).to('g/cm**2'),color='g')
    ax.scatter(annuli.r,(annuli.colDM).to('g/cm**2'),color='r')
    ax.scatter(annuli.r,(annuli.col).to('g/cm**2'),color='k')
    ax.scatter(annuli.r,(colTot).to('g/cm**2'),color='k')
    ax.plot(annuli.r,(annuli.colGas).to('g/cm**2'),color='b',label='gas')
    ax.plot(annuli.r,(annuli.colSt).to('g/cm**2'),color='g',label='stars')
    ax.plot(annuli.r,(annuli.colDM).to('g/cm**2'),color='r',label='DM')
    ax.plot(annuli.r,(annuli.col).to('g/cm**2'),color='k',label='total')
    ax.plot(annuli.r,(colTot).to('g/cm**2'),color='k',ls='--',label='added',lw=3)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel(r'Surface Density $\Sigma$ (g/cm$^2$)')
    #if rlogscale:
    ax.set_xscale('log')
    return returnSwitch(fig, ax, basename, filetype, annuli.pfName,annuli.pfDir)

def plotSphericalMasses(annuli, axIn=None, basename="spheres", filetype="png",rlogscale=True):
    fig,ax=axisSwitch(axIn)
    r = annuli.r 
    spTotalMass = annuli.spMassEnclosed 
    spDM = annuli.spMassDM 
    spSt = annuli.spMassSt 
    spGas = annuli.spMassGas
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
    #if rlogscale:
    ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotVelocityDispersions(annuli, axIn=None, basename="dispersions", filetype="png", rlogscale=True):
    fig,ax=axisSwitch(axIn)

    r = annuli.r 
    sigR = annuli.sigR 
    sigPhi = annuli.sigPhi 
    sigZ = annuli.sigZ 
    sigth = annuli.sigTh 
    # Add in the thermal component to each direction (HC: why?)
    sigR = np.sqrt(np.power(sigR,2.0))#+np.power(sigth,2.0))
    sigPhi = np.sqrt(np.power(sigPhi,2.0))#+np.power(sigth,2.0))
    sigZ = np.sqrt(np.power(sigZ,2.0))#+np.power(sigth,2.0))

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
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotBulkVelocity(annuli, axIn=None, basename="bulk", filetype="png",rlogscale=True):
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
    if rlogscale:
        ax.set_xscale('log')
    ax.set_yscale('symlog')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotRotationCurve(annuli, axIn=None, basename="rotation", filetype="png",rlogscale=True):
    fig,ax=axisSwitch(axIn)
        
    r = annuli.r 
    vPhi = annuli.vPhi 
    vStPhi = annuli.vStPhi 
    spVcirc = annuli.spVcirc 
    spVcirc2 = annuli.spVcirc2
    
    ax.scatter(r,vPhi*1.0e-5,color='b')
    ax.scatter(r,spVcirc*1.0e-5,color='r')
    ax.plot(r,vPhi*1.0e-5,label='Average in annulus (gas)',color='b')
    ax.scatter(r,vStPhi*1.0e-5,color='b')
    ax.plot(r,vStPhi*1.0e-5,label='Average in annulus (stars)',color='b',ls='--')
    ax.plot(r,spVcirc*1.0e-5,label='From total mass enclosed',color='r')
    ax.plot(annuli.rFinest,(spVcirc2).to('cm/s')*1.0e-5,color='m',marker="D",\
            label=r'$\frac{\Delta x}{2}\sqrt{8G\rho + \frac{GM_{BH}}{(\Delta x/2)^3}}$')
    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Rotational Velocity (km/s)')
    if len(spVcirc2) == 0:
        np.save('./'+annuli.pfDir+'/'+annuli.pfName+'_spVcirc.npy',[r,spVcirc])
    else:
        np.save('./'+annuli.pfDir+'/'+annuli.pfName+'_spVcirc.npy',[r,spVcirc,np.ones(np.shape(spVcirc))*spVcirc2])
    if rlogscale:
        ax.set_xscale('log')
    ax.set_yscale('symlog')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotStability(annuli, axIn=None, basename="toomre", filetype="png",rlogscale=True):
    fig,ax=axisSwitch(axIn)
    r = annuli.r[1:] 
    QRW = annuli.QRW[1:] 
    Qgas = annuli.Qgas[1:] 
    Qst = annuli.Qst[1:] 
    QWS = annuli.QWS[1:]
    QHC = annuli.QHC[1:]
    QHC2 = annuli.QHC2[1:]

    #ax.scatter(r,QWS,color='b')
    #ax.plot(r,QWS,color='b',label=r'Wang-Silk')
    ax.scatter(r,QRW,color='k')
    ax.plot(r,QRW,color='k',label=r'Romeo-Wiegert')
    ax.scatter(r,Qgas,color='r')
    ax.plot(r,Qgas,color='r',label=r'Gas $\frac{\sqrt{c_s^2+\sigma_R^2}\kappa}{\pi G \Sigma}$')
    ax.scatter(r,Qst,color='g')
    ax.plot(r,Qst,color='g',ls='--',label=r'Stars')
    ax.scatter(r,QHC,color='m')
    ax.plot(r,QHC,color='m',ls='--',label=r'Gas $\frac{c_s\Omega}{\pi G \Sigma}$')
    #ax.scatter(r,QHC2,color='y')
    #ax.plot(r,QHC2,color='y',ls='--',label=r'Gas $\frac{\sigma_R \kappa}{\pi G \Sigma}$')
    ax.plot(r,r*0+1.0,label=r'Approx Marginal Stability',color='b',ls='--')
    #ax.plot(r,r*0+2.0,label=r'Approx Marginal Stability w/ Finite Thickness',color='k',ls='--')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Q')
    ax.set_yscale('log')
    if rlogscale:
        ax.set_xscale('log')
    ax.legend()
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotFluxes(annuli, axIn=None, basename="verticalFlux", filetype="png",rlogscale=True):
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
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotScaleheightRatio(annuli, axIn=None, basename="scaleheightRatio", filetype="png",rlogscale=True):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, (annuli.hGas/annuli.r).to(''), color='k')
    ax.scatter(annuli.r, (annuli.hGas/annuli.r).to(''), color='k')
    #ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('h/r')
    if rlogscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)


def plotScaleheights(annuli, axIn=None, basename="scaleheights", filetype="png",rlogscale=True):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.hGas, color='blue', label='gas')
    ax.scatter(annuli.r, annuli.hGas, color='blue')
    #ax.plot(annuli.r, annuli.hGas2, color='pink', label=r'gas $\frac{\sigma_R^2}{\pi G \Sigma}$')
    #ax.scatter(annuli.r, annuli.hGas2, color='pink')
    ax.plot(annuli.r, annuli.hStars, color='red', label='stars')
    ax.scatter(annuli.r, annuli.hStars, color='red')

    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Scaleheight (kpc)')
    if rlogscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotNResolutionElements(annuli, axIn=None, basename="nResElements", filetype="png",rlogscale=True):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.Ngas, color='blue', label='gas')
    ax.scatter(annuli.r, annuli.Ngas, color='blue')
    ax.plot(annuli.r, annuli.Nstars, color='red', label='stars')
    ax.scatter(annuli.r, annuli.Nstars, color='red')

    ax.legend()
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Number of Resolution Elements')
    ax.set_yscale('log')
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotMetallicity(annuli, axIn=None, basename='metallicity', filetype='png',rlogscale=True):
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
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotStellarAges(annuli, axIn=None, basename='stellarAges', filetype='png',rlogscale=True):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.stAge/1e6, color='blue', label='gas')
    ax.plot(annuli.r, (annuli.stAge+annuli.stAgeSpread)/1e6, color='blue', ls='--')
    ax.plot(annuli.r, (annuli.stAge-annuli.stAgeSpread)/1e6, color='blue', ls='--')
    ax.scatter(annuli.r, annuli.stAge/1e6, color='blue')

    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Stellar Ages (Myr)')
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotDensityDistribution(annuli, axIn=None, basename='densDist', filetype='png',rlogscale=True):
    fig,ax = axisSwitch(axIn)
    ax.plot(annuli.r, annuli.avgLogDensity, color='blue', label='gas')
    ax.plot(annuli.r, annuli.avgLogDensity+annuli.logDensityWidth, color='blue', ls='--')
    ax.plot(annuli.r, annuli.avgLogDensity-annuli.logDensityWidth, color='blue', ls='--')
    ax.scatter(annuli.r, annuli.avgLogDensity, color='blue')

    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Log Density (g/cc)')
    if rlogscale:
        ax.set_xscale('log')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

def plotInflowRate(annuli_next, annuli, interval, axIn=None, basename='inflowRate', filetype='png',rlogscale=True):
    fig,ax = axisSwitch(axIn)
    
    if interval is not None:
        cylMassEnclosed_next = annuli_next.cylMassEnclosedTallDisks
        cylMassEnclosed = annuli.cylMassEnclosedTallDisks
        inflowRate = ((cylMassEnclosed_next-cylMassEnclosed)/interval).to('Msun/yr')
    else:
        inflowRate = annuli.cylOutFluxes
    np.save('./'+annuli.pfDir+'/'+annuli.pfName+'_inflowRate.npy',[annuli.r,inflowRate])
    ax.plot(annuli.r, inflowRate, color='k')
    ax.scatter(annuli.r, inflowRate, color='k')

    ax.set_xlabel('r (kpc)')
    ax.set_ylabel(r'dM/dt ($M_\odot {\rm yr}^{-1}$)')
    if rlogscale:
        ax.set_xscale('log')
        ax.set_yscale('symlog')
    return returnSwitch(fig,ax,basename,filetype,annuli.pfName,annuli.pfDir)

# A few convenience functions for quick looks at the results.
#plotFunctions = [plotStability, plotRotationCurve, plotBulkVelocity, 
#                 plotVelocityDispersions, plotSphericalMasses,
#                 plotColumnDensities, plotTimescales, plotSfGasLaw, 
#                 plotSfLaw, plotFluxes, plotScaleheights, plotNResolutionElements,
#                 plotMetallicity, plotStellarAges, plotDensityDistribution ]
plotFunctions = [plotStability, plotRotationCurve, plotBulkVelocity, 
                 plotVelocityDispersions, plotSphericalMasses,
                 plotColumnDensities, plotTimescales,
                 plotScaleheights, plotNResolutionElements,
                 plotMetallicity, plotStellarAges, plotDensityDistribution,
                 plotSf, plotSfGasLaw, plotSfLaw, plotScaleheightRatio ]
def showAllPlots(annuli):
    for fn in plotFunctions:
        fn(annuli,filetype='screen')
def saveAllPlots(annuli,rlogscale=True):
    for fn in plotFunctions:
        fn(annuli,filetype='png',rlogscale=rlogscale)   


if __name__ == '__main__':

    # A use case

    #pf = load('/pfs/jforbes/run-enzo-dev-jforbes/dwarf_nf35/DD0172/DD0172')
    #pf = load('/data/hcho/MyAgoraGalaxy/run_200130_640pc/DD0010/DD0010')
    #pf_now = load('/data/hcho/HighAgora/run_200528_lv16_test/DD16_0008/DD0008')
    #pf_now = load('/data/hcho/HighAgora/run_200520_lv12norad/DD0061/DD0061')
    #pf_now = load('/data/hcho/HighAgora/run_200526_lv15local/DD0007/DD0007')
    #pf_now = load('/data/hcho/HighAgora/run_200604_lv17alpha/DD0003/DD0003')
    pf_now = load('/data/hcho/lv17HighAgora/run_200610_stdlv09/DD0148/DD0148') #DD0066/DD0066')
    fluxHeights = [0.1] # [50.] #[1.0/1000.0 * 8**i for i in range(1)]#(4)] # 1, 8, 64, 512 pc

    # determine BH position
    if 1:
        ad = pf_now.all_data()
        type_= np.array(ad['particle_type'])
        bhpos = []
        for pos in ['x','y','z']:
            bhpos +=\
            [((ad['particle_position_'+pos][np.where(type_==8)[0]][0]).to('code_length')).d]
        center = bhpos
    else:
        center = [0.5,0.5,0.5]

    # Look at annuli between 30 pc and 600 pc
    # Use 20 annuli
    # Up to a height of 100 pc
    # Centered at [.5,.5,.5] in code units
    # Look at vertical fluxes at the heights allocated above.
    inner = 0.15 # 0.001 #0.15 #0.0005 #0.002 #0.01 #
    outer = 20. # 35. #
    Nsteps = 20
    height = 4

    #annuli = annularAnalysis(.03, 0.6, 20, 0.1, [.5,.5,.5],fluxHeights) # set up
    #annuli = annularAnalysis(1.2, 35., 20, 4., [.5,.5,.5],fluxHeights) # set up
    #annuli = annularAnalysis(0.001, 35, 20, 4., bhpos,fluxHeights) # set up
    #annuli = annularAnalysis(0.01, 35., 20, 4, bhpos,fluxHeights) # set up
    annuli = annularAnalysis(inner, outer, Nsteps, height, center,fluxHeights) # set up
    rlogscale = False
    annuli.run(pf_now,rlogscale=rlogscale) # actually run the analysis

    # temporary
    np.save('./'+annuli.pfDir+'/'+annuli.pfName+'_upDownFlux.npy',[annuli.r,annuli.cylUpFluxes[:,0],annuli.cylDownFluxes[:,0]])
    #plotStability(annuli,filetype='png',rlogscale=rlogscale)

    saveAllPlots(annuli,False)#rlogscale) # save the plots as <pfName>_<plotName>.png

    if 0:
        # do the same thing with the output from one timestep after
        replace_str = '{:02d}'.format(int(pf_now.basename[-2:])+1)
        splitted = pf_now.directory.split('/')
        splitted2 = splitted[:-1]+[splitted[-1].replace(splitted[-1][-2:],replace_str),pf_now.basename.replace(pf_now.basename[-2:],replace_str)]
        pf_next = load('/'.join(splitted2))
    
        annuli_next = annularAnalysis(inner, outer, Nsteps, height, bhpos,fluxHeights) # set up
        annuli_next.run(pf_next,rlogscale=rlogscale) # actually run the analysis

        interval = (pf_next.current_time-pf_now.current_time).to('Myr')
    else:
        annuli_next = None
        interval = None

    plotInflowRate(annuli_next,annuli, interval, filetype='png',rlogscale=False)#rlogscale)
