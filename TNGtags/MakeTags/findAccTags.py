import illustris_python as il
import numpy as np
import h5py
from groupcat import gcPath, offsetPath


def loadIDsSubhalos(SubfindID,Snap0,dataPath):
    """
    For a given snapshot return the particleIDs of a given Subhalo or Subhalos
    """
    file1=dataPath+'data_'+str(Snap0)+'.hdf5'
    with h5py.File(file1,'r') as f1:
        if isinstance(SubfindID, (list, tuple, np.ndarray)) and len(SubfindID)>0:
            minSubfindIDs=np.min(SubfindID)
            maxSubfindIDs=np.max(SubfindID)
            dlen=f1['ids'].len()
            mask=np.zeros(dlen,dtype=np.bool)
            sStart_all=f1['SubhaloStart'][minSubfindIDs:maxSubfindIDs+1]
            sLen_all=f1['SubhaloLen'][minSubfindIDs:maxSubfindIDs+1]
            sStart=sStart_all[SubfindID-minSubfindIDs]
            sLen=sLen_all[SubfindID-minSubfindIDs]
            for i,j  in zip(sStart,sLen):
                if i !=-1 and j !=-1:
                    mask[i:i+j]=True
            particleIDs1=f1['ids'][mask]
            return particleIDs1
        if isinstance(SubfindID,int):
            sStart=f1['SubhaloStart'][SubfindID]
            sLen=f1['SubhaloLen'][SubfindID]
            if sStart !=-1 and sLen !=-1:
                particleIDs1=f1['ids'][sStart:sStart+sLen]
                return particleIDs1
            else:
                return np.array([])
    return np.array([])


def loadIDsGroups(GroupID,Snap0,dataPath):
    """
    For a given snapshot return the particleIDs of a given Group or Groups
    """
    file1=dataPath+'data_'+str(Snap0)+'.hdf5'
    with h5py.File(file1,'r') as f1:
        if isinstance(GroupID, (list, tuple, np.ndarray))and len(GroupID)>0:
            minGroupIDs=np.min(GroupID)
            maxGroupIDs=np.max(GroupID)
            dlen=f1['ids'].len()
            mask=np.zeros(dlen,dtype=np.bool)
            gStart_all=f1['GroupStart'][minGroupIDs:maxGroupIDs+1]
            gLen_all=f1['GroupLen'][minGroupIDs:maxGroupIDs+1]
            gStart=gStart_all[GroupID-minGroupIDs]
            gLen=gLen_all[GroupID-minGroupIDs]
            for i,j  in zip(gStart,gLen):
                if i !=-1 and j !=-1:
                    mask[i:i+j]=True
            particleIDs1=f1['ids'][mask]
            return particleIDs1
        if isinstance(GroupID,int):   
            gStart = f1['GroupStart'][GroupID]
            gLen   = f1['GroupLen'][GroupID]
            if gStart !=-1 and gLen !=-1:
                particleIDs1=f1['ids'][gStart:gStart+gLen]
                return particleIDs1
            else:
                return np.array([])
    return np.array([])


def getMergerTrees(subfindID0, basePath,snapNum0):
    """
    Returns a tuple containing 
    a)  a tree (with a number of essential fields)
    b)  the indices within the tree containing the mpb
    c)  the indices within the tree containing the galaxies belonging to the fof halo of the mpb, but are not on the mpb. 
    d)  the indices within the tree containing the galaxies outside the fof halo of the mpb.
    """
    fields=['SubhaloID','NextProgenitorID','MainLeafProgenitorID',
            'FirstProgenitorID','LastProgenitorID','SubhaloMassType',
            'SnapNum','SubfindID','FirstSubhaloInFOFGroupID','SubhaloGrNr']

    tree = il.sublink.loadTree(basePath,snapNum0,subfindID0,fields=fields,onlyMPB=False)
    
    mpb=np.where(tree['SubhaloID'][:]<=tree['MainLeafProgenitorID'][0])[0]    # main progenitor branch
    epb=np.where(tree['SubhaloID'][:]> tree['MainLeafProgenitorID'][0])[0]    # external progenitor branch
    
    mask_epb_fof=np.zeros_like(epb,dtype=np.bool)

    # Distinguish between FoF and External halos
    for Snap0 in np.unique(tree['SnapNum'][mpb]):
        # mpb
        arg0=np.where(tree['SnapNum'][mpb]==Snap0)[0]
        mpb_subhaloID=tree['SubhaloID'][arg0]
        # epb
        arg1=np.where(tree['SnapNum'][epb]==Snap0)[0]
        pointerIDs=tree['FirstSubhaloInFOFGroupID'][epb[arg1]]
        
        arg_fof=np.where(pointerIDs==mpb_subhaloID)[0]
        arg_mask=np.where(epb[arg1][arg_fof])[0]
        mask_epb_fof[arg1[arg_fof][arg_mask]]=True
        
    return (tree, mpb, epb[mask_epb_fof], epb[~mask_epb_fof])


def create_tags(particleIDs, subfindID0, snapNum0, basePath, dataPath):
    """
    Return 3 masks of the particleIDs indicating where the particles were born: mpb, fof and ext.
    
    The algorithm consists of characterizing the birth place of the stellar particles into 3 zones (mpb, fof and ext).
    We scan the snapshots going forward in time, determining in which zone the stellar particle was formed.
    
    To speed up calculations, we make effective use of: 
    a) boolean masks for each stellar particles.  (We currenlty use 4 masks, 3 for birth in each zone, and the 4th to store previously born stellar particles.).
    b) numpy in1d function in unqiue mode to reduce everything to masks. 
    c) Element wise boolean comparisons between masks. We detect new stellar particles using the **greater** operatior, 
    and we add up stellar particles using **logical_or** operator.

    Presently the algorithm is implemented using Subhalos and the merger trees. 
    It can be modified to use Groups (FoF haloes), or incorporate other rules to determine  
    """
    
    
    # get merger tree of galaxy
    tree, mpb, fof_branch, ext_branch =  getMergerTrees(subfindID0, basePath, snapNum0)
    
    # Mask for storing and registering where the particle was born.
    particleIDs_mpb=np.zeros_like(particleIDs,dtype=np.bool)
    particleIDs_fof=np.zeros_like(particleIDs,dtype=np.bool)
    particleIDs_ext=np.zeros_like(particleIDs,dtype=np.bool)

    # Mask for stellar particles already born.
    particleIDs_pre=np.zeros_like(particleIDs,dtype=np.bool)
    
    # The present algorithm only considers Subhaloes. More complicated rules can be constructed.

    for Snap0 in np.unique(tree['SnapNum'][mpb]): 
        if Snap0>4:
            # MPB
            arg0=np.where(tree['SnapNum'][mpb]==Snap0)[0]
            SubfindID=tree['SubfindID'][mpb][arg0]
            particleIDs0=loadIDsSubhalos(SubfindID,Snap0,dataPath)
            if len(particleIDs0)>0:
                particleIDs0_present=np.in1d(particleIDs,particleIDs0, assume_unique=True)
                particleIDs_mpb=np.logical_or(particleIDs_mpb,np.greater(particleIDs0_present,particleIDs_pre))

            # FOF
            arg1=np.where(tree['SnapNum'][fof_branch]==Snap0)[0]
            SubfindIDs=tree['SubfindID'][fof_branch][arg1]
            particleIDs1=loadIDsSubhalos(SubfindIDs,Snap0,dataPath)
            if len(particleIDs1)>0:
                particleIDs1_present=np.in1d(particleIDs,particleIDs1, assume_unique=True)
                particleIDs_fof=np.logical_or(particleIDs_fof,np.greater(particleIDs1_present,particleIDs_pre))

            # EXT
            arg2=np.where(tree['SnapNum'][ext_branch]==Snap0)[0]
            SubfindIDs=tree['SubfindID'][ext_branch][arg2]
            particleIDs2=loadIDsSubhalos(SubfindIDs,Snap0,dataPath)
            if len(particleIDs2)>0:
                particleIDs2_present=np.in1d(particleIDs,particleIDs2, assume_unique=True)
                particleIDs_ext=np.logical_or(particleIDs_ext,np.greater(particleIDs2_present,particleIDs_pre))


            # store if particle already exists.
            particleIDs_pre=np.logical_or(particleIDs_pre, particleIDs0_present)
            particleIDs_pre=np.logical_or(particleIDs_pre, particleIDs1_present)
            particleIDs_pre=np.logical_or(particleIDs_pre, particleIDs2_present)
            
    return (particleIDs_mpb, particleIDs_fof, particleIDs_ext)
        
    