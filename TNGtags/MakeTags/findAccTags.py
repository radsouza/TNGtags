import illustris_python as il
import numpy as np
import h5py
from groupcat import gcPath, offsetPath


def loadIDsSubhalos(SubfindID,Snap0,dataPath):
    """
    For a given snapshot return the particleIDs of a given Subhalo
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
    For a given snapshot return the particleIDs of a given Group
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
    