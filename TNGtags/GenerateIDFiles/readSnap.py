import illustris_python as il
import numpy as np
import h5py
from groupcat import gcPath, offsetPath

def create_particleID_file(basePath,snapNum,writeFile):
    """
    For a given simulation selected using basePath, 
    read the stellar particleIDs of a particular snapshot,
    read its offset and len values of the groups and subhalos
    and writes to a file writeFile.
    
    Notes:
    - Can deal with both Illustris and TNG formats, where the offset files are stored differently.
    - f['SubhaloLen'] and f['SubhaloStart'] are both arrays of length equal to the number of subhalos, 
      where the index of the array is the Subhalo Number. This sort of sparse matrix is used to save space.
      Similar considerations for GroupStart and GroupLen.
    """
    
    # select only stellar particles
    pType=4

    # Open file for writing
    f3=h5py.File(writeFile,'w')

    # Read the ParticleIDs of all stellar particles
    star_ids = il.snapshot.loadSubset(basePath,snapNum,'stars',['ParticleIDs'])
    f3['ids']=star_ids
    lstar_ids=len(star_ids)
    star_ids=''


    #find offsets for the Subhalo and the Groups of these stellar particles
    with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
        offset_subhalo  = f['Subhalo']['SnapByType'][:,pType]
        offset_group    = f['Group']['SnapByType'][:,pType]
    
    #find group and subhalo lens
    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/'+'Group'][()]
            subhaloFileOffsets = f['FileOffsets/'+'Subhalo'][()]
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_'+'Group']
            subhaloFileOffsets =f['Header'].attrs['FileOffsets_'+'Subhalo']


    len_group=np.zeros_like(offset_group,dtype='int32')
    len_subhalo=np.zeros_like(offset_subhalo,dtype='int32')

    for i in range(len(groupFileOffsets)):
        with h5py.File(gcPath(basePath, snapNum, i), 'r') as f:
            try:
                a1=f['Group']['GroupLenType'][:, pType]
            except:
                a1=np.array([])
            try:
                b1=f['Subhalo']['SubhaloLenType'][:,pType]
            except:
                b1=np.array([])
        len_group[groupFileOffsets[i]:groupFileOffsets[i]+len(a1)]=a1
        len_subhalo[subhaloFileOffsets[i]:subhaloFileOffsets[i]+len(b1)]=b1

    f3['GroupStart']=offset_group
    f3['GroupLen']=len_group
    f3['SubhaloStart']=offset_subhalo
    f3['SubhaloLen']=len_subhalo
    f3.close()