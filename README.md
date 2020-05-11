# TNGtags

version 0.1: 11/May/2020

TNGtags is divided into three sections:
- GenerateIDFiles
- MakeTags
- ReadTutorial


Step 1: Run the two files in GenerateIDFiles, this will create the ID Files for Illustris-1 and TNG-100 in two separate directories called 'Particles-Illustris-1' and 'Particles-TNG-100'.  Each will take approximately 2-3 hours to run. Can be run simultaneously, with both open in separate jupyter notebooks.

Step 2: Look at MakeTags, which contains the essence of the code to make insitu, communter or accreted tags. This code can be modified as necessary. Note that at the present moment, the tags are generated on the fly, its takes less than a minute of a single MW-mass galaxy. At a later stage, once a certain algorithm is chosen, the tags as 'masks' to the 'particleIDs' can be saved.

Step 3: A tutorial to extract simulation data for MW-mass galaxies is found in ReadTutorial. This particular notebook demonstrates how to choose MW-mass galaxies, how to read the snapshot data, and how to generate the insitu/accreted tags by calling the code in MakeTags.







