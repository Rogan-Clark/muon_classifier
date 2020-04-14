import os.path
from os import path
from muon_writer_clean import *            #Change here to switch from applying tailcut or not
    
#Takes simtelarray data, shuffles through to find the next possible sequential set,
#them runs the muon_writer script on that grouping to output a hdf5 file for use with our CNN

muon_no=1
proton_no=1
partial_no=1
runno=1

while partial_no+muon_no+proton_no<301: 
    exists_muon=path.exists('/store/muonsims/muons3/run'+str(muon_no)+'.simtel.gz')
    if exists_muon==True:
        
        exists_proton=path.exists('/store/muonsims/proton6/run'+str(proton_no)+'.simtel.gz')
        if exists_proton==True:
            
            exists_partial=path.exists('/store/muonsims/partial3/run'+str(partial_no)+'.simtel.gz')
            if exists_partial==True:
                
                cleanmuonwriter(muon_no,proton_no,partial_no,runno)
                print('Exists files of muon number '+str(muon_no)+' proton number '+str(proton_no)+' partial number '+str(partial_no)+' with runno '+str(runno))
                muon_no+=1
                proton_no+=1
                partial_no+=1
                runno+=1

            else:
                print('No Partials with runno '+str(partial_no))            
                partial_no+=1

        else:
            print('No Protons with runno '+str(proton_no))
            proton_no+=1
    else:
        print('No Muons with runno '+str(muon_no))
        muon_no+=1

print('Processing Complete')

