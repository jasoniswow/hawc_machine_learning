# coding: utf-8
import numpy as np
import xcdf
import pandas as pd

infile = xcdf.XCDFFile("data/sweets_dec19_noise_MPF_allParticle.xcd")

# selected features
fields = {
           'MC delCore'           : 'mc.delCore',
           'MC delAngle'          : 'mc.delAngle',
           'MC energy logGeV'     : 'mc.logEnergy',
           'corsikaParticleId'    : 'mc.corsikaParticleId',
           'sweets TWgt'          : 'sweets.TWgt',
           'tank with hit'        : 'rec.nTankHit',
           'nChAvail'             : 'rec.nChAvail',
           'nCh total'            : 'rec.nChTot',
           'SFCFChi2'             : 'rec.SFCFChi2',
           'planeChi2'            : 'rec.planeChi2',
           'CoreFit pos err'      : 'rec.coreFitUnc',
           'zenithAngle'          : 'rec.zenithAngle',
           'azimuthAngle'         : 'rec.azimuthAngle',
           'coreFitStatus'        : 'rec.coreFitStatus',
           'angleFitStatus'       : 'rec.angleFitStatus',
           'reco coreFiduScale'   : 'rec.coreFiduScale',
           'nHitSP10'             : 'rec.nHitSP10',
           'nHitSP20'             : 'rec.nHitSP20',
           'CxPE20'               : 'rec.CxPE20',
           'CxPE30'               : 'rec.CxPE30',
           'CxPE40'               : 'rec.CxPE40',
           'CxPE50'               : 'rec.CxPE50',
           'CxPE40SPTime'         : 'rec.CxPE40SPTime',
           'PINC'                 : 'rec.PINC',
           'GamCoreAge'           : 'rec.GamCoreAge',
           'GamCore params pack1' : 'rec.GamCoreChi2',
           'GamCore params pack2' : 'rec.GamCorePackInt',
           'dominant plane nHit'  : 'rec.nHit',
           'total nHit'           : 'rec.mPFnHits',
           'number of planes'     : 'rec.mPFnPlanes',
           '2nd plane nHit'       : 'rec.mPFp1nAssign',
           'summed charge 0'      : 'rec.fAnnulusCharge0',
           'summed charge 1'      : 'rec.fAnnulusCharge1',
           'summed charge 2'      : 'rec.fAnnulusCharge2',
           'summed charge 3'      : 'rec.fAnnulusCharge3',
           'summed charge 4'      : 'rec.fAnnulusCharge4',
           'summed charge 5'      : 'rec.fAnnulusCharge5',
           'summed charge 6'      : 'rec.fAnnulusCharge6',
           'summed charge 7'      : 'rec.fAnnulusCharge7',
           'summed charge 8'      : 'rec.fAnnulusCharge8',
           'max dis'              : 'rec.disMax',
           'logNNEnergy'          : 'rec.logNNEnergy'
          }


#header:
head = infile.header()
#fields:
headerf = [f.split() for f in head.split("\n")]
mheaderf = map(lambda h: h[0], filter(lambda x: len(x)>1, headerf[2:]))

indices = {}
values = []
for i,h in enumerate(mheaderf):
    for f in fields.iteritems():
        if (h==f[1]): 
            indices[h] = i
            values.append([])
print( indices )

c = infile.count
percent = 100. #00. #0.0001
n = 0
while (n<(percent/100.*c)):
    r = infile.getRecord(n)
    for k,i in enumerate(indices.iteritems()):
        values[k].append(r[i[1]])
    n += 1
print( "Read %d records."%n )
p = pd.DataFrame()
for k,i in enumerate(indices):
    p[i.split(".")[1]] = np.array(values[k]) 

print( p.head() )

hdf = pd.HDFStore('data/new_sweets_dec19_noise_MPF_allParticle_%d.h5'%(int(percent)))
hdf.put('sweets_xcdf', p, format='table', data_columns=True)
hdf.close()
