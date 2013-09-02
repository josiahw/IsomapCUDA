from subprocess import Popen
import shlex
import os
from DataUtils import loadMatrix,saveTable

from sklearn import manifold, datasets
from numpy import array,loadtxt,savetxt

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

k = 15
i = 0
for f in os.listdir('/media/Merlz/dimensionality_reduction/twists'):
    
    if f.endswith('.csv') and not f in os.listdir('/media/Merlz/dimensionality_reduction/Isomap'):
        X = loadtxt('/media/Merlz/dimensionality_reduction/twists/'+f,delimiter=',').T[:,:3]
        savetxt('current.csv',X,delimiter=',')
        os.system('optirun python Isomap.py --if=current.csv --of=/media/Merlz/dimensionality_reduction/Isomap/'+f+' --indims=3 --outdims=3 --k=15')
        """
        m_lle1 = manifold.LocallyLinearEmbedding(k, 3,
                                            eigen_solver='auto',
                                            method='standard')
        m_lle2 = manifold.LocallyLinearEmbedding(k, 2,
                                            eigen_solver='auto',
                                            method='standard')
        
        print len(X)
        
        Y = m_lle2.fit_transform(m_lle1.fit_transform(m_lle1.fit_transform(X)))
        saveTable(Y,'LLE_3iter/'+f)
        del m_lle1
        del m_lle2
        """
        """
        Y = manifold.LocallyLinearEmbedding(k, 2,
                                        eigen_solver='dense',
                                        method='hessian').fit_transform(X)
        saveTable(Y,'HLLE/'+f)
        """
        """
        Y = manifold.LocallyLinearEmbedding(k, 2,
                                        eigen_solver='dense',
                                        method='modified').fit_transform(X)
        saveTable(Y,'MLLE/'+f)
        """
        """
        Y = manifold.LocallyLinearEmbedding(k, 2,
                                        eigen_solver='dense',
                                        method='ltsa').fit_transform(X)
        saveTable(Y,'LTSA/'+f)
        """
    print "processed ",i
    i += 1
