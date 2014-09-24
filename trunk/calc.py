''' Calc module contains useful mathematical methods used throughout ledsim.
'''
from ledsim import *

def alternate(a,b):
  ''' Alternate the elements of vectors a and b. The result vector's first 
      element is the first element from a; the second element is the second 
      element from b; the third element is the third from a, etc.
  '''
  num = len(a)
  ind = scipy.arange(0,num)
  result = scipy.zeros(num)
  result[ind%2 == 0] = a[ind%2 == 0]
  result[ind%2 == 1] = b[ind%2 == 1]
  return result

def stretch(a,step):
  ''' Return a 'streteched' version of a, i.e. [1,2,3] -> [1,1,2,2,3,3] for
      step = 2.
  '''
  return a[scipy.sort(range(len(a))*step)]

def sort(a,order='descend'):
  ''' Sort the given vector in either ascending or descending order; return
      both the sorted vector and the indices.
  '''
  if order not in ['ascend','descend']:
    raise ValueError, 'Specified ordering was not understood!'
  ind = scipy.argsort(a)
  if order == 'descend':
    ind = ind[::-1]
  return a[ind], ind

def eq_tol(a,b,tol=1e-6):
  ''' Check whether a and b are equal within the specified tolerance.
  '''
  return abs(0.5*(a-b)/(a+b)) < tol

def neq_tol(a,b,tol=1e-6):
  ''' Check whether a and b are unequal allowing for the specified tolerance.
  '''
  return not eq_tol(a,b,tol)

def leq_tol(a,b,tol=1e-6):
  ''' Check whether a is less or equal to b allowing for the specified tolerance.
  '''
  return (a < b) + eq_tol(a,b,tol)

def geq_tol(a,b,tol=1e-6):
  ''' Check whether a is greater or equal to b allowing for the specified tolerance.
  '''
  return (a > b) + eq_tol(a,b,tol)

def merge_eigs(eigsDict,tol=1e-6):
  ''' Merge the eigenvectors/eigenvalues which are the items of the dictionary 
      eigsDict. Duplicate eigenvectors/eigenvalues are discarded. Eigenvectors
      a and b are considered equal if eq_tol(a'.b,a'.a,tol)
  '''
  def merge(w1,v1,w2,v2):
    newInd = scipy.array([scipy.prod(abs((w2[ii]-w1)/w2[ii]) > tol) > 0 for ii in range(len(w2))])
    w,ind  = sort(scipy.concatenate((w1,w2[newInd])),'ascend')
    v = scipy.hstack((v1,v2[:,newInd]))[:,ind]
    return w,v
  guesses = scipy.sort(eigsDict.keys())
  w,v = eigsDict[guesses[0]]
  for guess in guesses[1:]:
    w2,v2 = eigsDict[guess]
    w,v = merge(w,v,w2,v2)
  return w,v
 
def eigs_sorted(mat,sigma,k,isTime=False,isHermitian=True):
  ''' Get k eigenvalues for the matrix mat using sigma as the guess. If
      isTiming is True, then the raw time spent in scipy.sparse.linalg.eigs
      is returned. The eigenvectors are scaled so that the phase is equal 
      to zero where the magnitude is largest.
  '''
  k = min(k,scipy.shape(mat)[0]-2)
  t0 = time.time()
  if isHermitian:
    eigval,eigvec = scipy.sparse.linalg.eigsh(mat,k=k,sigma=sigma)
  else:
    eigval,eigvec = scipy.sparse.linalg.eigs(mat,k=k,sigma=sigma)
  eigsTime = time.time()-t0
  w,ind = sort(scipy.real(eigval),'ascend')
  v = eigvec[:,ind]
  indMax = scipy.argmax(abs(v),0)
  for ii in range(len(w)):
    v[:,ii] = v[:,ii]/v[indMax[ii],ii]*abs(v[indMax[ii],ii])
  if isTime:
    return w,v,eigsTime
  else:
    return w,v

def eigs_range(mat,startVal,endVal,isUseSparse=True,isHermitian=True,isGetStats=False):
  '''
  '''
  if isUseSparse:
    return eigs_range_sparse(mat,startVal,endVal,isHermitian=True,isGetStats=isGetStats)
  else:
    t0 = time.time()
    stats = {}
    if isHermitian:
      w,v = scipy.linalg.eigh(mat.todense())
    else:
      w,v = scipy.linalg.eig(mat.todense())
    stats['eigsTime']  = time.time()-t0
    ind = leq_tol(w,max([startVal,endVal]))*geq_tol(w,min([startVal,endVal]))
    if True not in ind:
      dist = w-(startVal+endVal)/2
      ind[scipy.argmin(dist)] = True
    stats['numEigs']   = scipy.sum(ind)
    stats['eigsCalls'] = 1
    stats['totalTime'] = time.time()-t0
    return (w[ind],v[:,ind],stats) if isGetStats else (w[ind],v[:,ind])  
    
def eigs_range_sparse(mat,startVal,endVal,k=24,tol=1e-6,guessOffset=2,maxitr=100,isHermitian=True,isGetStats=False):
  ''' Find all the eigenvalues of mat between startVal and endVal. This is done
      with multiple calls to scipy.sparse.linalg.eigs, finding k eigenvalues at
      a time. At most maxitr calls are made; if this number is reached without
      finding all eigenvalues, an error is raised.
  '''
  def prepare_next():
    isGap = False
    done  = False
    eigsRanges = [(min(eigsDict[guess][0]),max(eigsDict[guess][0]))\
                  for guess in scipy.sort(eigsDict.keys())]
    for ii in range(1,len(eigsRanges)):
      if not geq_tol(eigsRanges[ii-1][1],eigsRanges[ii][0],tol):
        guess = 0.5*(eigsRanges[ii-1][1]+eigsRanges[ii][0])
        isGap = True
    if not isGap:
      if startVal < endVal:
        done  = eigsRanges[-1][1] > endVal or guess > endVal
        guess = min(endVal,max(guess,eigsRanges[-1][1])+\
          (eigsRanges[-1][1]-eigsRanges[-1][0])/(k-1)*(k/2.-guessOffset))
      else:
        done  = eigsRanges[ 0][0] < endVal or guess < endVal
        guess = max(endVal,min(guess,eigsRanges[ 0][0])-\
          (eigsRanges[ 0][1]-eigsRanges[ 0][0])/(k-1)*(k/2.-guessOffset))
    if guess in eigsDict.keys():
      done = True
    return guess,done,isGap
  
  t0    = time.time()
  itr   = 0
  done  = False
  guess = startVal
  stats = {'eigsTime':0,'gapCount':0}
  eigsDict = {}
  while not done and itr < maxitr:
    itr = itr+1
    w,v,eigsTime        = eigs_sorted(mat,sigma=guess,k=k,isTime=True)
    eigsDict[guess]     = (w,v)
    guess,done,isGap    = prepare_next()
    stats['eigsTime']  += eigsTime
    stats['gapCount']  += int(isGap) 
  
  if not done:
    raise ValueError, 'Maximum number of iterations exceeded!'
  w,v = merge_eigs(eigsDict)
  ind = leq_tol(w,max([startVal,endVal]))*geq_tol(w,min([startVal,endVal]))
  if True not in ind:
    dist = w-(startVal+endVal)/2
    ind[scipy.argmin(dist)] = True
  stats['numEigs']   = scipy.sum(ind)
  stats['eigsCalls'] = itr
  stats['totalTime'] = time.time()-t0
  return (w[ind],v[:,ind],stats) if isGetStats else (w[ind],v[:,ind])

def eigs_num(mat,startVal,direction,numEigs,k=24,tol=1e-6,guessOffset=2,maxitr=100,isGetStats=False):
  ''' Find all the eigenvalues of mat between startVal and endVal. This is done
      with multiple calls to scipy.sparse.linalg.eigs, finding k eigenvalues at
      a time. At most maxitr calls are made; if this number is reached without
      finding all eigenvalues, an error is raised.
  '''
  def prepare_next():
    isGap = False
    eigsRanges = [(min(eigsDict[guess][0]),max(eigsDict[guess][0]))\
                  for guess in scipy.sort(eigsDict.keys())]
    for ii in range(1,len(eigsRanges)):
      if not geq_tol(eigsRanges[ii-1][1],eigsRanges[ii][0],tol):
        guess = 0.5*(eigsRanges[ii-1][1]+eigsRanges[ii][0])
        isGap = True
    if not isGap:
      if direction > 0:
        guess = max(guess,eigsRanges[-1][1])+\
          (eigsRanges[-1][1]-eigsRanges[-1][0])/(k-1)*(k/2.-guessOffset)
      else:
        guess = min(guess,eigsRanges[ 0][0])-\
          (eigsRanges[ 0][1]-eigsRanges[ 0][0])/(k-1)*(k/2.-guessOffset)
    if guess in eigsDict.keys():
      done = True
    return guess,isGap
  
  t0    = time.time()
  itr   = 0
  done  = False
  guess = startVal
  stats = {'eigsTime':0,'gapCount':0}
  eigsDict = {}
  while not done and itr < maxitr:
    itr = itr+1
    wNew,vNew,eigsTime = eigs_sorted(mat,sigma=guess,k=k,isTime=True)
    eigsDict[guess]    = (wNew,vNew)
    guess,isGap        = prepare_next()
    if not isGap:
      w,v = merge_eigs(eigsDict)
      eigsDict = {max(eigsDict.keys()) if direction > 0 else min(eigsDict.keys()):(w,v)}
      done = len(w) > numEigs
    stats['eigsTime'] += eigsTime
    stats['gapCount'] += int(isGap) 
  
  if not done:
    raise ValueError, 'Maximum number of iterations exceeded!'
  ind = scipy.arange(0,numEigs) if direction > 0 else scipy.arange(len(w)-numEigs,len(w))
  stats['numEigs']   = len(ind)
  stats['eigsCalls'] = itr
  stats['totalTime'] = time.time()-t0
  return (w[ind],v[:,ind],stats) if isGetStats else (w[ind],v[:,ind])

class SparseMaker():
  ''' SparseMaker class facilitates creation of sparse matrices, either by
      specifying the diagonals with an offset (relative to the top-left element
      on the diagonal) or by directly specifying the row and column for a
      desired matrix entry.
  '''
  def __init__(self,nzmax,shape=None,isComplex=False):
    self.nzmax = nzmax
    self.row = scipy.zeros(nzmax)
    self.col = scipy.zeros(nzmax)
    if isComplex:
      self.val = scipy.complex128(scipy.zeros(nzmax))
    else:
      self.val = scipy.zeros(nzmax)
    self.shape = shape
    self.index = 0
  
  def add_diag(self,d,offset,val):
    ''' Add the values given by val to diagonal d, with an offset relative to
        to the top left element of the matrix.
    '''
    num = len(val);
    if self.index+num > self.nzmax:
      raise ValueError, 'nzmax too small!'
    r = max(0,-d)+offset+scipy.arange(0,num)
    c = max(0, d)+offset+scipy.arange(0,num)
    self.row[self.index:self.index+num] = r
    self.col[self.index:self.index+num] = c
    self.val[self.index:self.index+num] = val
    self.index = self.index+num
  
  def add_elem(self,r,c,val):
    ''' Add elements to the matrix, with values given by val, at the row and
        column location specified by r and c.
    '''
    num = len(val)
    if self.index+num > self.nzmax:
      raise ValueError, 'nzmax too small!'
    self.row[self.index:self.index+num] = r
    self.col[self.index:self.index+num] = c
    self.val[self.index:self.index+num] = val
    self.index = self.index+num
  
  def assemble(self,format='csr'):
    ''' Assemble the matrix. The default data type returned is sparse coo 
        matrix from the scipy.sparse module.
    '''
    ij = [self.row[:self.index],self.col[:self.index]]
    if self.shape == None:
      self.shape = (max(self.row)+1,max(self.col)+1)
    if format == 'csr':
      return scipy.sparse.coo_matrix((self.val[:self.index],ij),self.shape).tocsr()
    elif format == 'lil':
      return scipy.sparse.coo_matrix((self.val[:self.index],ij),self.shape).tolil()
    else:
      raise AttributeError, 'matrix format is not supported!'
