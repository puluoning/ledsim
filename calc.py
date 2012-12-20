''' Calc module contains useful mathematical methods used throughout ledsim.
'''
import scipy

class Access():
  ''' Access class provides attribute and index access to the object dictionary.
  '''
  def __setattr__(self,attr,value):
    ''' Set the specified attribute to the given value. Note that __setattr__
        directly accesses the object dictionary.
    '''
    self.__dict__[attr] = value
  
  def __setitem__(self,attr,value):
    ''' Set the specified attribute to the given value. Note that __setitem__
        merely calls __setattr__ instead of accessing the object dictionary.
        For classes inheriting from Access, it is sufficient to overload
        __setattr__ to change the behavior of both methods.
    '''
    self.__setattr__(attr,value)
   
  def __getitem__(self,attr):
    ''' Get the specified attribute by accessing the object dictionary.
    '''
    return self.__dict__[attr]
  
  def attrs(self):
    ''' Return the keys of the object dictionary.
    '''
    return self.__dict__.keys()
  
class GenericOpts():
  ''' GenericOpts implements standard methods for options objects, such as
      GridOpts and SolverOpts. __setattr__ is overloaded to prevent setting
      of attributes which are not valid for the class.
  '''
  def __setattr__(self,attr,value):
    ''' Set the specified attribute to the given value, checking to make sure
        that the attribute is valid/allowed.
    '''
    if attr not in self.validAttrs:
      raise AttributeError, 'Opts error: %s is not a valid option' %(attr)
    else:
      self.__dict__[attr] = value
  
  def __str__(self):
    ''' Print the attributes of this object 
    '''
    outStr = ''
    attrWidthMax = max([len(attr) for attr in self.validAttrs]) 
    for attr in self.validAttrs:
      outStr += attr.ljust(attrWidthMax)+' : '+str(self.__dict__[attr])+'\n'
    return outStr

def diffuse(a,dz,Ld):
  ''' Diffuse the quantity a on the grid specified by dz. The diffusion 
      length is constant and given by Ld.
  '''
  if Ld == 0:
    return a
  else:
    b = scipy.copy(a)
    T = 1.
    D = Ld**2/(4*T)
    D = D*scipy.ones(len(dz))
    rnum   = len(dz)
    dtmax  = 0.25/max(D/dz**2)
    nsteps = max(rnum,scipy.ceil(T/dtmax))
    dt     = T/nsteps;
    ind1   = range(1,rnum)+[rnum-1]
    ind2   = [0]+range(0,rnum-1)
    for ii in scipy.arange(0,nsteps):
      b[1:-1] = (b+dt/(dz*(dz+dz[ind1]))*(D+D[ind1])*(b[ind1]-b)+ \
                   dt/(dz*(dz+dz[ind2]))*(D+D[ind2])*(b[ind2]-b))[1:-1]
    return b
  
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
  if order == 'ascend':
    ind = scipy.flipud(ind)
  return a[ind], ind

def eigs_range(mat,eigRange,k=40,overlap=2,offset=2,tol=1e-6,maxitr=2,forcePairs=True):
  ''' Calculate the eigenvalues for matrix mat within the range given by
      eigRange. This is done my multiple calls to eigs, each time searching
      for k eigenvalues. eigRange should start with the eigenvalue spectrum
      extremum.
  '''
  def eigs_sorted(mat,k,sigma,forcePairs):
    eigval,eigvec = scipy.sparse.linalg.eigs(mat,k=k,sigma=sigma)
    w,ind = sort(scipy.real(eigval),'descend')
    v = eigvec[:,ind]
    if forcePairs and w[-1] != w[-2]:
      return w[:-1],v[:,:-1]      
    else:
      return w,v
  def merge(w1,v1,w2,v2,overlap,tol):
    isGap  = (w2[0]-w1[-1-overlap])/w2[0] > tol
    newInd = scipy.array([False]*len(w2))
    for ii in range(0,len(newInd)):
      newInd[ii] = scipy.prod(abs((w2[ii]-w1)/w2[ii]) > tol) > 0
    w,ind  = sort(scipy.concatenate((w1,w2[newInd])),'descend')
    v = scipy.hstack((v1,v2[:,newInd]))[:,ind]
    return w,v,isGap
  isReverse = False
  factor = 1.
  if eigRange[0] > eigRange[1]:
    factor    = -1.
    eigRange  = [-val for val in eigRange]
    isReverse = True
  w,v = eigs_sorted(factor*mat,k,eigRange[0],forcePairs)
  done = max(w) > eigRange[-1]
  space = scipy.mean(w[1:]-w[:-1])
  guess = w[-1]+space*(k/2-offset)
  itr = 0
  while not done and itr < maxitr:
    itr = itr+1
    wNew,vNew = eigs_sorted(factor*mat,k,guess,forcePairs)
    w,v,isGap = merge(w,v,wNew,vNew,overlap,tol)
    if isGap:
      guess = 0.5*wNew[-1]
    else:
      space = scipy.mean(w[1:]-w[:-1])
      guess = w[-1]+space*(k/2-offset)
      done = max(w) > eigRange[-1]
  ind = (w <= eigRange[1])*(w >= eigRange[0])
  ind[0] = True
  if not isReverse:
    return  w[ind],v[:,ind]
  else:
    return -w[ind],v[:,ind]

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