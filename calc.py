''' Calc module contains useful mathematical methods used throughout ledsim.
'''
import scipy

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