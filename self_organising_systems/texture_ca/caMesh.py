# Lint as: python3
"""
Cellular Automata Model, adapted to run a trained model on a closed mesh.
"""
from self_organising_systems.texture_ca.config import cfg
import tensorflow as tf
import tensorflow_graphics as tfg
import numpy as np
import json
import os

import trimesh
#shorten stoopid function name
from tensorflow_graphics.geometry.convolution.graph_convolution import edge_convolution_template as meshConvolve

#from tensorflow_graphics.math.vector import cross as tfcross??


#take a trimesh mesh, and either 1 vector for a uniform orientation field, or 
#Assign each vertex an orientation vector that is tangent to mesh and closest to flow direction.
#Unitize it and also return its original magnitude.
#
def VertTans(mesh,dirs):
  cross = np.cross(mesh.vertex_normals,dirs)
  flows = np.cross(cross, mesh.vertex_normals)

  flowMags = tf.reshape(tf.norm(flows,axis=1),[-1,1])

  flows = tf.math.divide(flows,flowMags)     #unitize
  
  #***where flowMag is 0 set flow to 0,0,0 otherwise there will have been a divide by 0 - error or NaN in the data? idk.

  #print("flows is ",flows[:10])
  #print("flowMags is ",flowMags[:10])
  #print("norms ",tf.norm(flows,axis=1)[:10])
  
  return flows,flowMags
#end VertTans


#Sobel weights must sum to 0 else they'll detect gradient in a uniform field.
#fix by multiplying positive values for each vertex by one factor, and negative values by another, so their sums are equal.
#then normalize each vertex so |max| = .25.
def FixSobel(s,lens):
  sP = np.where(s > 0,s,0)
  sN = np.where(s < 0,s,0)

  #print("sP",sP[:lens[0]])
  
  sPsum = np.split(sP,np.cumsum(lens)[:-1])         #sum over each vertex - because ragged, it's a list of arrays
  sPsum = [ sum(x) for x in sPsum ]                 #np can't do this?
  sPsum = sP / np.repeat(sPsum,lens,axis=0)         #now + sums are 1
  
  sNsum = np.split(sN,np.cumsum(lens)[:-1])    
  sNsum = [ sum(x) for x in sNsum ]
  sNsum = -sN / np.repeat(sNsum,lens,axis=0)        #now - sums are -1

  s = sPsum + sNsum                                 #now sums are 0
  
  sSplit = np.split(s,np.cumsum(lens)[:-1])         #normalize on each vertex so |max| = .25
  sMax = [ max(abs(x)) for x in sSplit ]            
  s *= .25/np.repeat(sMax,lens,axis=0)               

  return s                                 
#end FixSobel


#Sobel weights must sum to 0 else they'll detect gradient in a uniform field.
#fix by setting center to - sum
#this is annoying because you have to add self pairs
#
def FixSobel2(s,lens):
  s *= .25                                    

  sSum = np.split(s,np.cumsum(lens)[:-1])            #sum over each vertex - because ragged, it's a list of arrays
  sSum = np.asarray([ sum(x) for x in sSum ])        
  
  center = sSum * -.25

  return s, center
#end FixSobel2


#discretized Sobels
#try this with center weights?
def FixSobel3(a,s,lens):
  s0 = np.sin(np.pi/8.0)                         #bin values into 2, 1, 0, -1, -2
  s1 = np.sin(3.0 * np.pi/8.0)
  d =  np.digitize(s,[-s1,-s0,s0,s1]).astype(int)
  d = np.take([-2.0, -1.0, 0.0, 1.0, 2.0],d)

  aSplit = np.split(a,np.cumsum(lens)[:-1])  
  sSplit = np.split(s,np.cumsum(lens)[:-1])      #sum over each vertex
  dSplit = np.split(d,np.cumsum(lens)[:-1])  

  dSum = np.asarray([ sum(x) for x in dSplit ])  #some don't sum to 0
  print("dsum",min(dSum),max(dSum),sum(abs(dSum))/len(dSum)) 

  # m = 0
  # for i in range(0,len(dSum)):
  #   if dSum[i] == 2.0:
  #     m = i
  #     print("a2",180.0 * aSplit[i]/np.pi)
  #     print("s2",sSplit[i])
  #     print("d2",dSplit[i])
  #     break

  bad = np.nonzero(dSum)[0]                      #indices of nonzero sums
  for i in bad:
    dS = dSplit[i]
    sSort = np.argsort(sSplit[i])                #sort values, consider only those not already |2|
    sSort = [ s for s in sSort if abs(dS[s]) < 2 ]    
    
    if dSum[i] == 1:                             #demote the lowest not already -2
      dSplit[i][sSort[0]] -= 1
    elif dSum[i] == -1:                          #promote the highest not already 2
      dSplit[i][sSort[-1]] += 1
    elif dSum[i] == 2:
      dSplit[i][sSort[0]] -= 1
      dSplit[i][sSort[1]] -= 1
    elif dSum[i] == -2:
      dSplit[i][sSort[-1]] += 1
      dSplit[i][sSort[-2]] += 1
  #end bad

  dSum = np.asarray([ sum(x) for x in dSplit ])    #crosscheck
  print("dsum",min(dSum),max(dSum),sum(abs(dSum))/len(dSum))
  
  d = np.concatenate(dSplit) * .125
  
  return d
#end FixSobel3


#discretize by taking the smallest value over s & c.  assign that to nearest axis. then use canned
#values like this:
#   2   2           -1   2
#0    0     <->  -2    0  
#  -2  -2           -1   2
#
#   2   2           -1   1
#0    0   0 <->  -2    0   2 
#  -2  -2           -1   1
#
#   1  2  1         -1  0  2
#0    0     <-> -2     0
#  -1 -2 -1         -1  0  2
#
#this is very slow as implemented. it needs canned values for every possible nblen.
#It demonstrates that a super crude approximation is good enough
#It gives slightly bigger feature size than trig versions, ?because it guaranteeds that every direction is maxed.
def FixSobel4(a,s,c,lens):
  aSplit = np.split(a,np.cumsum(lens)[:-1])
  sSplit = np.split(s,np.cumsum(lens)[:-1])
  cSplit = np.split(c,np.cumsum(lens)[:-1])
  size = len(lens)

  seq5a = [ 0, 2, 2,-2,-2 ]        #5-fold sin's starting from 0
  seq5b = [ 2, 1,-2,-2, 1 ]        #5-fold cos's
  
  seq6a = [ 0, 2, 2, 0,-2,-2 ]     #6-fold 
  seq6b = [ 2, 1,-1,-2,-1, 1 ]

  seq7a = [ 0, 1, 2, 1,-1,-2,-1 ]  #7-fold
  seq7b = [ 2, 1, 0,-2,-2, 0, 1 ]

  seqAs = [seq5a,seq6a,seq7a]
  seqBs = [seq5b,seq6b,seq7b]

  for i in range(0,size):
    nn = lens[i]
    seqA = seqAs[nn - 5]      #n-fold
    seqB = seqBs[nn - 5]

    ai = np.asarray([ x + 2.0*np.pi if x < 0 else x for x in aSplit[i] ])  #angles 0 - 2pi
    
    ms = np.argmin(abs(sSplit[i]))
    mc = np.argmin(abs(cSplit[i]))
    m = min([abs(sSplit[i][ms]),abs(cSplit[i][mc])])

    aSort = np.argsort(ai)                #indices that would sort this array
    aSort = np.lexsort((range(0,nn),aSort))   #index of each angle after sorting

    if m == abs(sSplit[i][ms]):
      aStart = ms                         #seqA starts at this angle
      seqS = seqA
      seqC = seqB
      
      if cSplit[i][ms] < 0:               #start at -2,0 (no action needed in 2,0 case)
        seqS = np.negative(seqS)
        seqC = np.negative(seqC)
    else:
      aStart = mc
      seqS = seqB
      seqC = seqA
      
      if sSplit[i][mc] < 0:               #start at 0,-2
        seqS = np.negative(seqS)
      else:
        seqC = np.negative(seqC)          #start at 0,2
      
    for j in range(0,nn):
      sSplit[i][j] = seqS[aSort[j] - aSort[aStart]]
      cSplit[i][j] = seqC[aSort[j] - aSort[aStart]]

  s = tf.concat(sSplit,axis=0) * .125
  c = tf.concat(cSplit,axis=0) * .125

  return s,c
#end FixSobel4


#Setup: pre-compute Sobel and Laplacian weights as sparse tensors.
#Flow vectors are unitized.
#Not used rn: flowMags gives their original magnitude
#i.e. how close the mesh-tangent vector was to the direction request: 0 if perpendicular, 1 if parallel
#use this to moderate dirF: interpolate it to range [1,dirF]
#
def Setup(mesh,flows,flowMags):
  #dir = [1,0,0]
  vs = mesh.vertices
  #ns = mesh.vertex_normals
  size = len(vs)

  neighbors = mesh.vertex_neighbors          #a ragged list
  nblens = np.asarray([len(i) for i in neighbors])
  nedges = sum(nblens)

  #mesh edge index pairs
  nbpairs = [ [ [i,j] for j in n ] for i,n in enumerate(neighbors) ]
  nbpairs = np.concatenate( nbpairs,axis=0 )

  #Sobel operators -- centers are 0 so only need to set neighbor values
  #
  nbX = vs[np.concatenate(neighbors)]        #concatenate neighbor vertices - doing it this way because ragged arrays are a PITA
  #broadcast stuff across neighbors
  vertX = np.repeat(vs,nblens,axis=0)                       #vertices 
  normX = np.repeat(mesh.vertex_normals,nblens,axis=0)      #vertex normals
  flowX = np.repeat(flows,nblens,axis=0)                    #flows
  
  nbDirX = trimesh.unitize(nbX - vertX)      #project direction from vertex to each neighbor, onto vertex normal plane
  d = np.einsum('ij,ij->i', nbDirX,normX)    #einsum dot
  d = np.einsum('i,ij->ij', d,     normX)    #einsum scalar * vector
  projX = trimesh.unitize(nbDirX - d)        

  cosX = np.einsum('ij,ij->i',projX,flowX)   #cos of angle from flow to each neighbor   
  sinX = np.einsum('ij,ij->i',normX,np.cross(flowX,projX))   #sin
  
  angleX = np.arctan2(sinX,cosX)            #angle, if needed        

  #s = sinX * .25                             #plausible, though weights don't sum to 0.
  #c = cosX * .25

  #want to see this work with:
  #*base FS: multiply low side of +- by factor to make it = high side, then normalize each vertex |max| to .25 
  #FS2: don't change edges, add center -sum.  need to add selfpairs.
  #FS3: discretize by binning & adjusting (needs angles) *bug*  -- try this with center weights?
  #*FS4: discretize by finding point nearest axis and using canned values (needs angles)
  
  s = FixSobel(sinX,nblens)
  c = FixSobel(cosX,nblens)

  #s = FixSobel3(angleX,sinX,nblens)
  #c = FixSobel3(angleX,cosX,nblens)

  #s,c = FixSobel4(angleX,sinX,cosX,nblens)

  #***if using flowmags this is the place?
  
  SXWeights = tf.cast(s,tf.float32)
  SYWeights = tf.cast(c,tf.float32)

  dx = tf.SparseTensor(nbpairs, values=SXWeights, dense_shape=[size,size])
  dy = tf.SparseTensor(nbpairs, values=SYWeights, dense_shape=[size,size])

  #Laplacians
  #
  #LPWeights = tf.repeat(tf.divide(2.0,nblens),nblens)       #this makes everything too big
  #LPWeights = tf.concat([LPWeights,tf.repeat(-2.0,size)],axis=0)
  
  LPWeights = tf.repeat(.25,nedges)  #even weighting
  #add center values of -.25 * nblen for each [v,v]
  LPWeights = tf.concat([LPWeights,[-0.25 * x for x in nblens]],axis=0)
  
  #add center index pairs
  selfpairs = list(zip(range(0,size),range(0,size)))  
  nbpairs = tf.concat([nbpairs,selfpairs],axis=0)
  
  lp = tf.SparseTensor(nbpairs, values=LPWeights, dense_shape=[size,size])

  #reshape to rank 3
  lp = tf.sparse.reshape(lp,[1,size,size])
  dx = tf.sparse.reshape(dx,[1,size,size])
  dy = tf.sparse.reshape(dy,[1,size,size])

  return(lp,dx,dy)

#end Setup


def get_variables(f):
  '''Get all vars involved in computing a function. used during training to grab parameters for saving.'''
  with tf.GradientTape() as g:
    f()
    return g.watched_variables()

def fake_quant(x, min, max):
  y = tf.quantization.fake_quant_with_min_max_vars(x, min=min, max=max)
  return y

def fake_param_quant(w):
  bound = tf.stop_gradient(tf.reduce_max(tf.abs(w)))
  w = fake_quant(w, -bound, bound)
  return w

def to_rgb(x):
  return x[..., :3]/(cfg.texture_ca.q) + 0.5

@tf.function
def meshPerceive(x, lp,dx,dy, angle=0.0, repeat=True):
  chn = tf.shape(x)[-1]
  size = tf.shape(x)[-2]   #can't do these after x has been touched

  ####################
  # #in 2D world
  # kernel = tf.stack([identity, c*dx-s*dy, s*dx+c*dy, laplacian], -1)[:, :, None, :]    #stack is (3, 3, 4), kernel is (3, 3, 1, 4) 
  # kernel = tf.repeat(kernel, chn, 2)                                                   #(3, 3, chn, 4)
  # y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], pad_mode)
  #####################
  #in mesh world x has shape=(1, size, chn) instead of (1, size, size, chn)
  #meshConvolve does the right thing with the channels, but gotta run each operator separately and stack them up after.
  
  #x and these have shape [1,size,chn]
  xdx = meshConvolve(x, dx, sizes=None, edge_function=lambda x,y: y , reduction='weighted', edge_function_kwargs={})  
  xdy = meshConvolve(x, dy, sizes=None, edge_function=lambda x,y: y , reduction='weighted', edge_function_kwargs={})  
  xlp = meshConvolve(x, lp, sizes=None, edge_function=lambda x,y: y , reduction='weighted', edge_function_kwargs={})
  #*** apply angle here in future ***

  x   = tf.expand_dims(x,-1)           #[1,size,chn,1]
  xdx = tf.expand_dims(xdx,-1) 
  xdy = tf.expand_dims(xdy,-1)
  xlp = tf.expand_dims(xlp,-1)

  y = tf.concat([x,xdx,xdy,xlp],-1)    #[1,size,chn,4]   

  y = tf.reshape(y,[1,size,4*chn])     #[1,size,4*chn]  interleave the 4 channels: the 4 0th elements, then the 4 1st...

  return y
#end meshPerceive

class DenseLayer:
  def __init__(self, in_n, out_n,
              init_fn=tf.initializers.glorot_uniform()):
    w0 = tf.concat([init_fn([in_n, out_n]), tf.zeros([1, out_n])], 0)
    self.w = tf.Variable(w0)

  #end init

  def embody(self):
    w = fake_param_quant(self.w) 
    w, b = w[:-1], w[-1]
    w = w[None,...]             #in 2D world, w = w[None, None, ...] 
    #w is 1,48,96 (layer 1)  1,96,12 (layer 2)
    #b is 96,                12,

    def f(x):
      #in 2d world receive 1,size,size,48 (layer 1) -> 1,size,size,96 (layer 2) -> 1,size,size,12 (out)
      #in mesh world only one size dimension.

      #in 2d world:
      # TF's matMul doesn't work with non-2d tensors, so using conv2d instead of 'tf.matmul(x, w)+b'
      #return tf.nn.conv2d(x, w, 1, 'VALID')+b          

      #in mesh world away with this nonsense just multiply them
      z = tf.matmul( tf.squeeze(x), tf.squeeze(w) ) + b
      return tf.expand_dims(z, axis=0)    
    #end f
    return f
  #end embody
#end DenseLayer

class CAMeshModel:

  #mesh is a trimesh, dirs is orientation vectors, either 1 only or 1 per vertex.
  #
  def __init__(self, mesh, dirs, params=None):
    super().__init__()
    self.fire_rate = cfg.texture_ca.fire_rate
    self.channel_n = cfg.texture_ca.channel_n

    init_fn = tf.initializers.glorot_normal(cfg.texture_ca.fixed_seed or None)
    self.layer1 = DenseLayer(self.channel_n*4, cfg.texture_ca.hidden_n, init_fn)
    self.layer2 = DenseLayer(cfg.texture_ca.hidden_n, self.channel_n, tf.zeros)

    #Mesh pre-computation: find flow direction and strength for each vertex
    flows, flowMags = VertTans(mesh,dirs)

    #get laplacian and sobel operators as sparseTensors
    self.Laplacian, self.SobelX, self.SobelY = Setup(mesh,flows,flowMags)

    global lpg
    lpg = self.Laplacian
    global sxg
    sxg = self.SobelX
    global syg
    syg = self.SobelY
    
    self.params = get_variables(self.embody)
    if params is not None:
      self.set_params(params)
  #end init

  def embody(self, quantized=True):
    layer1 = self.layer1.embody()
    layer2 = self.layer2.embody()
    #lp = self.Laplacian
    #dx = self.SobelX
    #dy = self.SobelY

    def noquant(x, min, max):
      return tf.clip_by_value(x, min, max)
    qfunc = fake_quant if quantized else noquant

    @tf.function
    def f(x, fire_rate=None, angle=0.0, step_size=1.0):
      y = meshPerceive(x,lpg,sxg,syg,angle)
      
      #print("Perceived",y.shape)

      #from here doesn't care about mesh topo, but everything's 1 dimension lower than with image
      y = qfunc(y, min=-cfg.texture_ca.q, max=cfg.texture_ca.q)
      y = tf.nn.relu(layer1(y))
      #print("layer 1 shape",y.shape)

      y = qfunc(y, min=0.0, max=cfg.texture_ca.q)
      y = layer2(y)
      #print("layer 2 shape",y.shape)
      
      dx = y*step_size
      dx = qfunc(dx, min=-cfg.texture_ca.q, max=cfg.texture_ca.q)
      if fire_rate is None:
        fire_rate = self.fire_rate

      #tf.random.set_seed(0)    #***put this in this and add a seed arg to update_mask to fix seed
      update_mask = tf.random.uniform(tf.shape(x[:, :, :1])) <= fire_rate     #lowered dimension here
      
      x += dx * tf.cast(update_mask, tf.float32)        #zero out change to cells not being updated
      x = qfunc(x, min=-cfg.texture_ca.q, max=cfg.texture_ca.q)

      return x
    #end f
    
    return f
  #end embody

  def get_params(self):
    return [p.numpy() for p in self.params]

  def set_params(self, params):
    for v, p in zip(self.params, params):
      v.assign(p)

  def save_params(self, filename):
    with tf.io.gfile.GFile(filename, mode='wb') as f: 
      np.save(f, self.get_params())

  def load_params(self, filename):
    with tf.io.gfile.GFile(filename, mode='rb') as f: 
      params = np.load(f, allow_pickle=True)
      self.set_params(params)


      
#end CAMeshModel
