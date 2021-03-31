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
  #cross = tfg.math.vector.cross(dirs,  mesh.vertex_normals)
  #flows = tfg.math.vector.cross(cross, mesh.vertex_normals)
  cross = np.cross(mesh.vertex_normals,dirs)
  flows = np.cross(cross, mesh.vertex_normals)

  flowMags = tf.reshape(tf.norm(flows,axis=1),[-1,1])

  #unitize
  flows = tf.math.divide(flows,flowMags) 
  
  #***where flowMag is 0 set flow to 0,0,0 otherwise there will have been a divide by 0 - error or NaN in the data? idk.

  #print("flows is ",flows[:10])
  #print("flowMags is ",flowMags[:10])
  #print("norms ",tf.norm(flows,axis=1)[:10])
  
  return flows,flowMags
#end VertTans

#Setup: pre-compute Sobel and Laplacian weights as sparse tensors.
#Flow vectors are unitized.
#Not used rn: flowMags gives their original magnitude
#i.e. how close the mesh-tangent vector was to the direction request: 0 if perpendicular, 1 if parallel
#use this to moderate dirF: interpolate it to range [1,dirF]
#
def Setup(mesh,flows,flowMags):
  #dir = [1,0,0]
  vs = mesh.vertices
  ns = mesh.vertex_normals
  size = len(vs)

  neighbors = mesh.vertex_neighbors     #a ragged list
  nblens = [len(i) for i in neighbors]
  nedges = sum(nblens)

  nbs = np.full((nedges,2),0)           #list of neighbor index pairs - gotta be a pythonic way to do this
  ct = 0
  for i,row in enumerate(neighbors):
    for j in row:
      nbs[ct] = [i,j]
      ct += 1
  #end nbs
  
  #Sobel operators
  #their center values are 0, the sparsetensor does that automatically.

  nbX = vs[np.concatenate(neighbors)]      #concatenate neighbor vertices
  vertX = np.repeat(vs,nblens,axis=0)      #broadcast vertices across neighbors

  nbDirX = trimesh.unitize(nbX - vertX)    #direction from vertex to each neighbor
  #print("nbdirs",nbDirX.shape)

  normX = np.repeat(ns,nblens,axis=0)      #?***check dimension here
  #print("normals",normX.shape)
  
  flowX = np.repeat(flows,nblens,axis=0)       
  #print("flows",flowX.shape)

                                           #project directions onto mesh-normal plane
  dotz = [ normX[i] * np.dot(nbDirX[i],normX[i]) for i in range(ct) ]
  #print("dotz",dotz.shape)

  projX = trimesh.unitize(nbDirX - dotz)

  #print("proj",projX[:10])
  
  cosAX = [ np.dot(projX[i],flowX[i]) for i in range(ct) ]
  print("cos ",min(cosAX),max(cosAX))

  #this is *unbelievably* inefficient.  if I could get to tfg maybe I could do better...
  sinAX = [ np.dot(normX[i], np.cross(flowX[i],projX[i])) for i in range(ct) ]
  print("sin ",min(sinAX),max(sinAX))

  angleX = np.arctan2(sinAX,cosAX)
  print("Angles ",angleX.min(),angleX.max())
  print("First angles",angleX[:nblens[0]])
  

  #if using flowmags this is where they would go, biasing sobels toward more even distribution?
  s = np.sin(angleX) * 2.0    
  c = np.cos(angleX) * 2.0
  
  SXWeights = tf.reshape(s,[-1])           #Sobel operators. flow is y direction
  SYWeights = tf.reshape(c,[-1])

  #divide by # neighbors
  sxsplit = np.split(SXWeights,np.cumsum(nblens)[:-1])
  sxsplit = [ w/nblens[i] for i,w in enumerate(sxsplit) ]
  SXWeights = np.concatenate(sxsplit)
  SXWeights = tf.cast(SXWeights,tf.float32)   #default trig type is float64

  sysplit = np.split(SYWeights,np.cumsum(nblens)[:-1])
  sysplit = [ w/nblens[i] for i,w in enumerate(sysplit) ]
  SYWeights = np.concatenate(sysplit)    
  SYWeights = tf.cast(SYWeights,tf.float32)
  
  dx = tf.SparseTensor(nbs, values=SXWeights, dense_shape=[size,size])
  dy = tf.SparseTensor(nbs, values=SYWeights, dense_shape=[size,size])  

  #Laplacians
  LPWeights = tf.repeat(tf.divide(2.0,nblens),nblens)  #even weighting
  #add center values of -2 for each [v,v]
  LPWeights = tf.concat([LPWeights,tf.repeat(-2.0,size)],axis=0)

  #add extra index pairs
  extra_nbs = list(zip(range(0,size),range(0,size)))  
  nbs = tf.concat([nbs,extra_nbs],axis=0)
  
  lp = tf.SparseTensor(nbs, values=LPWeights, dense_shape=[size,size])

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
  #meshConvolve does the right thing with the channels.
  
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

  y = tf.reshape(y,[1,size,4*chn])     #[1, 1387, 48]  interleave the 4 channels: the 4 0th elements, then the 4 1st...

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
