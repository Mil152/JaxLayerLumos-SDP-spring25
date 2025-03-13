from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import jax.numpy as jnp
import jaxlayerlumos
from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials
import matplotlib.pyplot as plt
import time
import os
import csv
import json
from pathlib import Path
from jaxlayerlumos import utils_spectra
from jaxlayerlumos import utils_radar_materials


#------------------------------Inputs------------------------------
lofreq=.2 #GHz lower bound for frequency test range
hifreq=2 #GHz higher bound for frequency test range
Nlayers=5 #number of RAM layers
maxthick=4.5 #total thickness
minthick=.001 #minimum layer thickness
allowdiff=.05 #5 percent allowed difference from total thickness
Nevals=3000 #max possible function evaluations
#------------------------------------------------------------------

maxlayer=maxthick*.75 #maximum layer thickness
frequencies = jnp.logspace(np.log10(lofreq*10**9), np.log10(hifreq*10**9), 500) #frequencies to evaluate
freqplot = jnp.logspace(np.log10(0.1), np.log10(10), 500)

#this is just a condensed version to do the stackrt function
def stacksolve(tlist,matsin,output):

  #make the thickness list
  stacklist=[]
  stacklist.append(0)
  for i in range(len(tlist)):
    stacklist.append(tlist[i])
  stacklist.append(0)
  d_stack = jnp.array(stacklist) *10**-3

  #make the materials list
  mats=[]
  mats.append("Air")
  for i in range(len(matsin)):
      mats.append(str(matsin[i]))
  mats.append("PEC")

  #get eps, mu, and then solve the stack
  eps_stack, mu_stack = utils_materials.get_eps_mu(mats, frequencies)
  R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies, 0.0) #eps, mu, thick, freq, angle

  R_avg = (R_TE + R_TM) / 2
  R_db = 10 * jnp.log10(R_avg).squeeze()
  #output 1 is for the objective funciton and output 2 is for the whole frequency range
  if output==1:
    return max(R_db)
  if output==2:
    return R_db

keepthick=[] #for plotting total thicknesses
keepref=[] #for plotting reflections
colors=[] #for coloring the point
evals=0 #keeping track of total funciton evaluations


# Define the objective function
def objective(params):

    #create the thickness list and materials dictionary entries
    tlist=[]
    mlist=[]
    for i in range(len(params)):
      if i<len(params)/2:
        string="t"+str(i+1)
        tlist.append(params[string])
      if i>=len(params)/2:
        string="m"+str(int(i+1-len(params)/2))
        mlist.append(params[string])

    global maxthick
    global allowdiff
    if sum(tlist)<=(1-allowdiff)*maxthick or sum(tlist)>=(1+allowdiff)*maxthick: #make sure the thickness is within a range
      loss=10**20 #punishment
      return {'loss': loss, 'status': STATUS_OK}


    else: #total thickness is within range
      loss=stacksolve(tlist,mlist,1)
      global evals
      evals+=1 #increment function evaluations by 1

    global Nevals
    if evals>=Nevals:
      raise StopIteration("Reached max evaluations... Exiting")

    if evals%100==0:
      print(round(evals/Nevals*100,3),"percent done")

    global keepthick
    keepthick.append(sum(tlist))
    global keepref
    keepref.append(loss)
    global colors
    colors.append((0,evals/Nevals,0))


    return {'loss': loss, 'status': STATUS_OK}

#Define the search space including continuous and discrete parameters
space = {}
for i in range(Nlayers*2):
  if i<Nlayers:
    string="t"+str(i+1)
    space.update({string:hp.uniform(string,minthick,maxlayer)})

  if i>=Nlayers:
    string="m"+str(i+1-Nlayers)
    space.update({string:hp.choice(string,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16])})
    #space[string] = hp.choice(string,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16])


#Use the optimizer
trials = Trials()
try:
  best = fmin(
      fn=objective,
      space=space,
      algo=tpe.suggest,
      max_evals=100000000, #this number needs to be unreasonably large
      trials=trials
  )

except StopIteration:
  best=trials.argmin

print("Best Parameters: ", best)

#pull out the best parameter results so they can be plugged back in
tlist=[]
mlist=[]
for i in range(len(best)):
  if i<len(best)/2:
    string="t"+str(i+1)
    tlist.append(best[string])
  if i>=len(best)/2:
    string="m"+str(int(i+1-len(best)/2))
    mlist.append(best[string]+1)

#print(t1_best, t2_best, t3_best, t4_best, t5_best, m1_best, m2_best, m3_best, m4_best, m5_best)
R_db = stacksolve(tlist,mlist,2) #this is nice to see the best result plotted
print("max:", max(R_db),"min:",min(R_db))
print("total thickness:", sum(tlist))

#IEEE paper LF thicknesses and reflections
paperLFx=[5.512,3.588,2.934,2.478]
paperLFy=[-33,-21,-18,-14]

#plot every single evalulated thickness and reflection result within the evaluated range
plt.figure(figsize=(10,6))
plt.scatter(keepthick, keepref,c=colors)
plt.plot(paperLFx,paperLFy,"g-*")

plt.xlabel("thickness [mm]")
plt.ylabel("Reflection (dB)")
plt.title("thickness vs reflection for Best Parameters")
plt.grid(True)
plt.show()
