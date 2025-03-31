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
import jax
import random
import csv

# cd /d E:\spring2025school\SeniorDesign\code
# singlewindow1.py

# ------------------------------Inputs------------------------------
lofreq = 2  # GHz lower bound for frequency test range
hifreq = 8  # GHz higher bound for frequency test range
Nevals = 50  # max possible function evaluations
minthick = .001  # minimum layer thickness
allowdiff = .35  # multiple of allowed difference from total thickness value
maxcoeff = .85  # max individual layer percent of goal thickness allowed
Nlayers = 5  # number of RAM layers
current = 1  # goal thickness [mm]
# ------------------------------------------------------------------

allmats = []
alllayerthick = []
alltotthick = []
allref = []
paretomats = []
parlayerthick = []
partotthick = []
parref = []

keepthick = []  # for plotting total thicknesses
keepref = []  # for plotting reflections
colors = []  # for coloring the point

plotcolor = (random.random(), random.random(), random.random())  # i

maxthick = current  # total thickness
maxlayer = maxthick * maxcoeff  # maximum layer thickness
frequencies = jnp.logspace(np.log10(lofreq * 10 ** 9), np.log10(hifreq * 10 ** 9), 500)  # frequencies to evaluate
freqplot = jnp.logspace(np.log10(0.1), np.log10(10), 500)


# this is just a condensed version to do the stackrt function
def stacksolve(tlist, matsin, output):
    # make the thickness list
    stacklist = []
    stacklist.append(0)
    for i in range(len(tlist)):
        stacklist.append(tlist[i])
    stacklist.append(0)
    d_stack = jnp.array(stacklist) * 10 ** -3

    # make the materials list
    mats = []
    mats.append("Air")
    for i in range(len(matsin)):
        mats.append(str(matsin[i]))
    mats.append("PEC")

    # get eps, mu, and then solve the stack
    eps_stack, mu_stack = utils_materials.get_eps_mu(mats, frequencies)
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies,
                                            0.0)  # eps, mu, thick, freq, angle

    R_avg = (R_TE + R_TM) / 2
    R_db = 10 * jnp.log10(R_avg).squeeze()
    # output 1 is for the objective funciton and output 2 is for the whole frequency range
    if output == 1:
        return max(R_db)
    if output == 2:
        return R_db


evals = 0  # keeping track of total funciton evaluations


# Define the objective function
def objective(params):
    # create the thickness list and materials dictionary entries
    tlist = []
    mlist = []
    for i in range(len(params)):
        if i < len(params) / 2:
            string = "t" + str(i + 1)
            tlist.append(params[string])
        if i >= len(params) / 2:
            string = "m" + str(int(i + 1 - len(params) / 2))
            mlist.append(params[string])

    global maxthick
    global allowdiff
    if sum(tlist) <= (1 - allowdiff) * maxthick or sum(tlist) >= (
            1 + allowdiff) * maxthick:  # make sure the thickness is within a range
        loss = 10 ** 20  # punishment
        return {'loss': loss, 'status': STATUS_OK}


    else:  # total thickness is within range
        loss = stacksolve(tlist, mlist, 1)
        global evals
        evals += 1  # increment function evaluations by 1

        ##get pareto
        global allmats
        global alllayerthick
        global alltotthick
        global allref
        allmats.append(mlist)
        alllayerthick.append(tlist)
        alltotthick.append(sum(tlist))
        allref.append(loss)

    global Nevals
    if evals >= Nevals:
        raise StopIteration("Reached max evaluations... Exiting")

    if evals % 100 == 0:
        print(round(evals / Nevals * 100, 3), "percent done")

    global keepthick
    keepthick.append(sum(tlist))
    global keepref
    keepref.append(loss)
    global colors
    global plotcolor
    global runs
    colors.append(plotcolor)  # ((plotcolor/runs,0,0))

    return {'loss': loss, 'status': STATUS_OK}


# Define the search space including continuous and discrete parameters
space = {}
for i in range(Nlayers * 2):
    if i < Nlayers:
        string = "t" + str(i + 1)
        space.update({string: hp.uniform(string, minthick, maxlayer)})

    if i >= Nlayers:
        string = "m" + str(i + 1 - Nlayers)
        space.update({string: hp.choice(string, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16])})
        # space[string] = hp.choice(string,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16])

maxcoeff -= .01
allowdiff -= .015
# Use the optimizer
trials = Trials()
try:
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100000000,  # this number needs to be unreasonably large
        trials=trials
    )

except StopIteration:
    best = trials.argmin

##get pareto
thinnest = 100
thinnestidx = 0
for i in range(len(alltotthick)):
    if alltotthick[i] < thinnest:
        thinnest = alltotthick[i]
        thinnestidx = i

paretomats.append(allmats[thinnestidx])
parlayerthick.append(alllayerthick[thinnestidx])
partotthick.append(alltotthick[thinnestidx])
parref.append(allref[thinnestidx])
npareto = 1

# make new sorted lists
newmats = []
newlayerthick = []
newtotthick = []
newref = []
newthinnsetidx = 0
cont = 1
while cont == 1:
    newthinnest = 100
    for i in range(len(alltotthick)):
        if alltotthick[i] > thinnest:
            if alltotthick[i] < newthinnest:
                newthinnest = alltotthick[i]
                newthinnestidx = i
    thinnest = newthinnest
    newmats.append(allmats[newthinnestidx])
    newlayerthick.append(alllayerthick[newthinnestidx])
    newtotthick.append(alltotthick[newthinnestidx])
    newref.append(allref[newthinnestidx])

    if len(newmats) >= len(allmats):
        cont = 0

# make pareto points
for i in range(len(newref)):
    if newref[i] < parref[npareto - 1]:
        paretomats.append(newmats[i])
        parlayerthick.append(newlayerthick[i])
        partotthick.append(newtotthick[i])
        parref.append(newref[i])
        npareto += 1

# IEEE paper LF thicknesses and reflections
paperLFx = [5.512, 3.588, 2.934, 2.478]
paperLFy = [-33, -21, -18, -14]

# IEEE paper HF thicknesses and reflections
paperHFx = [5.244, 2.670, 1.761, 1.236]
paperHFy = [-23.5, -19.8, -17, -13]

# plot every single evalulated thickness and reflection result within the evaluated range
plt.figure(figsize=(10, 6))
plt.scatter(keepthick, keepref, c=colors, label="All Evaluations")
# plt.plot(partotthick,parref,"r-o",label="BO Pareto")


##gradient descend each pareto point
print(" ")
cparref = []  # copy pareto points to mess with
cparlayerthick = []
cpartotthick = []

for i in range(len(parref)):
    cparref.append(parref[i])
    cparlayerthick.append(parlayerthick[i])
    cpartotthick.append(partotthick[i])


def ref4grad(tlist, matsin):
    return stacksolve(tlist, matsin, 1)


for i in range(len(parref)):
    print(round(i / len(parref) * 100, 4), "percent done with gradient descent")
    cont1 = 1
    for k in range(10):
        # R_db=stacksolve(parlayerthick[i],paretomats[i],2)
        gradients = jax.grad(ref4grad, argnums=0)
        gradtlist = gradients(parlayerthick[i], paretomats[i])
        cparref[i] = ref4grad(parlayerthick[i], paretomats[i])
        for j in range(len(gradtlist)):  # all 5 layers
            grad = gradtlist[j]  # gradient for a current layer
            cparlayerthick[i][j] -= grad * .001  # descend a little bit
            cpartotthick[i] = sum(parlayerthick[i])

        # add gradient points to the original pareto set
allref = []
alllayerthick = []
alltotthick = []

for i in range(len(parref)):
    allref.append(cparref[i])
    alllayerthick.append(cparlayerthick[i])
    alltotthick.append(cpartotthick[i])
    allmats.append(allmats[i])
for i in range(len(parref)):
    allref.append(parref[i])
    alllayerthick.append(parlayerthick[i])
    alltotthick.append(partotthick[i])

##get a new pareto
paretomats = []
parlayerthick = []
partotthick = []
parref = []
thinnest = 100
thinnestidx = 0
for i in range(len(alltotthick)):
    if alltotthick[i] < thinnest:
        thinnest = alltotthick[i]
        thinnestidx = i

paretomats.append(allmats[thinnestidx])
parlayerthick.append(alllayerthick[thinnestidx])
partotthick.append(alltotthick[thinnestidx])
parref.append(allref[thinnestidx])
npareto = 1

# make new sorted lists
newmats = []
newlayerthick = []
newtotthick = []
newref = []
newthinnsetidx = 0
cont = 1
while cont == 1:
    newthinnest = 100
    for i in range(len(alltotthick)):
        if alltotthick[i] > thinnest:
            if alltotthick[i] < newthinnest:
                newthinnest = alltotthick[i]
                newthinnestidx = i
    thinnest = newthinnest
    newmats.append(allmats[newthinnestidx])
    newlayerthick.append(alllayerthick[newthinnestidx])
    newtotthick.append(alltotthick[newthinnestidx])
    newref.append(allref[newthinnestidx])

    if len(newmats) >= len(allmats):
        cont = 0

# make pareto points
for i in range(len(newref)):
    if newref[i] < parref[npareto - 1]:
        paretomats.append(newmats[i])
        parlayerthick.append(newlayerthick[i])
        partotthick.append(newtotthick[i])
        parref.append(newref[i])
        npareto += 1
##

# get best reflection materials and thicknesses
allref = np.array(allref)
allmats = np.array(allmats)
alllayerthick = np.array(alllayerthick)
alltotthick = np.array(alltotthick)
sorted_indices = np.argsort(allref)

bestmats = []
bestlayerthick = []
besttot = []
bestref = []

for i in range(min(40, len(sorted_indices))):
    idx = sorted_indices[i]
    bestmats.append(allmats[idx])
    bestlayerthick.append(alllayerthick[idx])
    besttot.append(alltotthick[idx])
    bestref.append(allref[idx])

for i in range(len(bestmats)):
    print(f"Top {i + 1} Material Configuration:")
    print("Materials:", bestmats[i])
    print("Layer Thicknesses:", bestlayerthick[i])
    print("Total Thickness:", besttot[i])
    print("Reflection (dB):", bestref[i])
    print("-" * 20)

# save data to csv
data = [partotthick, parref]
filename = "singlewindowsave.csv"
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file", filename, " saved successfully.")

plt.plot(partotthick, parref, "b-o", label="BO with Gradient Pareto Front")
plt.plot(paperHFx, paperHFy, "g-*", label="IEEE Paper Results")

plt.xlabel("Total Structure Thickness [mm]")
plt.ylabel("Reflection (dB)")
plt.title("5 Layer RAM Structures in 2-8 GHz")
plt.legend()
plt.grid(True)
plt.show()