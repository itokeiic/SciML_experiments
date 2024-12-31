import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
#TODO: DUSPSA need to be modeled as a torch.nn
#TODO: train_spsa needs to be implemented as incremental learning loop
init_val = 0.01 #initial value of stepsize parameter a
itr = 10 # number of iteration for spsa and number of increment, i.e., a is a torch.nn.parameters array of size itr
bs = 20 # mini batch size
train_itr = 50 # train iteration for optimizing a using Adam.

def func(x, q=8):
    return x[0]**2 + q*x[1]**2

def rosen(x, a=1, b=100):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
def spsa(x0, func, bounds=None, alpha=0.602, gamma=0.101, deltax_0=0.1, a=None, a_min=1.0e-6, c=1.0e-1, stepredf=0.5, gtol=1.0e-5, graditer=1, memsize=100, IniNfeval=0, maxiter=5000, adaptive_step=False, relaxation=True, dynamic_system=False, *args):
    #print("starting spsa")
    redcounter = 0
    if not dynamic_system:
        Npar = len(x0)
    else:
        Npar = len(x0) - 1

    def g_sa(x, func, ck, niter, *args):
        p = len(x)
        gsum = torch.zeros(p)
        yp = 0.0
        ym = 0.0
        xp = x.clone()
        xm = x.clone()
        delta = torch.zeros(p)

        if niter > 0:
            for m in range(niter):
                delta = 2 * torch.floor(2 * torch.rand(p)) - 1
                xp = x + ck * delta
                xm = x - ck * delta
                if dynamic_system:
                    xp[-1] = xm[-1] = x[-1]
                yp = func(xp, *args)
                ym = func(xm, *args)
                gsum += (yp - ym) / (2 * ck * delta)
            ghat = gsum / niter
            if torch.linalg.norm(ghat).item() == 0.:
                print("ghat is zero!")
                print ("yp,ym = ",yp.item(),ym.item())
                print ("x,xp,xm = ",x,xp,xm)
                print ("ck = ",ck)
                print ("delta = ",delta)
        else:
            ghat = torch.zeros(p)
        if dynamic_system:
            ghat[-1] = 0
        return ghat, yp, ym, xp, xm, delta

    Xmax = []
    Xmin = []
    if bounds is None:
        bounds = [(-10.0, 10.0) for _ in range(Npar)]
        # print("No bounds specified. Default:(-10,10).")
    if len(bounds) != Npar:
        raise ValueError("Number of parameters Npar != length of bounds")
    Xmin = [bound[0] for bound in bounds]
    Xmax = [bound[1] for bound in bounds]
    Nfeval = IniNfeval
    #x0 = torch.tensor(x0, requires_grad=True)
    x0 = x0.clone().detach().requires_grad_(True)
    # history = []
    # historyx = []
    p = len(x0)
    A = int(np.floor(0.1 * maxiter))
    y0 = func(x0, *args)
    Nfeval += 1
    mem = [y0] * memsize
    x = x0.clone()
    a_ini = 0.0
    # print("initial objective value = ", y0)
    x_best = x0.clone()
    y_best = y0
    for k in range(1, maxiter + 1):
        if dynamic_system:
            x[-1] = k
        if type(c) is torch.nn.parameter.Parameter:
            ck = c[k-1] / (k + 1)**gamma
        else:
            ck = c / (k + 1)**gamma
        ghat, yp, ym, xp, xm, delta = g_sa(x, func, ck, graditer, *args)
        Nfeval += graditer * 2
        if k == 1:
            if a is None:
                a = deltax_0 * (A + 1)**alpha / (torch.min(torch.abs(ghat[0:Npar])).item() + gtol)
                #a=deltax_0*(A+1)**alpha/(min(abs(ghat[:Npar])))
                #print("a: ",a)
            a_ini = a
            # print("ghat0 = ", ghat[:])
            # print ("a_ini: ",a_ini)
            # print("ck =",ck)
        if type(a) is torch.nn.parameter.Parameter:
            ak = a[k-1] / (k + 1 + A)**alpha
        else:
            #print("k: %d, a = %f, "%(k,a),"ghat = ",ghat)
            ak = a / (k + 1 + A)**alpha
        # print("k: $k, ym = $ym, yp = $yp, a = $a")
        xold = x.clone()
        x = x - ak * ghat
        # for m in range(Npar):
            # x[m] = torch.clamp(x[m], min=Xmin[m], max=Xmax[m])
        # y = func(x, *args)
        # push!(history, [Nfeval, y])
        # push!(historyx, copy(x))
        mem = mem[1:] + [min(ym.item(), yp.item())]
        if ym < y_best:
            x_best = xm
            y_best = ym
        if yp < y_best:
            x_best = xp
            y_best = yp
        if adaptive_step: #Automatic Differentiation does not work when a is changed in place.
            if ((y0 - min(yp.item(), ym.item())) < 0):
                print("divergence detected. reinitializing.")
                redcounter += 1
                print("current a: ", a)
                x = x_best.clone()
                #a[k] = stepredf * a[k]
                a = stepredf * a
                print("new a[%d]: "%k, a)
                if (redcounter > int(np.floor(0.05 * maxiter))) and relaxation:
                    print("Too many divergence. Resetting a and relaxing threshold!")
                    a = a_ini
                    y0 = min(yp.item(), ym.item())
                    redcounter = 0
        #x.requires_grad = True  # Ensure gradients are tracked for the next iteration

    y = func(x, *args)
    Nfeval += 1
    #print("achieved objective: ", y)
    # push!(history, [Nfeval, y])
    # push!(historyx, copy(x))
    # print("number of function evaluation: ", Nfeval)
    # return x, y, history, historyx, Nfeval
    return x
    
class DUSPSA(nn.Module):
    def __init__(self, func, num_itr, Npar = 2):
        super(DUSPSA, self).__init__()
        self.a = nn.Parameter(init_val*torch.ones(num_itr))
        self.c = nn.Parameter(init_val*torch.ones(num_itr))
        self.Npar = Npar
        self.func = func
        
    def forward(self, num_itr, bs): #number of increments and batch size
        X0 = (torch.rand(bs, self.Npar)*20.0 - 10.0)
        s = [spsa(x0, self.func, a=self.a, c=self.c, maxiter = num_itr) for x0 in X0]        
        return torch.stack(s, dim=0)

class TrainASPSA(nn.Module):
    def __init__(self, func, num_itr, Npar = 2):
        super(TrainASPSA, self).__init__()
        self.Npar = Npar
        self.func = func
        
    def forward(self, num_itr, bs):
        X0 = (torch.rand(bs, self.Npar)*20.0 - 10.0) # ランダムな初期点を設定
        s = [spsa(x0, self.func, a=None, maxiter = num_itr, adaptive_step=True) for x0 in X0]
        return torch.stack(s, dim=0)
# def train_spsa(opt, eta, max_itr, train_itr):
    # eta = torch.tensor(eta, requires_grad=True)
    # l = []
    # for i in range(train_itr):
        # l.append(eta[0].item())
        # x0 = torch.rand(2)
        # def closure():
            # loss = spsa(x0, rosen, a=eta[0], maxiter=max_itr)
            # loss.backward()
            # return loss
        # opt.zero_grad()
        # loss = closure()
        # opt.step()
    # return l

model=DUSPSA(func, itr, Npar=2)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
solution = torch.tensor([[0.0,0.0]]).repeat(bs,1)
l=[]
for gen in range(itr):
    for i in range(train_itr):
        opt.zero_grad()
        s=model(gen+1, bs)
        loss = loss_func(s, solution)
        loss.backward()
        opt.step()
    print(gen, loss.item())
    l.append(loss.item())

X0 = (torch.rand(bs, 2)*20.0 - 10.0)
s = [spsa(x0, func, a=None, maxiter = itr, adaptive_step=True) for x0 in X0]
loss = loss_func(torch.stack(s, dim=0), solution)
print ("original_a_spsa: ", loss.item())

aspsa_model = TrainASPSA(func, itr, Npar=2)
## trained DUSPSA model
bs = 100
solution = torch.tensor([[0.0, 0.0]]).repeat(bs,1) #解
plt.figure()
with torch.no_grad():
    for i in range(1): 
        norm_list = []
        itr_list = []
        for i in range(itr):
            s_hat = model(i, bs)
            err = (torch.norm(solution - s_hat)**2).item()/bs
            norm_list.append(math.log10(err))
            itr_list.append(i)
        plt.plot(itr_list, norm_list, color="red", label="DUSPSA",marker='o')
## normal ASPSA

norm_list = []
itr_list = []
for i in range(itr):
    s_hat = aspsa_model(i, bs)
    err = (torch.norm(solution - s_hat)**2).item()/bs
    norm_list.append(math.log10(err))
    itr_list.append(i)

plt.plot(itr_list, norm_list, color="green", label="A_SPSA",marker='o')
plt.title("Error curves")
plt.grid()
plt.xlabel("iteration")
plt.ylabel("log10 of squared error")
plt.legend()

g = model.a.to("cpu")
h = model.c.to("cpu")
gval = g.detach().numpy()
gval = gval[0:itr]
hval = h.detach().numpy()
hval = hval[0:itr]

ind = np.linspace(0,itr-1,itr)
plt.figure()
plt.plot(ind, gval,label="a",marker='o')
plt.plot(ind, hval,label="c",marker='o')
plt.xlabel("index t")
plt.ylabel("parameter values")
plt.grid()
plt.legend()
plt.show()
# l = train_spsa(opt, eta, max_itr, train_itr)
# print("a: ", l)

# # Plotting (using matplotlib)
# import matplotlib.pyplot as plt
# plt.plot(range(1, train_itr + 1), l, xlabel="Iteration", ylabel="a", lw=2, color='black')
# plt.show()