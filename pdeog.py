import torch
import torch.autograd as tgrad

S = torch.Tensor([80]).requires_grad_()
t = torch.Tensor([0]).requires_grad_()
sigma = torch.Tensor([0.3]).requires_grad_()
r = torch.Tensor([0.05]).requires_grad_()
K = torch.Tensor([70])
T = torch.Tensor([1])
t2m = T-t
d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * t2m)/(sigma * torch.sqrt(t2m))
d2 = d1 - sigma * torch.sqrt(t2m)
N0 = lambda value: 0.5 * (1 + torch.erf((value/2**0.5)))
Nd1 = N0(d1)
Nd2 = N0(d2)
C = S* Nd1 - K* Nd2 *torch.exp(-r*t2m)

print("Option Price:", C.item()) #17.01496

dCdt, = (tgrad.grad(C, t, grad_outputs=torch.ones(C.shape), create_graph=True, only_inputs=True))
dCdS, = tgrad.grad(C, S, grad_outputs=torch.ones(C.shape), create_graph=True, only_inputs=True)
d2CdS2, = tgrad.grad(dCdS, S, grad_outputs=torch.ones(dCdS.shape), create_graph=True, only_inputs=True)
dCdvol, = tgrad.grad(C, sigma, grad_outputs=torch.ones(C.shape), create_graph=True, only_inputs=True)

dCdr = tgrad.grad(C, r, grad_outputs=torch.ones(C.shape), create_graph=True, only_inputs=True)
theta, delta, gamma, vega, rho = -dCdt[0], dCdS[0], d2CdS2[0], dCdvol[0], dCdr[0]

for og in [theta, delta, gamma, vega, rho]:
    print(f'{og.item():.4f}')

# Theta 5.8385
# Delta 0.7769
# Gamma 0.0124
# Vega 23.8776
# Rho 45.1372

print((-theta + 0.5*sigma**2 * S**2*gamma + r*S*delta - r*C).item())
# 0