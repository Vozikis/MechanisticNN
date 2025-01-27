
# import torch.nn as nn
# import torch
# from torch.nn.parameter import Parameter
# import numpy as np

# import matplotlib.pyplot as plt
# import torch
# import torch.optim as optim
# from torch.autograd import gradcheck


# from torch.utils.data import Dataset, DataLoader
# from scipy.integrate import odeint

# from extras.source import write_source_files, create_log_dir

# from solver.ode_layer import ODEINDLayer
# import discovery.basis as B
# import ipdb
# import extras.logger as logger
# import os

# from scipy.special import logit
# import torch.nn.functional as F
# from tqdm import tqdm
# import discovery.plot as P


# log_dir, run_id = create_log_dir(root='logs')
# write_source_files(log_dir)
# L = logger.setup(log_dir, stdout=False)

# DBL=True
# dtype = torch.float64 if DBL else torch.float32
# STEP = 0.01
# cuda=True
# T = 2000 #minimum that has to be else there is error
# n_step_per_batch = 30
# batch_size= 512
# #weights less than threshold (absolute) are set to 0 after each optimization step.
# threshold = 0.1


# class LorenzDataset(Dataset):
#     def __init__(self, n_step_per_batch=100, n_step=1000):
#         self.n_step_per_batch=n_step_per_batch
#         self.n_step=n_step
#         self.end= n_step*STEP
#         x_train = self.generate()

#         self.down_sample = 1

#         self.x_train = torch.tensor(x_train, dtype=dtype) 
#         self.x_train = self.x_train 

#         #Create basis for some stats. Actual basis is in the model
#         basis,basis_vars =B.create_library(x_train, polynomial_order=2, use_trig=False, constant=True)
#         self.basis = torch.tensor(basis)
#         self.basis_vars = basis_vars
#         self.n_basis = self.basis.shape[1]

#     def generate(self):
#         rho = 28.0
#         sigma = 10.0
#         beta = 8.0 / 3.0
#         dt = 0.01

#         def f(state, t):
#             x, y, z = state
#             return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

#         state0 = [1.0, 1.0, 1.0]
#         time_steps = np.linspace(0, self.end, self.n_step)
#         self.time_steps = time_steps

#         x_train = odeint(f, state0, time_steps)
#         return x_train

#     def __len__(self):
#         return (self.n_step-self.n_step_per_batch)//self.down_sample

#     def __getitem__(self, idx):
#         i = idx*self.down_sample
#         d=  self.x_train[i:i+self.n_step_per_batch]
#         return i, d


# ds = LorenzDataset(n_step=T,n_step_per_batch=n_step_per_batch)#.generate()
# print("the shape is 1 ",len(ds))
# print("the shape is ",len(ds[0][1]))
# train_loader =DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True) 

# #plot train data
# P.plot_lorenz(ds.x_train, os.path.join(log_dir, 'train.pdf'))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class Model(nn.Module):
#     def __init__(self, bs, n_step,n_step_per_batch, n_basis, device=None, **kwargs):
#         super().__init__()

#         self.n_step = n_step #+ 1
#         self.order = 2
#         # state dimension
#         self.bs = bs
#         self.device = device
#         self.n_iv=1
#         self.n_ind_dim = 3
#         self.n_step_per_batch = n_step_per_batch

#         self.n_basis = ds.n_basis

#         self.init_xi = torch.randn((1, self.n_basis, self.n_ind_dim), dtype=dtype).to(device)

#         self.mask = torch.ones_like(self.init_xi).to(device)

#         #Step size is fixed. Make this a parameter for learned step
#         self.step_size = (logit(0.01)*torch.ones(1,1,1))
#         self.xi = nn.Parameter(self.init_xi.clone())
#         self.param_in = nn.Parameter(torch.randn(1,64))

#         init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2, dtype=dtype)
#         self.init_coeffs = nn.Parameter(init_coeffs)
        
#         self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, solver_dbl=True, double_ret=True,
#                                     n_iv=self.n_iv, n_iv_steps=1,  gamma=0.05, alpha=0, **kwargs)


#         self.param_net = nn.Sequential(
#             nn.Linear(64, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, self.n_basis*self.n_ind_dim)
#         )

#         self.net = nn.Sequential(
#             nn.Linear(self.n_step_per_batch*self.n_ind_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, self.n_step_per_batch*self.n_ind_dim)
#         )
    
#     def reset_params(self):
#         #reset basis weights to random values
#         self.xi.data = torch.randn_like(self.init_xi)

#     def update_mask(self, mask):
#         self.mask = self.mask*mask
    
#     def get_xi(self):
#         xi = self.param_net(self.param_in)
#         xi = xi.reshape(self.init_xi.shape)
#         return xi

#     def forward(self, index, net_iv):
#         # apply mask
#         xi = self.get_xi()
#         #xi = _xi

#         #xi = self.mask*self.xi
#         xi = self.mask*xi
#         _xi = xi
#         xi = xi.repeat(self.bs, 1,1)


#         var = self.net(net_iv.reshape(self.bs,-1))
#         var = var.reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)

#         #create basis
#         var_basis,_ = B.create_library_tensor_batched(var, polynomial_order=2, use_trig=False, constant=True)

#         rhs = var_basis@xi
#         rhs = rhs.permute(0,2,1)

#         z = torch.zeros(1, self.n_ind_dim, 1,1).type_as(net_iv)
#         o = torch.ones(1, self.n_ind_dim, 1,1).type_as(net_iv)

#         coeffs = torch.cat([z,o,z], dim=-1)
#         coeffs = coeffs.repeat(self.bs,1,self.n_step_per_batch,1)

#         init_iv = var[:,0]

#         #steps = self.step_size*torch.ones(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(net_iv)
#         steps = self.step_size.repeat(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(net_iv)

#         steps = torch.sigmoid(steps)
#         #self.steps = self.steps.type_as(net_iv)

#         x0,x1,x2,eps,steps = self.ode(coeffs, rhs, init_iv, steps)
#         x0 = x0.permute(0,2,1)

#         return x0, steps, eps, var,_xi

# model = Model(bs=batch_size,n_step=T, n_step_per_batch=n_step_per_batch, n_basis=ds.n_basis, device=device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# if DBL:
#     model = model.double()
# model=model.to(device)


# def print_eq(stdout=False):
#     #print learned equation
#     xi = model.get_xi()
#     #repr_dict = B.basis_repr(model.xi*model.mask, ds.basis_vars)
#     repr_dict = B.basis_repr(xi*model.mask, ds.basis_vars)
#     code = []
#     for k,v in repr_dict.items():
#         L.info(f'{k} = {v}')
#         if stdout:
#             print(f'{k} = {v}')
#         code.append(f'{v}')
#     return code

# def simulate(gen_code):
#     #simulate learned equation
#     def f(state, t):
#         x0, x1, x2= state

#         dx0 = eval(gen_code[0])
#         dx1 = eval(gen_code[1])
#         dx2 = eval(gen_code[2])

#         return dx0, dx1, dx2
        
#     state0 = [1.0, 1.0, 1.0]
#     time_steps = np.linspace(0, T*STEP, T)

#     x_sim = odeint(f, state0, time_steps)
#     return x_sim

# def train():
#     """Optimize and threshold cycle"""
#     model.reset_params()

#     max_iter = 150
#     for step in range(max_iter):
#         print(f'Optimizer iteration {step}/{max_iter}')

#         #threshold
#         if step > 0:
#             xi = model.get_xi()
#             mask = (xi.abs() > threshold).float()

#             L.info(xi)
#             L.info(xi*model.mask)
#             L.info(model.mask)
#             L.info(model.mask*mask)

#         code = print_eq(stdout=True)
#         #simulate and plot

#         x_sim = simulate(code)
#         P.plot_lorenz(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

#         #set mask
#         if step > 0:
#             model.update_mask(mask)
#             model.reset_params()

#         optimize()


# def optimize(nepoch=100):
#     with tqdm(total=nepoch) as pbar:
#         for epoch in range(nepoch):
#             pbar.update(1)
#             for i, (index, batch_in) in enumerate(train_loader):
#                 batch_in = batch_in.to(device)

#                 optimizer.zero_grad()
#                 x0, steps, eps, var,xi = model(index, batch_in)

#                 x_loss = (x0- batch_in).pow(2).mean()
#                 loss = x_loss +  (var- batch_in).pow(2).mean()
                

#                 loss.backward()
#                 optimizer.step()


#             xi = xi.detach().cpu().numpy()
#             meps = eps.max().item()
#             L.info(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')
#             print(f'basis\n {xi}')
#             pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')


# if __name__ == "__main__":
#     train()

#     print_eq()












import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

from solver.ode_layer import ODEINDLayer
import discovery.basis as B
import ipdb
import extras.logger as logger
import os

from scipy.special import logit
import torch.nn.functional as F
from tqdm import tqdm
import discovery.plot as P

log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=False)

DBL = True
dtype = torch.float64 if DBL else torch.float32
STEP = 0.01
cuda = True
T = 2000 
n_step_per_batch = 30
batch_size = 512
# Weights less than threshold (absolute) are set to 0 after each optimization step.
threshold = 0.1

# **Set the global polynomial order here**
POLYNOMIAL_ORDER = 1  # Change this to 3, 4, or 5 as needed

# Define the BouncingBallDataset
class BouncingBallDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000):
        self.n_step_per_batch = n_step_per_batch
        self.n_step = n_step
        self.end = n_step * STEP  # Use global STEP
        x_train = self.generate()

        self.down_sample = 1

        self.x_train = torch.tensor(x_train, dtype=dtype)

        # Define variable names
        self.variable_names = ['x', 'y', 'v']

        # Create basis for some stats. Actual basis is in the model
        basis, basis_vars = B.create_library(
            x_train,
            polynomial_order=POLYNOMIAL_ORDER,  # Use the global variable here
            use_trig=False,
            constant=True
        )
        self.basis = torch.tensor(basis)
        self.basis_vars = basis_vars
        self.n_basis = self.basis.shape[1]

    def generate(self):
        g = 9.8
        y_initial = 10.0
        e = 0.75
        time_step = self.end / self.n_step
        v_threshold = 0.1
        x_initial = 0.5

        y = y_initial
        v = 0.0
        x = x_initial
        vx = 0.01

        x_train = []

        for i in range(self.n_step):
            if y <= 0 and v < 0:
                v = -v * e
                vx += np.random.uniform(-0.01, 0.01)

            v -= g * time_step
            y += v * time_step
            x += vx * time_step

            if y < 0:
                y = 0

            if abs(v) < v_threshold and y == 0:
                pass  # You can add a break here if needed

            if x < 0 or x > 1:
                vx = -vx

            # Store state variables: x, y, v
            x_train.append([x, y, v])

        x_train = np.array(x_train)
        return x_train/100

    def __len__(self):
        return (self.n_step - self.n_step_per_batch) // self.down_sample

    def __getitem__(self, idx):
        i = idx * self.down_sample
        d = self.x_train[i: i + self.n_step_per_batch]
        return i, d

# Create dataset instance
ds = BouncingBallDataset(n_step=T, n_step_per_batch=n_step_per_batch)

print("Dataset length:", len(ds))
print("Sample shape:", ds[0][1].shape)

train_loader = DataLoader(
    ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)

# Plot training data
def plot_bouncing_ball(data, filepath):
    x = data[:, 0]
    y = data[:, 1]
    v = data[:, 2]
    t = np.arange(len(x)) * STEP

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(t, x)
    plt.title('x over time')

    plt.subplot(1, 3, 2)
    plt.plot(t, y)
    plt.title('y over time')

    plt.subplot(1, 3, 3)
    plt.plot(t, v)
    plt.title('v over time')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

plot_bouncing_ball(ds.x_train.cpu().numpy(), os.path.join(log_dir, 'train.pdf'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Model class
class Model(nn.Module):
    def __init__(self, bs, n_step, n_step_per_batch, n_basis, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step
        self.order = 2
        self.bs = bs
        self.device = device
        self.n_iv = 1
        self.n_ind_dim = 3  # Number of independent variables (x, y, v)
        self.n_step_per_batch = n_step_per_batch
        self.n_basis = n_basis  # Use n_basis from the parameter

        self.init_xi = torch.randn((1, self.n_basis, self.n_ind_dim), dtype=dtype).to(device)

        self.mask = torch.ones_like(self.init_xi).to(device)

        # Step size is fixed. Make this a parameter for learned step
        self.step_size = (logit(0.01) * torch.ones(1, 1, 1))
        self.param_in = nn.Parameter(torch.randn(1, 64))

        self.ode = ODEINDLayer(
            bs=bs,
            order=self.order,
            n_ind_dim=self.n_ind_dim,
            n_step=self.n_step_per_batch,
            solver_dbl=True,
            double_ret=True,
            n_iv=self.n_iv,
            n_iv_steps=1,
            gamma=0.05,
            alpha=0,
            **kwargs
        )

        self.param_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_basis * self.n_ind_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(self.n_step_per_batch * self.n_ind_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch * self.n_ind_dim)
        )

    def reset_params(self):
        # Reset basis weights to random values
        self.init_xi.data = torch.randn_like(self.init_xi)

    def update_mask(self, mask):
        self.mask = self.mask * mask

    def get_xi(self):
        xi = self.param_net(self.param_in)
        xi = xi.reshape(self.init_xi.shape)
        return xi

    def forward(self, index, net_iv):
        # Apply mask
        xi = self.get_xi()
        xi = self.mask * xi
        xi = xi.repeat(self.bs, 1, 1)

        var = self.net(net_iv.reshape(self.bs, -1))
        var = var.reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)

        # Create basis with matching polynomial order
        var_basis, _ = B.create_library_tensor_batched(
            var, polynomial_order=POLYNOMIAL_ORDER, use_trig=False, constant=True
        )

        rhs = var_basis @ xi
        rhs = rhs.permute(0, 2, 1)

        z = torch.zeros(1, self.n_ind_dim, 1, 1).type_as(net_iv)
        o = torch.ones(1, self.n_ind_dim, 1, 1).type_as(net_iv)

        coeffs = torch.cat([z, o, z], dim=-1)
        coeffs = coeffs.repeat(self.bs, 1, self.n_step_per_batch, 1)

        init_iv = var[:, 0]

        steps = self.step_size.repeat(self.bs, self.n_ind_dim, self.n_step_per_batch - 1).type_as(net_iv)
        steps = torch.sigmoid(steps)

        x0, x1, x2, eps, steps = self.ode(coeffs, rhs, init_iv, steps)
        x0 = x0.permute(0, 2, 1)

        return x0, steps, eps, var, xi

model = Model(
    bs=batch_size,
    n_step=T,
    n_step_per_batch=n_step_per_batch,
    n_basis=ds.n_basis,  # Ensure n_basis matches the dataset's n_basis
    device=device
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

if DBL:
    model = model.double()
model = model.to(device)

# Define the function to print learned equations
def print_eq(stdout=False):
    xi = model.get_xi()
    repr_output = B.basis_repr(
        xi * model.mask,
        ds.basis_vars
    )
    print('repr_output:', repr_output)
    code = []
    if isinstance(repr_output, list):
        # Assume variables are x0, x1, x2
        variable_names = [f'x{idx}' for idx in range(len(repr_output))]
        for idx, eq in enumerate(repr_output):
            var_name = variable_names[idx]
            L.info(f'd{var_name}/dt = {eq}')
            if stdout:
                print(f'd{var_name}/dt = {eq}')
            code.append(f'{eq}')
    elif isinstance(repr_output, dict):
        # Use keys from repr_output as variable names
        for var_name, eq in repr_output.items():
            L.info(f'd{var_name}/dt = {eq}')
            if stdout:
                print(f'd{var_name}/dt = {eq}')
            code.append(f'{eq}')
    else:
        print('Unknown format of repr_output')
    return code

def simulate(gen_code):
    def f(state, t):
        # Define variables based on gen_code length
        variables = {'np': np}
        for idx, value in enumerate(state):
            var_name = f'x{idx}'
            variables[var_name] = value
        derivatives = []
        for idx, eq in enumerate(gen_code):
            derivative = eval(eq, {"__builtins__": None}, variables)
            derivatives.append(derivative)
        return derivatives
    # Initial state corresponding to x0, x1, x2
    state0 = [0.5, 100.0, 0.0]
    time_steps = np.linspace(0, T * STEP, T)
    x_sim = odeint(f, state0, time_steps)
    return x_sim

# Define the training function
def train():
    """Optimize and threshold cycle"""
    model.reset_params()

    max_iter = 150
    for step in range(max_iter):
        print(f'Optimizer iteration {step}/{max_iter}')

        # Thresholding
        if step > 0:
            xi = model.get_xi()
            mask = (xi.abs() > threshold).float()
            model.update_mask(mask)
            model.reset_params()

        code = print_eq(stdout=True)
        # Simulate and plot
        x_sim = simulate(code)
        plot_bouncing_ball(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

        optimize()

# Define the optimization function
def optimize(nepoch=100):
    with tqdm(total=nepoch) as pbar:
        for epoch in range(nepoch):
            pbar.update(1)
            for i, (index, batch_in) in enumerate(train_loader):
                batch_in = batch_in.to(device)

                optimizer.zero_grad()
                x0, steps, eps, var, xi = model(index, batch_in)

                x_loss = (x0 - batch_in).pow(2).mean()
                loss = x_loss + (var - batch_in).pow(2).mean()

                loss.backward()
                optimizer.step()

            xi = xi.detach().cpu().numpy()
            meps = eps.max().item()
            L.info(
                f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} '
            )
            pbar.set_description(
                f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} '
            )

if __name__ == "__main__":
    train()
    print_eq()
