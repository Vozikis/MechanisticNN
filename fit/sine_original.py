
import torch.nn as nn
import torch
import pandas as pd
from solver.ode_layer import ODEINDLayer
from torch.nn.parameter import Parameter
import numpy as np
import pickle
import torch
import ipdb
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import sympy as sp 
from scipy.integrate import odeint
import os
from torch.utils.data import Dataset








#------------------------------------------------------------------------------
#Working on this

import torch.nn as nn
import torch
import pandas as pd
from solver.ode_layer import ODEINDLayer 
from torch.nn.parameter import Parameter
import numpy as np
import pickle
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
global_len = 0
class RealDataset(Dataset):
    def __init__(self):
        global global_len
        # with open('3D_centers16.pkl', 'rb') as f:
        with open('3D_centers16.pkl', 'rb') as f:
            data = pickle.load(f)
        
        data_len = len(data)
        t = np.arange(1, data_len + 1)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])
        
        df = pd.DataFrame({'t': t, 'x': x, 'y': y})

        first_valid_index_x = df['x'].first_valid_index()
        first_valid_index_y = df['y'].first_valid_index()
        first_valid_index = min(first_valid_index_x, first_valid_index_y)
        
        df = df.loc[first_valid_index:].reset_index(drop=True)

        df['x'] = df['x'].interpolate(method='linear') / 1000
        df['y'] = df['y'].interpolate(method='linear') /1000

        df['x'].fillna(method='bfill', inplace=True)
        df['x'].fillna(method='ffill', inplace=True)
        df['y'].fillna(method='bfill', inplace=True)
        df['y'].fillna(method='ffill', inplace=True)

        self.y = torch.tensor(df['x'].values, dtype=torch.float32)
        global_len = len(self.y)

    def __len__(self):
        return 1  

    def __getitem__(self, idx):
        return self.y 

class RealDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=True)
        return train_loader 

class Method(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.0001
        _ = RealDataset()
        self.model = ComplexModel(device=self.device)
        
        self.func_list = []
        self.y_list = []
        self.funcp_list = []
        self.funcpp_list = []
        self.steps_list = []

    def forward(self, y):
        return self.model(y)
    
    def training_step(self, batch, batch_idx):
        y = batch
        eps, u0, u1, u2, steps = self(y)
        
        loss = (u0 - y).pow(2).mean()
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('eps', eps, prog_bar=True, logger=True)
        
        self.func_list.append(u0.detach().cpu().numpy())
        self.funcp_list.append(u1.detach().cpu().numpy())
        self.funcpp_list.append(u2.detach().cpu().numpy())
        self.steps_list.append(steps.detach().cpu().numpy())
        
        self.y_list.append(y.detach().cpu().numpy())
        
        return {"loss": loss, 'y': y, 'u0': u0, 'u1': u1, 'u2': u2}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_end(self):
        coeffs = self.model.coeffs.detach().cpu().numpy()
        print("Learned Coefficients:", coeffs)
        d = self.model.rhs.mean().item()
        reald = self.model.rhs[-1][0][:]

        a = coeffs[2]  # Coefficient for u''(t)
        b = coeffs[1]  # Coefficient for u'(t)
        c = coeffs[0]  # Coefficient for u(t)

        ode_str = f"{a:.4f} * u''(t) + {b:.4f} * u'(t) + {c:.4f} * u(t) + {d:.4f} = 0"
        print("Learned ODE:", ode_str)

        if self.y_list and self.func_list:
            try:
                y_true = self.y_list[0].reshape(-1)
                y_pred = self.func_list[-1].reshape(-1)

                step_size = self.model.step_size
                t = np.arange(len(y_true)) * step_size

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(t, y_true, label='True y', linestyle='-', marker='o')
                ax.plot(t, y_pred, label='Learned', linestyle='--', marker='x')

                ax.text(0.95, 0.6, f"Learned ODE:\n{ode_str}", transform=ax.transAxes,
                        fontsize=10, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))

                ax.set_xlabel('Time')
                ax.set_ylabel('y')
                ax.set_title('True y vs Learned')
                ax.legend()
                ax.grid(True)

                plt.tight_layout(rect=[0, 0, 0.85, 1])

                save_path = 'learned_equation_plot.png'
                plt.savefig(save_path)
                print(f"Plot saved as '{save_path}'.")

                plt.close(fig)
                print("Plot closed.")
                return a,b,c,reald
            except Exception as e:
                print(f"Error while plotting: {e}")
        else:
            print("No data available to plot.")

class ComplexModel(nn.Module):
    def __init__(self, bs=1, device=None):
        super().__init__()

        self.step_size = 0.1
        self.end = global_len * self.step_size
        self.n_step = int(self.end / self.step_size)
        self.order = 2  # Second-order ODE
        self.n_dim = 1  # State dimension
        self.bs = bs
        self.device = device
        dtype = torch.float32

        self.cf_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
        )

        self.coeffs = nn.Parameter(torch.randn(self.order + 1, dtype=dtype))

        self.param_in = nn.Parameter(torch.randn(1, 64, dtype=dtype))

        self.n_iv = 1 
        if self.n_iv > 0:
            iv_rhs = torch.zeros(self.n_dim, self.n_iv, dtype=dtype)
            self.iv_rhs = Parameter(iv_rhs)
        else:
            self.iv_rhs = None

        self.steps = torch.logit(self.step_size * torch.ones(1, self.n_step - 1, self.n_dim, dtype=dtype))
        self.steps = nn.Parameter(self.steps)

        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_dim, n_iv=self.n_iv,
                               n_step=self.n_step, n_iv_steps=1)

    def forward(self, y):
        bs = y.size(0)
        dtype = y.dtype
        device = y.device

        rhs = self.cf_cnn(y.view(bs, 1, -1))  # Output shape: (bs, 1, n_step)
        rhs = rhs.view(bs, self.n_dim, self.n_step)

        coeffs = self.coeffs.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, order + 1)
        coeffs = coeffs.expand(bs, self.n_dim, self.n_step, self.order + 1)

        iv_rhs = self.iv_rhs
        if self.n_iv > 0:
            iv_rhs = iv_rhs.unsqueeze(0).repeat(bs, 1, 1)

        steps = torch.sigmoid(self.steps)
        steps = steps.repeat(bs, 1, 1)

        u0, u1, u2, eps, steps = self.ode(coeffs, rhs, iv_rhs, steps)

        self.coeffs_mean = self.coeffs 

        self.rhs = rhs

        return eps, u0, u1, u2, steps

method = Method()

def train():
    dataset = RealDataset()
    datamodule = RealDataModule(dataset=dataset)

    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            # Add any callbacks you need
        ],
        log_every_n_steps=1,
    )
    trainer.fit(method, datamodule=datamodule)

if __name__ == "__main__":
    train()


#------------------------------------------------------------------------------

# import torch.nn as nn
# import torch
# import pandas as pd
# from solver.ode_layer import ODEINDLayer 
# from torch.nn.parameter import Parameter
# import numpy as np
# import pickle
# import pytorch_lightning as pl
# import matplotlib.pyplot as plt
# import os
# from torch.utils.data import Dataset

# global_len = 313

# class RealDataset(Dataset):
#     def __init__(self):
#         with open('3D_centers6.pkl', 'rb') as f:
#         # with open('3D_centers_pend2.pkl', 'rb') as f:
#             data = pickle.load(f)
        
#         data_len = len(data)
#         t = np.arange(1, data_len + 1)
#         x = np.array([item[0] if item is not None else np.nan for item in data])
#         y = np.array([item[1] if item is not None else np.nan for item in data])
        
#         df = pd.DataFrame({'t': t, 'x': x, 'y': y})

#         first_valid_index_x = df['x'].first_valid_index()
#         first_valid_index_y = df['y'].first_valid_index()
#         first_valid_index = min(first_valid_index_x, first_valid_index_y)
        
#         df = df.loc[first_valid_index:].reset_index(drop=True)

#         df['x'] = df['x'].interpolate(method='linear') / 1000
#         df['y'] = df['y'].interpolate(method='linear') /1000

#         df['x'].fillna(method='bfill', inplace=True)
#         df['x'].fillna(method='ffill', inplace=True)
#         df['y'].fillna(method='bfill', inplace=True)
#         df['y'].fillna(method='ffill', inplace=True)

#         self.y = torch.tensor(df['x'].values, dtype=torch.float32)

#     def __len__(self):
#         return 1  

#     def __getitem__(self, idx):
#         return self.y 

# class RealDataModule(pl.LightningDataModule):
#     def __init__(self, dataset):
#         super().__init__()
#         self.dataset = dataset

#     def train_dataloader(self):
#         train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=True)
#         return train_loader 

# class Method(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.learning_rate = 0.0001
#         self.model = ComplexModel(device=self.device)
        
#         self.func_list = []
#         self.y_list = []
#         self.funcp_list = []
#         self.funcpp_list = []
#         self.steps_list = []

#     def forward(self, y):
#         return self.model(y)
    
#     def training_step(self, batch, batch_idx):
#         y = batch
#         eps, u0, u1, u2, steps, rhs, coeeefs = self(y)
#         a = coeeefs[0][0][-1][2]
#         b = coeeefs[0][0][-1][1]
#         c = coeeefs[0][0][-1][0]
        
#         loss1 = (u0 - y).pow(2).mean()
#         loss2 = (c*u0+b*u1+a*u2+rhs).pow(2).mean()
#         loss = loss1 + loss2
        
#         self.log('train_loss', loss, prog_bar=True, logger=True)
#         self.log('eps', eps, prog_bar=True, logger=True)
        
#         self.func_list.append(u0.detach().cpu().numpy())
#         self.funcp_list.append(u1.detach().cpu().numpy())
#         self.funcpp_list.append(u2.detach().cpu().numpy())
#         self.steps_list.append(steps.detach().cpu().numpy())
        
#         self.y_list.append(y.detach().cpu().numpy())
        
#         return {"loss": loss, 'y': y, 'u0': u0, 'u1': u1, 'u2': u2}

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         return optimizer

#     def on_train_end(self):
#         coeffs = self.model.coeffs.detach().cpu().numpy()
#         print("Learned Coefficients:", coeffs)
#         d = self.model.d.item()
#         print("Learned constant term d:", d)

#         a = coeffs[2]  # Coefficient for u''(t)
#         b = coeffs[1]  # Coefficient for u'(t)
#         c = coeffs[0]  # Coefficient for u(t)

#         ode_str = f"{a:.4f} * u''(t) + {b:.4f} * u'(t) + {c:.4f} * u(t) + {d:.4f} = 0"
#         print("Learned ODE:", ode_str)

#         if self.y_list and self.func_list:
#             try:
#                 y_true = self.y_list[0].reshape(-1)
#                 y_pred = self.func_list[-1].reshape(-1)

#                 step_size = self.model.step_size
#                 t = np.arange(len(y_true)) * step_size

#                 fig, ax = plt.subplots(figsize=(12, 8))
#                 ax.plot(t, y_true, label='True y', linestyle='-', marker='o')
#                 ax.plot(t, y_pred, label='Learned', linestyle='--', marker='x')

#                 ax.text(0.95, 0.6, f"Learned ODE:\n{ode_str}", transform=ax.transAxes,
#                         fontsize=10, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))

#                 ax.set_xlabel('Time')
#                 ax.set_ylabel('y')
#                 ax.set_title('True y vs Learned')
#                 ax.legend()
#                 ax.grid(True)

#                 plt.tight_layout(rect=[0, 0, 0.85, 1])

#                 save_path = 'learned_equation_plot.png'
#                 plt.savefig(save_path)
#                 print(f"Plot saved as '{save_path}'.")

#                 plt.close(fig)
#                 print("Plot closed.")
#                 return a, b, c, d, coeffs
#             except Exception as e:
#                 print(f"Error while plotting: {e}")
#         else:
#             print("No data available to plot.")

# class ComplexModel(nn.Module):
#     def __init__(self, bs=1, device=None):
#         super().__init__()

#         self.step_size = 0.1
#         self.end = global_len * self.step_size
#         self.n_step = int(self.end / self.step_size)
#         self.order = 2  # Second-order ODE
#         self.n_dim = 1  # State dimension
#         self.bs = bs
#         self.device = device
#         dtype = torch.float32

#         self.cf_cnn = nn.Sequential(
#             nn.Conv1d(1, 64, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.Conv1d(64, 128, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.Conv1d(128, 256, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.Conv1d(256, 256, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.Conv1d(256, 128, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.Conv1d(128, 64, kernel_size=5, padding=2),
#             nn.ELU(),
#             nn.Conv1d(64, 1, kernel_size=5, padding=2),
#         )

#         self.coeffs = nn.Parameter(torch.randn(self.order + 1, dtype=dtype))

#         self.d = nn.Parameter(torch.randn(1, dtype=dtype))  # Added here

#         self.param_in = nn.Parameter(torch.randn(1, 64, dtype=dtype))

#         self.n_iv = 1 
#         if self.n_iv > 0:
#             iv_rhs = torch.zeros(self.n_dim, self.n_iv, dtype=dtype)
#             self.iv_rhs = Parameter(iv_rhs)
#         else:
#             self.iv_rhs = None

#         self.steps = torch.logit(self.step_size * torch.ones(1, self.n_step - 1, self.n_dim, dtype=dtype))
#         self.steps = nn.Parameter(self.steps)

#         self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_dim, n_iv=self.n_iv,
#                                n_step=self.n_step, n_iv_steps=1)

#     def forward(self, y):
#         bs = y.size(0)
#         dtype = y.dtype
#         device = y.device

#         rhs = self.cf_cnn(y.view(bs, 1, -1)) - self.d  # Subtract d from the output
#         rhs = rhs.view(bs, self.n_dim, self.n_step)

#         coeffs = self.coeffs.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, order + 1)
#         coeffs = coeffs.expand(bs, self.n_dim, self.n_step, self.order + 1)

#         iv_rhs = self.iv_rhs
#         if self.n_iv > 0:
#             iv_rhs = iv_rhs.unsqueeze(0).repeat(bs, 1, 1)

#         steps = torch.sigmoid(self.steps)
#         steps = steps.repeat(bs, 1, 1)

#         u0, u1, u2, eps, steps = self.ode(coeffs, rhs, iv_rhs, steps)

#         self.coeffs_mean = self.coeffs 

#         self.rhs = rhs

#         return eps, u0, u1, u2, steps, rhs, coeffs

# method = Method()

# def train():
#     dataset = RealDataset()
#     datamodule = RealDataModule(dataset=dataset)

#     trainer = pl.Trainer(
#         max_epochs=1000,
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1,
#         callbacks=[
#             # Add any callbacks you need
#         ],
#         log_every_n_steps=1,
#     )
#     trainer.fit(method, datamodule=datamodule)

# if __name__ == "__main__":
#     train()


