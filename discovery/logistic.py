import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from scipy.special import logit
from tqdm import tqdm
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from scipy.special import logit
from tqdm import tqdm
import os
import sys

# Insert the parent directory to sys.path to access custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules (ensure these are correctly defined in your environment)
from extras.source import write_source_files, create_log_dir
from solver.ode_layer import ODEINDLayer
import extras.logger as logger

# Setup logging
log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=True)

# Configuration
DBL = True
dtype = torch.float64 if DBL else torch.float32
STEP = 1
tend = 200
T = int(tend / STEP)

n_step_per_batch = T
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogisticDataset(Dataset):
    def __init__(self, n_step_per_batch=10, n_step=100, train=True):
        data, shared_r, Ks, y0s = self.generate_logistic(N=5000, noise=1e-3)

        data = torch.tensor(data, dtype=dtype)
        shared_r = torch.tensor(shared_r, dtype=dtype)
        Ks = torch.tensor(Ks, dtype=dtype)

        train_len = int(0.8 * data.shape[0])
        if train:
            self.data = data[:train_len]
            self.shared_r = shared_r[:train_len]
            self.Ks = Ks[:train_len]
        else:
            self.data = data[train_len:]
            self.shared_r = shared_r[train_len:]
            self.Ks = Ks[train_len:]

        print('data shape ', self.data.shape)

    def generate_logistic(self, N=5000, noise=1e-3):
        with open('3D_centers5.pkl', 'rb') as f:
            data = pickle.load(f)

        data_len = len(data)
        t = np.arange(1, data_len + 1)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])

        df = pd.DataFrame({'t': t, 'x': x, 'y': y})

        first_valid_index = min(df['x'].first_valid_index(), df['y'].first_valid_index())
        df = df.loc[first_valid_index:].reset_index(drop=True)

        df['x'] = df['x'].interpolate(method='linear') / 1000
        df['y'] = df['y'].interpolate(method='linear') / 1000
        df['x'].fillna(method='bfill', inplace=True)
        df['y'].fillna(method='ffill', inplace=True)

        y = np.array(df['x'].values, dtype=np.float32)
        N = len(y)
        shared_r = np.random.randn(N) * np.random.uniform(0.3, 0.4, N) + np.random.uniform(0.001, 0.04, N)
        Ks = np.ones(N)
        y0s = np.random.uniform(0, 40, N).astype(int)
        ts = np.linspace(1e-6, tend, T)

        y_arrays = [np.pad(y[start_index:start_index + 200], (0, max(0, 200 - len(y[start_index:start_index + 200]))), 'constant')
                    for start_index in y0s]
        traj = np.array(y_arrays) * Ks[:, None] + shared_r[:, None]

        # Normalize traj
        traj_mean = traj.mean(axis=1, keepdims=True)
        traj_std = traj.std(axis=1, keepdims=True) + 1e-6  # Avoid division by zero
        traj = (traj - traj_mean) / traj_std

        return traj, shared_r, Ks, y0s

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.shared_r[idx], self.Ks[idx]


# Initialize datasets and dataloaders
ds = LogisticDataset(n_step=T, n_step_per_batch=n_step_per_batch, train=True)
eval_ds = LogisticDataset(n_step=T, n_step_per_batch=n_step_per_batch, train=False)
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, bs, n_step, n_step_per_batch, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step
        self.order = 2
        self.bs = bs
        self.device = device
        self.n_iv = 1
        self.n_ind_dim = 1
        self.n_step_per_batch = n_step_per_batch
        self.seq_len = self.n_step_per_batch  # sequence length
        self.num_params = 3

        # Initialize the ODEINDLayer (ensure it's properly defined)
        self.ode = ODEINDLayer(
            bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch,
            solver_dbl=True, double_ret=True, n_iv=self.n_iv, n_iv_steps=1, gamma=0.5, alpha=0, **kwargs
        )

        pm = 'zeros'
        self.cf_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 1, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.Tanh()  # Added activation to bound outputs
        )

        self.param_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.Flatten(),
            nn.Linear(self.seq_len * 128, self.num_params),
            nn.Tanh()  # Added activation to bound outputs
        )

        self.step_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.ELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1, padding_mode=pm),
            nn.Flatten(),
        )
        self.step_mlp = nn.Linear(self.seq_len * 128, 1)

        step_bias_t = logit(0.05)
        self.step_mlp.weight.data.fill_(0.0)
        self.step_mlp.bias.data.fill_(step_bias_t)

    def forward(self, x):
        # Dynamically compute the batch size
        current_batch_size = x.size(0)  # Get the actual batch size
        x_reshaped = x.reshape(current_batch_size, -1).unsqueeze(1)

        # Compute var
        var = self.cf_cnn(x_reshaped)
        var = var.reshape(current_batch_size, self.n_step_per_batch)

        # Check for NaNs in var
        if torch.isnan(var).any():
            print('Warning: var contains NaNs')

        # Compute parameters
        params = self.param_cnn(x_reshaped)
        params = params.reshape(-1, self.num_params, 1)

        # Clamp var and params to prevent numerical issues
        var = var.clamp(min=-1.0, max=1.0)
        params = params.clamp(min=-1.0, max=1.0)

        # Compute rhs
        rhs = params[:, 0] * torch.ones_like(var) + params[:, 1] * var + params[:, 2] * (var ** 2)

        # Check for NaNs in rhs
        if torch.isnan(rhs).any():
            print('Warning: rhs contains NaNs')

        # Prepare coefficients
        z = torch.zeros(1, self.n_ind_dim, 1, 1).type_as(x)
        o = torch.ones(1, self.n_ind_dim, 1, 1).type_as(x)
        coeffs = torch.cat([z, o, z], dim=-1).repeat(current_batch_size, 1, self.n_step_per_batch, 1)

        # Initial condition
        init_iv = var[:, 0]

        # Compute steps
        steps = self.step_cnn(x_reshaped)
        steps = self.step_mlp(steps).reshape(current_batch_size, 1, 1).repeat(1, self.n_ind_dim, self.n_step_per_batch - 1).type_as(x)
        steps = torch.sigmoid(steps).clip(min=0.01, max=0.1)  # Adjusted clipping

        # Pass through ODE layer
        x0, x1, x2, eps, steps = self.ode(coeffs, rhs, init_iv, steps)

        # Reshape output
        x0 = x0.permute(0, 2, 1).squeeze()

        # Check for NaNs in x0
        if torch.isnan(x0).any():
            print('Warning: x0 contains NaNs')

        return x0, steps, eps.abs().max(), var, params


# Remaining functions are unchanged


# Initialize the model
model = Model(bs=batch_size, n_step=T, n_step_per_batch=n_step_per_batch, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if DBL:
    model = model.double()
model = model.to(device)


def loss_func(x, y):
    return (x - y).abs().sum(dim=-1).mean()


def train_step(batch):
    model.train()
    data, shared_r, Ks = batch
    data = data.to(device)
    shared_r = shared_r.to(device)
    Ks = Ks.to(device)
    optimizer.zero_grad()

    # Forward pass
    x0, steps, eps, var, xi = model(data)

    # Check for NaNs in outputs
    if torch.isnan(x0).any():
        print('Warning: x0 contains NaNs during training')
    if torch.isnan(var).any():
        print('Warning: var contains NaNs during training')
    if torch.isnan(xi).any():
        print('Warning: xi contains NaNs during training')

    # Compute losses
    x_loss = loss_func(x0, data)
    v_loss = loss_func(var, x0)
    loss = x_loss + v_loss

    # Check for NaNs in loss
    if torch.isnan(loss).any():
        print('Warning: loss contains NaNs during training')

    # Backward pass and optimization
    loss.backward()

    # Optional: Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {'loss': loss.item(), 'x_loss': x_loss.item(), 'v_loss': v_loss.item(),
            'x0': x0, 'eps': eps, 'var': var, 'xi': xi}


def eval_step(batch):
    model.eval()
    data, shared_r, Ks = batch
    data = data.to(device)
    shared_r = shared_r.to(device)
    Ks = Ks.to(device)

    with torch.no_grad():
        # Forward pass
        x0, steps, eps, var, xi = model(data)

    # Compute losses
    x_loss = loss_func(x0, data)
    v_loss = loss_func(var, x0)
    loss = x_loss + v_loss

    # Check for NaNs in outputs
    if torch.isnan(x0).any():
        print('Warning: x0 contains NaNs during evaluation')
    if torch.isnan(var).any():
        print('Warning: var contains NaNs during evaluation')

    return {'loss': loss.item(), 'x_loss': x_loss.item(), 'v_loss': v_loss.item(),
            'x0': x0, 'eps': eps, 'var': var, 'xi': xi}


def evaluate():
    losses = []
    for batch in eval_loader:
        loss_dict = eval_step(batch)
        losses.append(loss_dict['loss'])
    if losses:
        eval_loss = np.mean(losses)
        L.info(f'eval loss {eval_loss}')
        return eval_loss
    else:
        L.info('No evaluation batches found.')
        return float('nan')


def train_model(nepoch=100):
    with tqdm(total=nepoch) as pbar:
        eval_loss = float('inf')
        for epoch in range(nepoch):
            pbar.update(1)
            losses = []
            for batch in train_loader:
                loss_dict = train_step(batch)
                losses.append(loss_dict['loss'])
            train_loss = np.mean(losses)
            pbar.set_description(f'epoch {epoch}, loss {train_loss:.4f}')
            if epoch % 10 == 0:
                _eval_loss = evaluate()
                if not np.isnan(_eval_loss) and _eval_loss < eval_loss:
                    eval_loss = _eval_loss
                    torch.save(model.state_dict(), os.path.join(log_dir, 'model.ckpt'))
                    L.info(f'Model saved at epoch {epoch} with eval loss {eval_loss}')


if __name__ == "__main__":
    train_model()
