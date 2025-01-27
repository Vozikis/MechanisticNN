import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Load all trajectories
base_directory = "Videos/falling_ball/09_10_2024/fps_30"
num_trajectories = 11
file_name = "3D_centers.pkl"
all_trajectories = []

for trajectory_number in range(num_trajectories):
    video_dir = os.path.join(base_directory, f"video_{trajectory_number}")
    file_path = os.path.join(video_dir, file_name)
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        data_len = len(data)
        t = np.arange(data_len)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])
        
        valid_start_idx_x = np.where(~np.isnan(x))[0][0]
        x = x[valid_start_idx_x:]
        y = y[valid_start_idx_x:]
        t = t[valid_start_idx_x:]
        
        df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
        traj_x = -df['x'][:].to_numpy() / 100
        traj_y = -df['y'][:].to_numpy() / 100
        traj_t = df['t'][:].to_numpy()
        
        if trajectory_number == 10:
            traj_x = -df['x'][:20].to_numpy() / 100

        trajectory = traj_x.copy()
        all_trajectories.append(trajectory)

global_len = 0

class RealDataset(Dataset):
    def __init__(self, trajectory_data):
        global global_len

        self.y = torch.tensor(trajectory_data, dtype=torch.float32)
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
        return torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=True)


class ComplexModel(nn.Module):
    def __init__(self, bs=1, device=None):
        super().__init__()

        self.step_size = 0.1
        self.end = global_len * self.step_size
        self.n_step = int(self.end / self.step_size)
        self.order = 2  
        self.n_dim = 1 
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
            self.iv_rhs = nn.Parameter(iv_rhs)
        else:
            self.iv_rhs = None

        self.steps = torch.logit(self.step_size * torch.ones(1, self.n_step - 1, self.n_dim, dtype=dtype))
        self.steps = nn.Parameter(self.steps)

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

        return rhs, coeffs, iv_rhs, steps

predicted_trajectories = {}

class Method(pl.LightningModule):
    def __init__(self, trajectory_index):
        super().__init__()
        self.learning_rate = 0.0001
        self.model = ComplexModel(device=self.device)
        self.trajectory_index = trajectory_index  
        
    def forward(self, y):
        return self.model(y)
    
    def training_step(self, batch, batch_idx):
        y = batch
        rhs, coeffs, iv_rhs, steps = self.model(y)
        loss = (rhs - y).pow(2).mean()
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        if self.current_epoch % 100 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            self.plot_and_save_trajectory(y, rhs)
        
        if self.current_epoch == self.trainer.max_epochs - 1:
            predicted_trajectories[self.trajectory_index] = rhs.detach().cpu().numpy().flatten()
        
        return {"loss": loss}

    def plot_and_save_trajectory(self, real_y, predicted_rhs):
        real_y = real_y.detach().cpu().numpy().flatten()
        predicted_y = predicted_rhs.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 6))
        plt.plot(real_y, label="Real Trajectory", linestyle="-", color="blue")
        plt.plot(predicted_y, label="Predicted Trajectory", linestyle="--", color="orange")
        plt.xlabel("Time Steps")
        plt.ylabel("Trajectory")
        plt.title(f"Predicted vs Real Trajectory (Trajectory {self.trajectory_index + 1})")
        plt.legend()
        plt.grid(True)

        output_dir = f"trajectory_plots/trajectory_{self.trajectory_index + 1}"
        os.makedirs(output_dir, exist_ok=True)

        plot_path = os.path.join(output_dir, f"epoch_{self.current_epoch}.png")
        plt.savefig(plot_path)
        plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

def plot_all_trajectories():
    plt.figure(figsize=(12, 8))
    for i, predicted_y in predicted_trajectories.items():
        plt.plot(predicted_y, label=f"Trajectory {i + 1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Trajectory")
    plt.title("Discovered Trajectories")
    plt.legend()
    plt.grid(True)
    
    output_dir = "trajectory_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "all_trajectories.png"))
    plt.close()

def train():
    global all_trajectories

    for i, trajectory in enumerate(all_trajectories):
        print(f"Training on trajectory {i + 1}/{len(all_trajectories)}...")

        dataset = RealDataset(trajectory_data=trajectory)
        datamodule = RealDataModule(dataset=dataset)

        method = Method(trajectory_index=i)
        trainer = pl.Trainer(
            max_epochs=2000,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            log_every_n_steps=1,
        )
        
        trainer.fit(method, datamodule=datamodule)
        
        print(f"Finished training on trajectory {i + 1}/{len(all_trajectories)}.")

    plot_all_trajectories()

if __name__ == "__main__":
    train()
