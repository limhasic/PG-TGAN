import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.model(x)

class PGTGAN:
    def __init__(self, data_dim, latent_dim):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        d_real = self.discriminator(real_data)
        d_real_loss = self.criterion(d_real, real_labels)
        
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        d_fake = self.discriminator(fake_data.detach())
        d_fake_loss = self.criterion(d_fake, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        g_fake = self.discriminator(fake_data)
        g_loss = self.criterion(g_fake, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for batch in dataloader:
                real_data = batch[0]
                d_loss, g_loss = self.train_step(real_data)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

    def generate(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            generated_data = self.generator(z)
        return generated_data.numpy()

# 데이터 준비 및 전처리 함수
def prepare_data(data, current_dim):
    return data[:, :current_dim]

# 진행형 학습 과정
def progressive_training(model, data, epochs_per_stage, batch_size):
    n_stages = data.shape[1]
    for stage in range(1, n_stages + 1):
        print(f"Training stage {stage}/{n_stages}")
        current_data = prepare_data(data, stage)
        dataset = TensorDataset(torch.FloatTensor(current_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train(dataloader, epochs_per_stage)

# 메인 실행 코드
if __name__ == "__main__":
    # 가상의 데이터 생성 (실제 사용 시 실제 데이터로 대체)
    np.random.seed(0)
    data = np.random.randn(1000, 10)  # 1000개의 샘플, 10개의 특성
    
    latent_dim = 64
    batch_size = 32
    epochs_per_stage = 50
    
    model = PGTGAN(data_dim=data.shape[1], latent_dim=latent_dim)
    progressive_training(model, data, epochs_per_stage, batch_size)
    
    # 데이터 생성
    generated_data = model.generate(100)
    print("Generated data shape:", generated_data.shape)
