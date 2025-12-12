import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class BaseModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.rest_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
    def forward(self, x):
        return self.rest_layers(self.first_layer(x))
    
    def get_first_params(self):
        return next(self.first_layer.parameters()).detach().flatten()
    
    def set_first_params(self, params):
        with torch.no_grad():
            next(self.first_layer.parameters()).copy_(
                params.reshape(next(self.first_layer.parameters()).shape)
            )

def calculate_x1_x2(model_a, model_b, error_a, error_b):
    diff = torch.abs(model_a.get_first_params() - model_b.get_first_params())
    diff = torch.clamp(diff, 1e-10)
    logs_value = torch.log10(diff).mean()
    
    error_sum = error_a + error_b
    if error_sum == 0:
        X1 = 0.5
    else:
        X1 = logs_value * error_a / error_sum
    
    return torch.clamp(torch.tensor([X1, 1-X1]), 0, 1)

class Ecosystem:
    def __init__(self, num_models=10, input_size=10, hidden_size=64):
        self.num_models = num_models
        self.models = [BaseModel(input_size, hidden_size) for _ in range(num_models)]
        self.X, self.y = self._generate_data(1000)
        self.history = []
    
    def _generate_data(self, size):
        X = np.random.randn(size, 10).astype(np.float32)
        y = (X[:, :5]**2).sum(axis=1, keepdims=True) + \
            np.sin(X[:, 5:].sum(axis=1, keepdims=True)) + \
            np.random.randn(size, 1).astype(np.float32) * 0.1
        return X, y
    
    def run_generation(self):
        n = len(self.X)
        idx = int(n * 0.7)
        indices = np.random.permutation(n)
        X_train, y_train = self.X[indices[:idx]], self.y[indices[:idx]]
        X_test, y_test = self.X[indices[idx:]], self.y[indices[idx:]]
        
        results = []
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            X_t, y_t = torch.tensor(X_train), torch.tensor(y_train)
            
            for _ in range(200):  # 训练200个epoch
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(X_t), y_t)
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                train_err = nn.MSELoss()(model(X_t), y_t).item()
                test_err = nn.MSELoss()(model(torch.tensor(X_test)), 
                                      torch.tensor(y_test)).item()
            
            score = (train_err + test_err) * (1 + abs(train_err - test_err))
            results.append({'score': score, 'train_err': train_err, 
                           'test_err': test_err, 'model': model})
        
        results.sort(key=lambda x: x['score'])
        self.history.append(results)
        return results
    
    def evolve(self, results):
        n_keep = max(1, int(self.num_models * 0.3))
        new_models = [r['model'] for r in results[:n_keep]]
        
        while len(new_models) < self.num_models:
            i, j = np.random.choice(len(results), 2, replace=True)
            parent_a, parent_b = results[i]['model'], results[j]['model']
            
            X1, X2 = calculate_x1_x2(parent_a, parent_b, 
                                    results[i]['score'], results[j]['score'])
            
            child = BaseModel()
            child_params = X1 * parent_a.get_first_params() + \
                          X2 * parent_b.get_first_params()
            child.set_first_params(child_params)
            new_models.append(child)
        
        self.models = new_models
    
    def run_evolution(self, generations=5):
        for gen in range(generations):
            results = self.run_generation()
            best = results[0]
            avg = np.mean([r['score'] for r in results])
            
            print(f"Gen {gen+1}: Best={best['score']:.4f}, "
                  f"Avg={avg:.4f}, Train={best['train_err']:.4f}, "
                  f"Test={best['test_err']:.4f}")
            
            if gen < generations - 1:
                self.evolve(results)

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    
    eco = Ecosystem(num_models=12)
    eco.run_evolution(generations=50)
    
    best = min(eco.history[-1], key=lambda x: x['score'])
    print(f"\nFinal - Train: {best['train_err']:.4f}, "
          f"Test: {best['test_err']:.4f}, Score: {best['score']:.4f}")

if __name__ == "__main__":
    main()
