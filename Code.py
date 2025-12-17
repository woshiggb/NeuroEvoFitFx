import numpy as np
import random
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim

def F_(X, X2):
    return 1/(1+abs(X+X2))**2

class BaseModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.rest_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._update_weights()
    
    def forward(self, x):
        return self.rest_layers(self.first_layer(x))
    
    def _update_weights(self):
        for name, param in self.first_layer.named_parameters():
            if 'weight' in name:
                self.first_weights = param.data.clone().cpu().numpy().flatten()
                break
    
    def get_first_params(self):
        return self.first_weights
    
    def set_first_params(self, params):
        params = params.astype(np.float32)
        expected = self.input_size * self.hidden_size
        if len(params) != expected:
            raise ValueError(f"参数数量不正确，期望 {expected}，得到 {len(params)}")
        
        with torch.no_grad():
            for name, param in self.first_layer.named_parameters():
                if 'weight' in name:
                    param.data = torch.tensor(
                        params.reshape(self.hidden_size, self.input_size),
                        dtype=torch.float32
                    )
                    break
        self._update_weights()

def calculate_x1_x2(model_a, model_b, error_a, error_b):
    min_error = min(error_a, error_b)
    if min_error < 0:
        adjustment = abs(min_error)
        error_a += adjustment
        error_b += adjustment
    
    params_a = model_a.get_first_params()
    params_b = model_b.get_first_params()
    
    diff_list = []
    for i in range(len(params_a)):
        diff = abs(params_a[i] - params_b[i])
        if diff < 1e-10:
            diff = 1e-10
        diff_list.append(diff)
    
    log_sum = sum(np.log10(diff) for diff in diff_list)
    logs_value = np.tanh(log_sum / len(diff_list))
    
    denominator = abs(error_a) + abs(error_b)
    if denominator == 0:
        error_ratio = 0.5
    else:
        error_ratio = error_a / denominator
    
    X1 = error_ratio
    X2 = (1 - error_ratio) * logs_value
    
    X1 = max(0.0, min(1.0, X1))
    X2 = max(0.0, min(1.0, X2))
    
    total = X1 + X2
    if total > 0:
        X1 /= total
        X2 /= total
    
    return X1, X2

class Ecosystem:
    def __init__(self, num_models=10, input_size=10, hidden_size=64, output_size=1, seed=42):
        self.num_models = num_models
        self.input_size = input_size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.set_default_dtype(torch.float32)
        
        self.models = [BaseModel(input_size, hidden_size, output_size) for _ in range(num_models)]
        self.data_pool = self._generate_data(1000)
        self.history = []
    
    def _generate_data(self, size):
        X = np.random.randn(size, self.input_size).astype(np.float32)
        
        if self.input_size >= 2:
            y = F(X[:, 0], X[:, 1])
        else:
            y = F(X[:, 0], np.zeros_like(X[:, 0]))
        
        noise = np.random.randn(size, 1).astype(np.float32) * 0.1
        y = y.reshape(-1, 1) + noise
        
        return X, y
    
    def train_model(self, model, X_train, y_train, epochs=200, lr=0.01):
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
        
        model._update_weights()
        return model
    
    def evaluate_model(self, model, X, y):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            predictions = model(X_tensor)
            loss = nn.MSELoss()(predictions, y_tensor)
        return loss.item()
    
    def run_generation(self, generation):
        X, y = self.data_pool
        n = len(X)
        idx = int(n * 0.7)
        indices = np.random.permutation(n)
        
        X_train, y_train = X[indices[:idx]], y[indices[:idx]]
        X_test, y_test = X[indices[idx:]], y[indices[idx:]]
        
        results = []
        for i, model in enumerate(self.models):
            try:
                trained = self.train_model(copy.deepcopy(model), X_train, y_train)
                train_err = self.evaluate_model(trained, X_train, y_train)
                test_err = self.evaluate_model(trained, X_test, y_test)
                
                min_error = min(train_err, test_err)
                if min_error < 0:
                    adj = abs(min_error)
                    train_err += adj
                    test_err += adj
                
                avg_err = (train_err + test_err) / 2
                denominator = max(0.5 * 0.5, 1e-10)
                score = max(train_err, test_err) * (train_err + test_err) / denominator
                
                results.append({
                    'train_err': train_err,
                    'test_err': test_err,
                    'avg_err': avg_err,
                    'score': score,
                    'model': trained
                })
            except Exception as e:
                results.append({
                    'train_err': float('inf'),
                    'test_err': float('inf'),
                    'avg_err': float('inf'),
                    'score': float('inf'),
                    'model': model
                })
        
        results.sort(key=lambda x: x['score'])
        self.history.append(results)
        return results
    
    def genetic_operation(self, model_a, model_b, X1, X2):
        new_model = BaseModel(self.input_size, model_a.hidden_size, 1)
        params_a = model_a.get_first_params()
        params_b = model_b.get_first_params()
        new_params = X1 * params_a + X2 * params_b
        new_model.set_first_params(new_params)
        
        with torch.no_grad():
            for (name_a, param_a), (name_new, param_new) in zip(
                model_a.rest_layers.named_parameters(),
                new_model.rest_layers.named_parameters()
            ):
                param_new.data = param_a.data.clone()
        
        return new_model
    
    def evolve_population(self, results):
        n_models = len(results)
        n_keep = max(1, int(n_models * 0.3))
        new_models = [results[i]['model'] for i in range(n_keep)]
        
        probabilities = []
        for i in range(n_models):
            rank = i + 1
            probabilities.append(1 / rank)
        
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            probabilities = [1 / n_models] * n_models
        
        while len(new_models) < self.num_models:
            try:
                target_idx = np.random.choice(range(n_keep))
                selected_idx = np.random.choice(range(n_models), p=probabilities)
                
                target_model = new_models[target_idx]
                selected_model = results[selected_idx]['model']
                target_err = results[target_idx]['avg_err']
                selected_err = results[selected_idx]['avg_err']
                
                X1, X2 = calculate_x1_x2(target_model, selected_model, target_err, selected_err)
                child = self.genetic_operation(target_model, selected_model, X1, X2)
                new_models.append(child)
            except Exception as e:
                new_models.append(BaseModel(self.input_size, self.models[0].hidden_size, 1))
        
        self.models = new_models
    
    def run_evolution(self, generations=10):
        for gen in range(generations):
            results = self.run_generation(gen)
            
            valid = [r for r in results if r['score'] != float('inf')]
            if valid:
                best = valid[0]
                avg_score = np.mean([r['score'] for r in valid])
                
                print(f"Gen {gen+1}: Best={best['score']:.4f}, Avg={avg_score:.4f}, "
                      f"Train={best['train_err']:.4f}, Test={best['test_err']:.4f}")
            
            if gen < generations - 1:
                self.evolve_population(results)
        
        return self.history
    
    def get_best_model(self):
        if not self.history:
            return None, None
        
        all_results = [r for gen in self.history for r in gen if r['score'] != float('inf')]
        if not all_results:
            return None, None
        
        best = min(all_results, key=lambda x: x['score'])
        return best['model'], best

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    eco = Ecosystem(num_models=12, input_size=10, hidden_size=64)
    
    print("开始进化训练...")
    eco.run_evolution(generations=50)
    
    best_model, best_result = eco.get_best_model()
    
    if best_model is not None:
        print(f"\n最佳模型结果:")
        print(f"训练误差: {best_result['train_err']:.6f}")
        print(f"测试误差: {best_result['test_err']:.6f}")
        print(f"综合分数: {best_result['score']:.6f}")
        
        X_full, y_full = eco.data_pool
        full_error = eco.evaluate_model(best_model, X_full, y_full)
        print(f"完整数据集误差: {full_error:.6f}")
    else:
        print("未找到有效模型")

if __name__ == "__main__":
    F = F_
    main()
