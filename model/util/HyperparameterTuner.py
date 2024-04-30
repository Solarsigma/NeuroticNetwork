from itertools import product
from model.NeuroticNetwork import NeuroticNetwork

class HyperparameterTuner:
    def __init__(self, model_params={}, param_grid={}, scoring=None):
        self.model_params = model_params
        self.param_grid = param_grid
        self.optimum_params = {}
    
    def fit(self, x_train, y_train):
        param_combos = []
        param_grid = self.param_grid
        items = sorted(param_grid.items())
        keys, values = zip(*items)
        for v in product(*values):
            param_combos.append(dict(zip(keys, v)))
        
        best_loss = -1
        for param in param_combos:
            model = NeuroticNetwork(**self.model_params, **param)
            model.train(x_train, y_train)
            curr_loss = model.get_train_loss()
            if best_loss == -1 or curr_loss < best_loss:
                self.optimum_params = param
                best_loss = curr_loss
                print("Optimum params updated!")
        
    def get_optimized_model(self):
        return NeuroticNetwork(**self.model_params, **self.optimum_params)
    
    def get_optimum_params(self):
        return self.optimum_params