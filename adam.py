import numpy as np

#Adam Optimizer
class Adam():
    def __init__(self, layer_sizes,
                 optimizer_info):
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]

        for i in range(0, len(self.layer_sizes) - 1):
            self.m[i]["W"] = np.zeros(
                (self.layer_sizes[i], self.layer_sizes[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i+1]))
            self.v[i]["W"] = np.zeros(
                (self.layer_sizes[i], self.layer_sizes[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i+1]))

        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, td_errors_times_gradients):
        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.beta_m*self.m[i][param] + \
                    (1-self.beta_m)*td_errors_times_gradients[i][param]
                self.v[i][param] = self.beta_v*self.v[i][param] + \
                    (1-self.beta_v) * \
                    np.power(td_errors_times_gradients[i][param], 2)
                m_hat = self.m[i][param] / (1-self.beta_m_product)
                v_hat = self.v[i][param] / (1-self.beta_v_product)
                weight_update = self.step_size * m_hat / \
                    (np.sqrt(v_hat) + self.epsilon)

                weights[i][param] = weights[i][param] + weight_update
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights
