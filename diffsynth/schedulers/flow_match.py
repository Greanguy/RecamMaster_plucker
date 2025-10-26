import torch



class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas) # shift>1 时，会把中间的 sigma 值向某端聚拢；shift=1 时该变换是恒等变换（不改变 sigmas）
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps

            # 这是一个以 num_inference_steps/2 为中心，形状为高斯（或类高斯）的权重函数（注意常数因子 -2）。
            # 目的是在中间时刻给予更高权重（强调中段），而在两端时刻降低权重。
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)

            y_shifted = y - y.min() # 将其平移到非负（最小值为 0）。
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum()) # 对权重归一化，使得权重之和等于 num_inference_steps（或等于步数），即做了一个缩放，便于后续把该权重直接作为 batch 内加权或期望计数。
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs()) # 类似最近邻采样
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample # 将 sample 的计算公式代入可以发现此处 target 实质上为“残差化的噪声预测”
        return target
    

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
