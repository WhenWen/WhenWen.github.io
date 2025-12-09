muon_h_runs = [
    "https://wandb.ai/marin-community/marin/runs/qwen3_130m_muonh_4096_lr_0.02_adam_lr_0.008-354a89",
    "https://wandb.ai/marin-community/marin/runs/qwen3_300m_muonh_4096_lr_0.01-f5ae20",
    "https://wandb.ai/marin-community/marin/runs/qwen3_520m_muonh_4096_lr_0.01-cb97ec",
    "https://wandb.ai/marin-community/marin/runs/qwen3_1_2b_muonh_4096_low_lr-f47219"
]

adam_h_runs = [
    "https://wandb.ai/marin-community/marin/runs/llama_130m_adamh_lr0.02_adam_lr0.008_warmup1000_qk-add079",
    "https://wandb.ai/marin-community/marin/runs/llama_300m_adamh_lr0.02_adam_lr0.008_warmup1000_qk-dbc23e",
    "https://wandb.ai/marin-community/marin/runs/llama_1_2b_adamh_lr0.02_adam_lr0.0015_warmup1000_qk_0.0-28a210",
    "https://wandb.ai/marin-community/marin/runs/llama_520m_adamh_lr0.02_adam_lr0.004_warmup1000_qk-86d687",
    ""
]

muon_runs = [
    "https://wandb.ai/marin-community/marin/runs/qwen3_130m_muon_4096-04770b",
    "https://wandb.ai/marin-community/marin/runs/qwen3_300m_muon_4096-ee4f99",
    "https://wandb.ai/marin-community/marin/runs/qwen3_520m_muon_4096-361875",
    "https://wandb.ai/marin-community/marin/runs/qwen3_1_2b_muon_4096-d117d2"
]

adam_runs = [
    "https://wandb.ai/marin-community/marin/runs/llama_130m_adamc_cos_4096-55946b",
    "https://wandb.ai/marin-community/marin/runs/llama_300m_adamc_cos_4096-a94d5d",
    "https://wandb.ai/marin-community/marin/runs/llama_520m_adamc_cos_4096-a3c775",
    "https://wandb.ai/marin-community/marin/runs/llama_1_2b_adamc_cos_4096-6b3a21"
]

expected_params = {
    "130m": 134217728,  # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,  # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,  # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,  # 32 * (1536*6144*3 + 1536*1536*4)
}


from matplotlib import pyplot as plt

plt.plot(expected_params.values(), muon_h_losses, label="MuonH", color = "blue", linestyle = "--")
plt.plot(expected_params.values(), muon_losses, label="Muon", color = "darkblue")
plt.plot(expected_params.values(), adam_h_losses, label="AdamH", color = "red", linestyle = "--")
plt.plot(expected_params.values(), adam_c_losses, label="AdamC", color = "darkred")
plt.xlabel("Model Size")
plt.xticks(list(expected_params.values()), list(expected_params.keys()))
plt.ylabel("Loss")
# plt.yscale("log")
# plt.xscale("log")
plt.legend()
plt.savefig("muon_h_vs_muon.png")
plt.close()

# fit power law for four optimizers

from scipy.optimize import curve_fit

def power_law(x, a, b, c):
    return a * x ** b + c
import numpy as np
popt_muonh, _ = curve_fit(power_law, np.array(list(expected_params.values())), muon_h_losses, p0=[1.0, -0.5, 1.0], maxfev=10000)
popt_muon, _ = curve_fit(power_law, np.array(list(expected_params.values())), muon_losses, p0=[1.0, -0.5, 1.0], maxfev=10000)
popt_adamh, _ = curve_fit(power_law, np.array(list(expected_params.values())), adam_h_losses, p0=[1.0, -0.5, 1.0], maxfev=10000)
popt_adamc, _ = curve_fit(power_law, np.array(list(expected_params.values())), adam_c_losses, p0=[1.0, -0.5, 1.0], maxfev=10000)


# plot the power law to 7B model
extrapolation_params =   {
    "3B": 3207959552,
    "7B": 7207959552,
    "14B": 14207959552,
}


for key, value in expected_params.items():
    extrapolation_params[key] = value

reversed_extrapolation_params = {v: k for k, v in extrapolation_params.items()}

param_list = sorted(list(extrapolation_params.values()))
plt.plot(param_list, [power_law(x, *popt_muonh) for x in param_list], label="MuonH", color = "blue", linestyle = "--")
plt.plot(param_list, [power_law(x, *popt_muon) for x in param_list], label="Muon", color = "darkblue")
plt.plot(param_list, [power_law(x, *popt_adamh) for x in param_list], label="AdamH", color = "red", linestyle = "--")
plt.plot(param_list, [power_law(x, *popt_adamc) for x in param_list], label="AdamC", color = "darkred")
plt.xlabel("Model Size")
plt.xticks(list(param_list), [reversed_extrapolation_params[x] for x in param_list])
plt.ylabel("Loss")
plt.legend()
plt.savefig("muon_h_vs_muon_extrapolation.png")
