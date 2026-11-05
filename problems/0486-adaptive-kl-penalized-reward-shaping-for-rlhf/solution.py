import numpy as np

def adaptive_kl_penalized_reward(rewards, log_probs_policy, log_probs_reference, beta, kl_target, beta_update_factor=1.5):
	
	kl_divergence = np.array([
        np.sum(np.array(policy) - np.array(reference))
        for policy, reference in zip(log_probs_policy, log_probs_reference)
    ])
	shaped_reward = np.array(rewards) - kl_divergence*beta
	mean_kl = np.mean(kl_divergence)
	if mean_kl > 1.5*kl_target:
		beta = beta*beta_update_factor
	elif mean_kl < kl_target/1.5:
		beta = beta/beta_update_factor
	return {
		'per_response_kl': np.round(kl_divergence, 4).tolist(),
		'mean_kl': round(float(mean_kl), 4),
		'shaped_rewards': np.round(shaped_reward, 4).tolist(),
		'updated_beta': round(float(beta), 4)
	}