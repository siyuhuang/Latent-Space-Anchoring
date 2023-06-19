
def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def requires_grad_target_layer(model, flag=True, target_layer=None):
    for name, param  in model.named_parameters():
        if target_layer is None or target_layer in name:
            param.requires_grad = flag
            
def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name and k[len(name)]=='.'}
    return d_filt