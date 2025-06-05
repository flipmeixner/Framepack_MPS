# By lllyasviel


import torch
from diffusers_helper.device_utils import get_device, get_free_memory_gb, empty_cache


cpu = torch.device('cpu')
gpu = get_device()
gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    """
    Legacy function name kept for compatibility.
    Now uses the device-agnostic get_free_memory_gb function.
    """
    return get_free_memory_gb(device)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_free_memory_gb(target_device) <= preserved_memory_gb:
            empty_cache(target_device)
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    empty_cache(target_device)
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_free_memory_gb(target_device) >= preserved_memory_gb:
            empty_cache(target_device)
            return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    empty_cache(target_device)
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    empty_cache(gpu)
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
