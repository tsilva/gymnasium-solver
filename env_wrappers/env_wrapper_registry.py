class EnvWrapperRegistry():    
    _registry = {}

    @classmethod
    def register(cls, wrapper_classes):
        if not type(wrapper_classes) is list: wrapper_classes = [wrapper_classes]
        for wrapper_cls in wrapper_classes: cls._registry[wrapper_cls.__name__] = wrapper_cls

    @classmethod
    def apply(cls, env, wrapper_spec):
        wrapper_id = wrapper_spec['id']
        wrapper_kwargs = wrapper_spec.get('kwargs', {})
        wrapper_cls = cls._registry[wrapper_id]
        wrapped_env = wrapper_cls(env, **wrapper_kwargs)
        return wrapped_env
