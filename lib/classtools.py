def define_property(self, name, value=None, readable=True, writable=True):
    # "_User__name" のような name mangling 後の名前.
    field_name = "_{}__{}".format(self.__class__.__name__, name)

    # 初期値を設定する.
    setattr(self, field_name, value)

    # getter/setter を生成し, プロパティを定義する.
    getter = (lambda self: getattr(self, field_name)) if readable else None
    setter = (lambda self, value: setattr(self, field_name, value)) if writable else None
    setattr(self.__class__, name, property(getter, setter))

def define_properties(constructor=None, *, accessible=(), readable=(), writable=()):
    if callable(constructor):
        def wrapper(self, *args, **kwargs):
            for name, value in kwargs.items():
                define_property(self, name, value)
            constructor(self, *args, **kwargs)
        return wrapper
    else:
        to_set = lambda x: set(x) if any(isinstance(x, type_) for type_ in (set, list, tuple)) else {x}
        accessibles = to_set(accessible)
        readables = accessibles | to_set(readable)
        writables = accessibles | to_set(writable)

        def decorator(constructor):
            def wrapper(self, *args, **kwargs):
                for name in (readables | writables):
                    readable = name in readables
                    writable = name in writables
                    initial_value = kwargs.get(name, None)
                    define_property(self, name, initial_value, readable, writable)
                constructor_kwargs = dict([(key, kwargs[key]) for key in (constructor.__kwdefaults__ or {}) if key in kwargs])
                constructor(self, *args, **constructor_kwargs)
            return wrapper
        return decorator
