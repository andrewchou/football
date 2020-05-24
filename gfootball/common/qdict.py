from gfootball.env.football_action_set import CoreAction

class ADict(dict):
    def __getitem__(self, key):
        if key not in self:
            super().__setitem__(key, 0.0)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, CoreAction), key
        assert isinstance(value, float), value
        super().__setitem__(key, value)

    def get(self, key, default):
        assert isinstance(key, CoreAction), key
        assert isinstance(default, float), default
        return super().get(key, default)

class QDict(dict):
    def __getitem__(self, key):
        if key not in self:
            super().__setitem__(key, ADict())
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple), key
        assert isinstance(value, ADict), value
        super().__setitem__(key, value)

    def get(self, key, default):
        assert isinstance(key, tuple), key
        assert isinstance(default, ADict), default
        return super().get(key, default)


if __name__ == '__main__':
    q = QDict()