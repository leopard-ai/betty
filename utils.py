class LimitedList(list):
    # Read-only
    @property
    def maxLen(self):
        """[summary]
        Return the maximum length for the list
        """
        return self._maxlen

    def __init__(self, maxlen=5, *args, **kwargs):
        self._maxlen = maxlen
        list.__init__(self, *args, **kwargs)

    def _truncate(self):
        """Called by various methods to reinforce the maximum length."""
        dif = len(self)-self._maxlen
        if dif > 0:
            self[:dif]=[]

    def append(self, x):
        list.append(self, x)
        self._truncate()

    def insert(self, *args):
        list.insert(self, *args)
        self._truncate()

    def extend(self, x):
        list.extend(self, x)
        self._truncate()

    def __setitem__(self, *args):
        list.__setitem__(self, *args)
        self._truncate()
