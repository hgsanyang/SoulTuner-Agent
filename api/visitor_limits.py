"""Single-process visitor admission limits; no client-supplied IP trust."""
import time


class VisitorLimits:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.windows = {}
        self.active = {}
        self.total_active = 0

    def allow(self, key, limit, window=60):
        now = self.clock()
        expired = [k for k, (_, end) in self.windows.items() if end <= now]
        for k in expired:
            del self.windows[k]
        if key not in self.windows and len(self.windows) >= 4096:
            return False
        count, end = self.windows.get(key, (0, now + window))
        if count >= limit:
            return False
        self.windows[key] = (count + 1, end)
        return True

    def enter(self, subject):
        if self.active.get(subject, 0) >= 1 or self.total_active >= 4:
            return False
        self.active[subject] = 1
        self.total_active += 1
        return True

    def leave(self, subject):
        if self.active.pop(subject, None):
            self.total_active -= 1
