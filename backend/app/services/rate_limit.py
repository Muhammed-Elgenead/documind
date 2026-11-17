"""In-memory sliding-window rate limiting.

State lives in this process, which matches the single-instance deployment (see
docs/decisions.md). With several instances, move it to Redis or the database.
"""

import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
_RULE = re.compile(r"^\s*(\d+)\s*/\s*(second|minute|hour|day)s?\s*$")


@dataclass(frozen=True)
class Rule:
    limit: int
    seconds: int


def parse_rules(spec: str) -> list[Rule]:
    """'10/minute;200/day' -> [Rule(10, 60), Rule(200, 86400)]. Empty -> no rules."""
    rules = []
    for part in filter(str.strip, spec.split(";")):
        m = _RULE.match(part)
        if not m:
            raise ValueError(f"Bad rate limit rule {part!r}; use e.g. '10/minute;200/day'")
        rules.append(Rule(int(m.group(1)), _PERIODS[m.group(2)]))
    return rules


class RateLimiter:
    def __init__(self, rules: list[Rule], clock: Callable[[], float] = time.monotonic):
        self.rules = rules
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._longest = max((r.seconds for r in rules), default=0)
        self._calls = 0

    def hit(self, key: str) -> int | None:
        """Record a request for `key`. Returns seconds to wait if it's over a limit, else None.

        Rejected requests aren't recorded, so a client that keeps retrying isn't locked out
        for longer than the window.
        """
        if not self.rules:
            return None
        now = self.clock()
        self._calls += 1
        if self._calls % 1000 == 0:  # forget clients idle for longer than every window
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - self._longest}
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self._longest:
            hits.popleft()
        wait = 0.0
        for rule in self.rules:
            in_window = [t for t in hits if t > now - rule.seconds]
            if len(in_window) >= rule.limit:
                # The oldest request that still counts must leave the window first.
                oldest = in_window[-rule.limit]
                wait = max(wait, oldest + rule.seconds - now)
        if wait > 0:
            return max(1, math.ceil(wait))
        hits.append(now)
        return None


@lru_cache
def limiter_for(scope: str, spec: str) -> RateLimiter:
    """One shared limiter per (scope, rules); tests reset with `limiter_for.cache_clear()`."""
    return RateLimiter(parse_rules(spec))
