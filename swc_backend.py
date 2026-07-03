"""
swc_backend.py -- one interface, two backends.

This is the seam that lets every reproduction script run UNCHANGED against either:
  - the hosted TrueLoop endpoint (HTTP; supports n up to 4096), or
  - the offline build (in-process compiled runtime; no n cap).

You never edit the reproduction scripts to switch. You pass backend="endpoint" or
backend="offline" (or set TRUELOOP_BACKEND). Everything else is identical.

The runtime's update law is NOT in this file. For the endpoint it runs server-side; for
the offline build it runs inside the licensed, compiled `trueloop` package. This file only
opens a session, sends measurements/scores, and reads back the configuration.
"""
import os, json, math, time, urllib.request, urllib.error

DEFAULT_ENDPOINT = os.environ.get("TRUELOOP_ENDPOINT", "https://trueloopcompute.com")
ENDPOINT_MAX_N = 4096   # hosted sessions are capped at n<=4096; use the offline build beyond


class SWCError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Endpoint backend: raw HTTP, no third-party dependencies.
# --------------------------------------------------------------------------- #
class _EndpointSession:
    def __init__(self, key, n, mode, target=None, config=None, base=None):
        self.key = key
        self.base = (base or DEFAULT_ENDPOINT).rstrip("/")
        self.n = n
        self.mode = mode
        self.token = None
        self.phi = None
        payload = {"n": n, "mode": mode}
        if target is not None:
            payload["target"] = target
        if config is not None:
            payload["config"] = config
        j = self._post("/api/session/start", payload)
        self.token = j.get("token") or j.get("session")
        self.phi = j.get("config") or j.get("phi")

    def _post(self, path, payload, retries=3):
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.key}
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.base + path, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if 400 <= e.code < 500:
                    raise SWCError("HTTP %d: %s" % (e.code, body))
                last = SWCError("HTTP %d: %s" % (e.code, body))
            except Exception as e:
                last = SWCError(str(e))
            time.sleep(1.0 * (attempt + 1))
        raise last

    def step(self, measurement, score=None, target=None):
        payload = {"token": self.token, "measurement": list(measurement)}
        if score is not None:
            payload["score"] = score
        if target is not None:
            payload["target"] = target
        j = self._post("/api/session/step", payload)
        self.phi = j.get("config") or j.get("phi")
        return self.phi

    def end(self):
        try:
            self._post("/api/session/end", {"token": self.token})
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Offline backend: the licensed compiled runtime (no n cap).
# --------------------------------------------------------------------------- #
class _OfflineSession:
    def __init__(self, n, mode, target=None, config=None):
        try:
            import trueloop
        except Exception as e:
            raise SWCError(
                "Offline backend selected but the 'trueloop' package is not importable. "
                "Unzip your licensed offline build and either install it or add its folder "
                "to PYTHONPATH (see README). Underlying error: %r" % (e,))
        est = (config or {}).get("estimator", "elite")
        self._opt = trueloop.SWCRuntime(n=n, mode=mode, target=target, estimator=est)
        x0 = [math.pi / 2] * n if mode != "regulation" else (target or [0.5] * n)
        self.phi = self._opt.start(x0)

    def step(self, measurement, score=None, target=None):
        self.phi = self._opt.step(list(measurement), score=score, target=target)
        return self.phi

    def end(self):
        try:
            self._opt.end()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Public factory -- this is what the reproduction scripts call.
# --------------------------------------------------------------------------- #
def open_session(n, mode="optimization", target=None, config=None,
                 backend=None, key=None, base=None):
    """Open a runtime session against the chosen backend.

    backend : "endpoint" | "offline"  (default: env TRUELOOP_BACKEND or "endpoint")
    key     : license/eval key (endpoint only; default env TRUELOOP_KEY)
    base    : endpoint URL override (endpoint only)
    Returns an object with .step(measurement, score=, target=) -> config, and .end().
    """
    backend = (backend or os.environ.get("TRUELOOP_BACKEND", "endpoint")).lower()
    if backend == "endpoint":
        if n > ENDPOINT_MAX_N:
            raise SWCError(
                "n=%d exceeds the hosted endpoint cap of %d. Switch to the offline build "
                "(backend='offline') to run at this scale." % (n, ENDPOINT_MAX_N))
        key = key or os.environ.get("TRUELOOP_KEY")
        if not key:
            raise SWCError("No key. Set TRUELOOP_KEY or pass key=... for the endpoint backend.")
        return _EndpointSession(key, n, mode, target=target, config=config, base=base)
    elif backend == "offline":
        return _OfflineSession(n, mode, target=target, config=config)
    else:
        raise SWCError("Unknown backend %r (use 'endpoint' or 'offline')." % backend)


def backend_name(backend=None):
    return (backend or os.environ.get("TRUELOOP_BACKEND", "endpoint")).lower()
