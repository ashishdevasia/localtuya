"""Regression tests: a single device's IP change must not disrupt other devices.

These tests exercise the *real* integration code -- both
``custom_components/localtuya/__init__.py`` (``_device_discovered`` ->
``update_listener``) and the *real* ``custom_components/localtuya/common.py``
``TuyaDevice`` (``async_update_config`` -> ``close`` -> reconnect) -- while
stubbing the Home Assistant runtime and pytuya. No ``homeassistant`` install and
no crypto deps are needed, so it stays tiny in RAM.

Crucially, the fake pytuya interface models asyncio's ``connection_lost``
delivery: closing a transport schedules ``listener.disconnected()`` on a *later*
loop iteration. That is what makes the close->reconnect race (C1) observable --
without it the test would give false confidence.

What is verified:
  * Device A changing IP repoints/reconnects *only* device A, in place.
  * Device B (unchanged) is never closed/reconnected: same interface object.
  * No full-entry reload is triggered.
  * Device A actually ends up connected to the NEW IP (catches the C1 race
    where the old transport's disconnected() callback kills the new
    connection).

Run standalone:  python3 tests/test_single_device_ip_change.py
"""
import asyncio
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Stub the Home Assistant + pytuya surface so the REAL __init__.py and common.py
# can be imported and run.
# --------------------------------------------------------------------------- #
def _install_stub_modules():
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__path__ = []
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    # Any CONF_*/ATTR_*/SERVICE_*/STATE_* name resolves to its own string, which
    # keeps dict keys consistent between the integration and the fake entry.
    hconst = types.ModuleType("homeassistant.const")
    hconst.__getattr__ = lambda name: name  # PEP 562
    sys.modules["homeassistant.const"] = hconst

    module("homeassistant")
    module("homeassistant.config_entries", ConfigEntry=type("ConfigEntry", (), {}))
    module("homeassistant.core", HomeAssistant=type("HomeAssistant", (), {}),
           callback=lambda f: f)
    module("homeassistant.exceptions",
           HomeAssistantError=type("HomeAssistantError", (Exception,), {}))
    module("homeassistant.helpers")
    module("homeassistant.helpers.config_validation", string=str)
    module("homeassistant.helpers.entity_registry",
           async_get=lambda *a, **k: None,
           async_entries_for_config_entry=lambda *a, **k: [])
    module("homeassistant.helpers.device_registry",
           DeviceEntry=type("DeviceEntry", (), {}))
    module("homeassistant.helpers.event",
           async_track_time_interval=lambda *a, **k: (lambda: None))
    module("homeassistant.helpers.service",
           async_register_admin_service=lambda *a, **k: None)
    module("homeassistant.helpers.dispatcher",
           async_dispatcher_connect=lambda *a, **k: (lambda: None),
           async_dispatcher_send=lambda *a, **k: None)
    module("homeassistant.helpers.restore_state",
           RestoreEntity=type("RestoreEntity", (), {}))

    module("voluptuous",
           Schema=lambda *a, **k: (a[0] if a else None),
           Required=lambda x, *a, **k: x,
           Optional=lambda x, *a, **k: x)

    # Lightweight pytuya: base classes for TuyaDevice + a no-op logger mixin.
    # connect() is never called because we patch TuyaDevice._make_connection.
    class _ContextualLogger:
        def set_logger(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

    async def _connect(*a, **k):  # pragma: no cover - patched out
        raise AssertionError("pytuya.connect should not be called in this test")

    module("custom_components.localtuya.pytuya",
           TuyaListener=type("TuyaListener", (), {}),
           ContextualLogger=_ContextualLogger,
           connect=_connect)

    # Sibling modules that __init__.py imports but that are irrelevant here.
    module("custom_components.localtuya.config_flow",
           ENTRIES_VERSION=3, config_schema=lambda *a, **k: {})
    module("custom_components.localtuya.cloud_api",
           TuyaCloudApi=type("TuyaCloudApi", (), {"__init__": lambda self, *a, **k: None}))

    class _StubDiscovery:
        def __init__(self, callback=None):
            self.callback = callback

        async def start(self):
            return None

        def close(self):
            self.callback = None

    module("custom_components.localtuya.discovery", TuyaDiscovery=_StubDiscovery)


# --------------------------------------------------------------------------- #
# Fake HA core objects.
# --------------------------------------------------------------------------- #
class FakeBus:
    def async_listen_once(self, *a, **k):
        return lambda: None


class FakeServices:
    def async_register(self, *a, **k):
        return None


class FakeConfigEntry:
    def __init__(self, entry_id, data, version):
        self.entry_id = entry_id
        self.data = data
        self.version = version
        self.title = None
        self.state = types.SimpleNamespace(recoverable=True)
        self.update_listeners = []

    def add_update_listener(self, listener):
        self.update_listeners.append(listener)

        def _unsub():
            if listener in self.update_listeners:
                self.update_listeners.remove(listener)

        return _unsub


class FakeConfigEntries:
    def __init__(self, hass):
        self._hass = hass
        self._entries = []
        self.reload_calls = []

    def add(self, entry):
        self._entries.append(entry)

    def async_entries(self, domain=None):
        return list(self._entries)

    def async_update_entry(self, entry, data=None, title=None):
        if data is not None:
            entry.data = data
        if title is not None:
            entry.title = title
        for listener in list(entry.update_listeners):
            self._hass.async_create_task(listener(self._hass, entry))
        return True

    async def async_reload(self, entry_id):
        self.reload_calls.append(entry_id)
        entry = next(e for e in self._entries if e.entry_id == entry_id)
        import custom_components.localtuya as lt

        await lt.async_unload_entry(self._hass, entry)

    async def async_forward_entry_setups(self, entry, platforms):
        return True

    async def async_forward_entry_unload(self, entry, component):
        return True


class FakeHass:
    def __init__(self):
        self.data = {}
        self.bus = FakeBus()
        self.services = FakeServices()
        self.config_entries = FakeConfigEntries(self)
        self._tasks = []

    def async_create_task(self, coro):
        task = asyncio.ensure_future(coro)
        self._tasks.append(task)
        return task

    async def async_block_till_done(self):
        # Drain both tasks scheduled via async_create_task and raw asyncio tasks
        # spawned by the integration (e.g. device connect tasks created with
        # asyncio.create_task inside async_connect), until the loop is quiescent.
        for _ in range(1000):
            pending = self._tasks
            self._tasks = []
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task() and not task.done():
                    if task not in pending:
                        pending.append(task)
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Fake pytuya interface that models asyncio's connection_lost delivery.
# --------------------------------------------------------------------------- #
# Model asyncio's connection_lost being delivered SEVERAL loop iterations after
# transport.close() -- which is what happens when the socket has pending writes
# (the heartbeat writes right before close). A faithful test must not assume the
# benign single-iteration delivery.
DELIVERY_ITERS = 5


class FakeInterface:
    def __init__(self, device):
        # Mirror pytuya's TuyaProtocol: connection_lost reads self.listener and
        # only calls disconnected() if it is still attached. Detaching it (set to
        # None) is the supported way to neutralise a stale callback.
        self.listener = device

    async def close(self):
        loop = asyncio.get_event_loop()

        def _deliver(n):
            if n > 0:
                loop.call_soon(_deliver, n - 1)
                return
            listener = self.listener
            if listener is not None:
                listener.disconnected()

        loop.call_soon(_deliver, DELIVERY_ITERS)


async def _run():
    _install_stub_modules()
    sys.path.insert(0, REPO_ROOT)

    import custom_components.localtuya as lt
    from custom_components.localtuya import common, const

    H = sys.modules["homeassistant.const"]
    DOMAIN = const.DOMAIN
    HOST = H.CONF_HOST
    DEV_A, DEV_B = "device_a", "device_b"
    NEW_IP = "192.168.1.250"

    # Replace the real network connect with a fake that "connects" to whatever
    # host is currently configured, so we can observe repointing without I/O.
    # Also model the scan-interval poller real _make_connection registers, so the
    # test can detect the timer leak: each connection adds a live timer whose
    # unsub removes it; close() must clear it on repoint.
    active_timers = []

    async def _fake_make_connection(self):
        # Real _make_connection spans many awaits (connect, status, heartbeat,
        # restore_state); model that so a late disconnected() can land mid-way.
        for _ in range(3):
            await asyncio.sleep(0)
        self._interface = FakeInterface(self)
        self._connected_host = self._dev_config_entry[HOST]
        self._connect_task = None
        token = object()
        active_timers.append(token)

        def _unsub(tok=token):
            if tok in active_timers:
                active_timers.remove(tok)

        self._unsub_interval = _unsub

    common.TuyaDevice._make_connection = _fake_make_connection

    def _dev(host, key):
        return {
            H.CONF_DEVICE_ID: "id",
            H.CONF_FRIENDLY_NAME: "dev",
            const.CONF_LOCAL_KEY: "localkey",
            H.CONF_HOST: host,
            const.CONF_PRODUCT_KEY: key,
            H.CONF_ENTITIES: [{H.CONF_PLATFORM: "switch", H.CONF_ID: "1"}],
        }

    entry_data = {
        H.CONF_REGION: "eu",
        H.CONF_CLIENT_ID: "",
        H.CONF_CLIENT_SECRET: "",
        const.CONF_USER_ID: "",
        const.CONF_NO_CLOUD: True,
        H.CONF_DEVICES: {
            DEV_A: _dev("192.168.1.10", "keyA"),
            DEV_B: _dev("192.168.1.11", "keyB"),
        },
        const.ATTR_UPDATED_AT: "0",
    }
    entry = FakeConfigEntry("entry1", entry_data, version=3)

    hass = FakeHass()
    await lt.async_setup(hass, {})
    hass.config_entries.add(entry)
    await lt.async_setup_entry(hass, entry)
    await hass.async_block_till_done()

    device_a = hass.data[DOMAIN][const.TUYA_DEVICES][DEV_A]
    device_b = hass.data[DOMAIN][const.TUYA_DEVICES][DEV_B]
    assert device_a.connected and device_b.connected, "both devices should connect"
    iface_a_before = device_a._interface
    iface_b_before = device_b._interface

    # Simulate device A broadcasting a new IP (e.g. a new DHCP lease after a
    # power outage). Device B's IP is unchanged.
    discovery = hass.data[DOMAIN][const.DATA_DISCOVERY]
    discovery.callback({"ip": NEW_IP, "gwId": DEV_A, "productKey": "keyA"})
    await hass.async_block_till_done()

    print(f"reloads triggered   : {hass.config_entries.reload_calls}")
    print(f"device_a connected={device_a.connected} "
          f"host={device_a.device_config[HOST]} "
          f"interface_changed={device_a._interface is not iface_a_before}")
    print(f"device_b connected={device_b.connected} "
          f"host={device_b.device_config[HOST]} "
          f"interface_changed={device_b._interface is not iface_b_before}")

    # No full-entry reload should occur for a single device's IP change.
    assert hass.config_entries.reload_calls == [], (
        f"a full-entry reload was triggered ({hass.config_entries.reload_calls})"
    )

    # Device B must be completely untouched: same interface object, still up.
    assert device_b._interface is iface_b_before, "device B was reconnected/torn down"
    assert device_b.connected, "device B lost its connection"
    assert device_b.device_config[HOST] == "192.168.1.11", "device B host changed"

    # Device A must be repointed to the new IP AND actually reconnected. This is
    # what catches the C1 race: if the old transport's disconnected() callback
    # nulled the fresh interface, device A would end up disconnected here.
    assert device_a.device_config[HOST] == NEW_IP, "device A not repointed"
    assert device_a._interface is not iface_a_before, "device A was not reconnected"
    assert device_a.connected, (
        "device A is NOT connected after the IP change -- the close()->reconnect "
        "path lost the race with the old transport's disconnected() callback."
    )
    assert device_a._connected_host == NEW_IP, "device A connected to the wrong host"

    print("PASS: device A repointed+reconnected to the new IP in place; "
          "device B untouched; no full-entry reload.")

    # --- Scenario 2: concurrent repoints of the SAME device must not wedge it.
    # When several devices change IP at once (multi-device power outage), every
    # device's update_listener repoints every changed device, so one device can
    # receive overlapping async_update_config calls. Without serialization,
    # close() cancels+awaits the fresh connect task, CancelledError aborts the
    # second call before it resets _is_closing, and the device is wedged
    # (connected=False, _is_closing=True) until a reload/restart.
    c_entry = FakeConfigEntry(
        "entry2", {H.CONF_DEVICES: {"cdev": _dev("10.0.0.1", "keyC")}}, version=3)
    cdev = common.TuyaDevice(object(), c_entry, "cdev")
    cdev.async_connect()
    for _ in range(10):
        await asyncio.sleep(0)
    assert cdev.connected, "concurrent: device should connect initially"

    await asyncio.gather(
        cdev.async_update_config(_dev("10.0.0.50", "keyC")),
        cdev.async_update_config(_dev("10.0.0.99", "keyC")),
        return_exceptions=True,
    )
    for _ in range(20):
        await asyncio.sleep(0)
    # The 60s reconnect tick must be able to recover (proves it is not wedged).
    cdev.async_connect()
    for _ in range(10):
        await asyncio.sleep(0)

    print(f"concurrent: connected={cdev.connected} "
          f"host={getattr(cdev, '_connected_host', None)} "
          f"is_closing={cdev._is_closing} "
          f"live_scan_timers={len(active_timers)}")
    assert not cdev._is_closing, (
        "concurrent: device WEDGED (_is_closing=True) after overlapping repoints"
    )
    # MEDIUM 1 regression: each repoint must clear the previous scan-interval
    # timer. Three devices have connected (A, B, cdev); A was repointed once and
    # cdev twice. With the leak, those repoints would accumulate extra timers
    # (>3). With the fix, exactly one live timer per device remains.
    assert len(active_timers) == 3, (
        f"scan-interval timer leak: {len(active_timers)} live timers for 3 "
        "devices (close() must clear _unsub_interval on repoint)"
    )
    assert cdev.connected, (
        "concurrent: device stranded (not connected) after overlapping repoints"
    )
    assert cdev._connected_host in ("10.0.0.50", "10.0.0.99"), (
        f"concurrent: converged to unexpected host {cdev._connected_host}"
    )
    print("PASS: concurrent repoints converged without wedging the device.")

    # --- Scenario 3: a failing close() during repoint must not wedge the device.
    # If close() raises anything other than CancelledError, the teardown must
    # still reset _is_closing/_interface (try/finally) so async_connect can
    # recover -- otherwise the device is stuck on the dead interface forever.
    f_entry = FakeConfigEntry(
        "entry3", {H.CONF_DEVICES: {"fdev": _dev("10.0.0.1", "keyF")}}, version=3)
    fdev = common.TuyaDevice(object(), f_entry, "fdev")
    fdev.async_connect()
    for _ in range(10):
        await asyncio.sleep(0)
    assert fdev.connected, "close-raises: device should connect initially"

    async def _boom_close():
        raise RuntimeError("simulated transport close failure")

    fdev._interface.close = _boom_close
    try:
        await fdev.async_update_config(_dev("10.0.0.77", "keyF"))
    except Exception:  # noqa: BLE001 - tolerated; we assert on resulting state
        pass
    for _ in range(20):
        await asyncio.sleep(0)

    print(f"close-raises: connected={fdev.connected} "
          f"host={getattr(fdev, '_connected_host', None)} "
          f"is_closing={fdev._is_closing}")
    assert not fdev._is_closing, (
        "close-raises: device WEDGED (_is_closing=True) after a failing close()"
    )
    assert fdev.connected and fdev._connected_host == "10.0.0.77", (
        "close-raises: device did not recover/repoint after a failing close()"
    )
    print("PASS: a failing close() during repoint did not wedge the device.")

    # --- Scenario 4: close() then a deferred disconnected() must release the
    # dispatcher subscription exactly once. On unload the listener is NOT
    # detached, so close() runs and later connection_lost -> disconnected() runs
    # too; both release _disconnect_task. The real HA unsub raises on a second
    # call, so close() must null it after releasing.
    u_entry = FakeConfigEntry(
        "entry4", {H.CONF_DEVICES: {"udev": _dev("10.0.0.1", "keyU")}}, version=3)
    udev = common.TuyaDevice(object(), u_entry, "udev")
    udev.async_connect()
    for _ in range(10):
        await asyncio.sleep(0)
    unsub_calls = {"n": 0}

    def _tracking_unsub():
        unsub_calls["n"] += 1

    udev._disconnect_task = _tracking_unsub
    await udev.close()      # unload-style close (listener not detached)
    udev.disconnected()     # deferred connection_lost -> disconnected()
    print(f"dispatcher-unsub: released {unsub_calls['n']} time(s)")
    assert unsub_calls["n"] == 1, (
        f"dispatcher subscription released {unsub_calls['n']} times "
        "(close() must null _disconnect_task to avoid a double-release)"
    )
    print("PASS: dispatcher subscription released exactly once across "
          "close()+disconnected().")


def main():
    try:
        asyncio.run(_run())
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
