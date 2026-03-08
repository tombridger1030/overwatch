#!/usr/bin/env python3
"""Overwatch - Desk presence tracker via webcam + menu bar."""

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import rumps

# Lazy-loaded globals
_cv2 = None
_cascade = None

DATA_DIR = Path.home() / "Library" / "Application Support" / "Overwatch"
DATA_FILE = DATA_DIR / "presence.json"
DETECT_INTERVAL = 300  # 5 minutes
SESSION_GAP = 600  # 10 minutes = new session
FLUSH_INTERVAL = 300  # 5 minutes
DAILY_GOAL = 8 * 3600  # 8 hours in seconds
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240
SMOOTHING_WINDOW = 3  # 2-of-3 vote
CAMERA_INDEX = 1  # NexiGo N60 webcam
CAMERA_STARTUP_TIMEOUT = 2.5  # seconds to wait for first valid frame
CAMERA_SECOND_FRAME_TIMEOUT = 1.0  # seconds to wait for frame B
FRAME_READ_SLEEP = 0.05
DISCARD_FRAMES = 10
DETECT_MAX_ATTEMPTS = 3
DETECT_RETRY_BACKOFF = 0.2
CAMERA_GRACE_MISSES = 2  # keep present for transient camera misses
WARMUP_FRAMES = 60  # ~2s at 30fps — let NexiGo auto-exposure settle
UI_TICK_INTERVAL = 1  # seconds


def _get_cv2():
    """Lazy-load OpenCV to save ~20MB at startup."""
    global _cv2, _cascade
    if _cv2 is None:
        import cv2

        _cv2 = cv2
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        _cascade = cv2.CascadeClassifier(cascade_path)
    return _cv2, _cascade


class Detector:
    """Detect desk presence via face detection + motion sensing, smooth results."""

    def __init__(self):
        self._history = []  # ring buffer of booleans

    def _smooth(self, raw):
        """Asymmetric smoothing: instant away on any False, require unanimous True for present."""
        raw = bool(raw)
        self._history.append(raw)
        if len(self._history) > SMOOTHING_WINDOW:
            self._history = self._history[-SMOOTHING_WINDOW:]
        if len(self._history) < 2:
            return raw
        return all(self._history)

    @staticmethod
    def _open_capture(cv2):
        """Open configured camera index with explicit AVFoundation backend."""
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        backend = cap.getBackendName()
        return cap, backend

    @staticmethod
    def _wait_for_valid_frame(cap, timeout_s):
        """Poll until a valid frame arrives or timeout expires."""
        start = time.time()
        deadline = start + timeout_s
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 1.0:
                return frame, (time.time() - start) * 1000.0
            time.sleep(FRAME_READ_SLEEP)
        return None, (time.time() - start) * 1000.0

    def _detect_once(self):
        """
        Attempt one camera detection cycle.
        Returns dict with: raw, reason, backend, startup_ms, face_count, motion_score.
        """
        cv2, cascade = _get_cv2()
        cap, backend = self._open_capture(cv2)
        fail = {
            "raw": None,
            "reason": "",
            "backend": backend,
            "startup_ms": 0.0,
            "face_count": 0,
            "motion_score": 0.0,
        }
        if not cap.isOpened():
            fail["reason"] = "open_failed"
            return fail

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

            first_frame, startup_ms = self._wait_for_valid_frame(
                cap, CAMERA_STARTUP_TIMEOUT
            )
            fail["startup_ms"] = startup_ms
            if first_frame is None:
                fail["reason"] = "startup_timeout"
                return fail

            # Warmup: read frames to let NexiGo auto-exposure settle
            for _ in range(WARMUP_FRAMES):
                cap.read()

            # Capture frame A (post-warmup)
            frame_a, _ = self._wait_for_valid_frame(cap, CAMERA_SECOND_FRAME_TIMEOUT)
            if frame_a is None:
                fail["reason"] = "frame_a_post_warmup_failed"
                return fail
            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)

            for _ in range(DISCARD_FRAMES):
                ret, _frame = cap.read()
                if not ret:
                    fail["reason"] = "discard_failed"
                    return fail

            frame_b, _ = self._wait_for_valid_frame(cap, CAMERA_SECOND_FRAME_TIMEOUT)
            if frame_b is None:
                fail["reason"] = "frame_b_failed"
                return fail
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_b)
            faces = cascade.detectMultiScale(
                enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            face_count = len(faces)
            motion_score = float(cv2.absdiff(gray_a, gray_b).mean())
            raw = bool(face_count > 0 or motion_score > 3.0)
            return {
                "raw": raw,
                "reason": "ok",
                "backend": backend,
                "startup_ms": startup_ms,
                "face_count": face_count,
                "motion_score": motion_score,
            }
        finally:
            cap.release()

    def detect(self):
        """Returns dict with smoothed result and detection diagnostics."""
        last_result = {
            "raw": None,
            "reason": "unknown",
            "backend": "unknown",
            "startup_ms": 0.0,
            "face_count": 0,
            "motion_score": 0.0,
        }
        for attempt in range(1, DETECT_MAX_ATTEMPTS + 1):
            try:
                result = self._detect_once()
            except Exception:
                logging.exception("detect() attempt failed with exception")
                result = {
                    "raw": None,
                    "reason": "exception",
                    "backend": "unknown",
                    "startup_ms": 0.0,
                    "face_count": 0,
                    "motion_score": 0.0,
                }

            if result["raw"] is not None:
                smoothed = self._smooth(result["raw"])
                logging.info(
                    "detect success camera_index=%d backend=%s attempt=%d/%d "
                    "startup_ms=%.1f face_count=%d motion_score=%.1f raw=%s smoothed=%s",
                    CAMERA_INDEX,
                    result["backend"],
                    attempt,
                    DETECT_MAX_ATTEMPTS,
                    result["startup_ms"],
                    result["face_count"],
                    result["motion_score"],
                    result["raw"],
                    smoothed,
                )
                result["smoothed"] = smoothed
                result["attempt"] = attempt
                result["history"] = list(self._history)
                return result

            last_result = result
            logging.warning(
                "detect failure camera_index=%d attempt=%d/%d reason=%s backend=%s startup_ms=%.1f",
                CAMERA_INDEX,
                attempt,
                DETECT_MAX_ATTEMPTS,
                result["reason"],
                result["backend"],
                result["startup_ms"],
            )
            if attempt < DETECT_MAX_ATTEMPTS:
                time.sleep(DETECT_RETRY_BACKOFF)

        logging.warning(
            "detect retry exhausted camera_index=%d reason=%s startup_ms=%.1f",
            CAMERA_INDEX,
            last_result["reason"],
            last_result["startup_ms"],
        )
        last_result["reason"] = f"retry_exhausted:{last_result['reason']}"
        last_result["smoothed"] = None
        last_result["attempt"] = DETECT_MAX_ATTEMPTS
        last_result["history"] = list(self._history)
        return last_result


class DataStore:
    """Persist presence data as JSON. Only today's counters in memory."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._data = self._load()
        self._last_flush = time.time()

    def _load(self):
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                # Backup corrupt file, start fresh
                backup = DATA_FILE.with_suffix(".json.bak")
                DATA_FILE.rename(backup)
        return {"days": {}, "streak": 0, "best_streak": 0}

    def _flush(self, force=False):
        now = time.time()
        if not force and (now - self._last_flush) < FLUSH_INTERVAL:
            return
        self._last_flush = now
        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, DATA_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _today_key(self):
        return date.today().isoformat()

    def _ensure_today(self):
        key = self._today_key()
        if key not in self._data["days"]:
            self._data["days"][key] = {
                "total_seconds": 0,
                "sessions": [],
                "hourly": {},
            }
        return self._data["days"][key]

    def record_present(self):
        """Record a present sample. Returns updated today stats."""
        today = self._ensure_today()
        now = datetime.now()
        now_ts = now.isoformat(timespec="seconds")
        hour_key = str(now.hour)

        # Update hourly
        today["hourly"][hour_key] = today["hourly"].get(hour_key, 0) + DETECT_INTERVAL

        # Update sessions
        sessions = today["sessions"]
        if sessions:
            last = sessions[-1]
            last_end = datetime.fromisoformat(last["end"])
            gap = (now - last_end).total_seconds()
            if gap < SESSION_GAP:
                # Continue current session
                last["end"] = now_ts
            else:
                # New session
                sessions.append({"start": now_ts, "end": now_ts})
        else:
            sessions.append({"start": now_ts, "end": now_ts})

        today["total_seconds"] = self._compute_total(sessions)
        self._update_streak()
        self._flush()
        return today

    @staticmethod
    def _compute_total(sessions):
        """Derive total seconds from session spans (source of truth)."""
        total = 0
        for s in sessions:
            start = datetime.fromisoformat(s["start"])
            end = datetime.fromisoformat(s["end"])
            total += (end - start).total_seconds()
        return int(total)

    def _update_streak(self):
        """Update streak based on consecutive days with >0 presence."""
        today = date.today()
        streak = 0
        d = today
        while True:
            key = d.isoformat()
            day_data = self._data["days"].get(key, {})
            if day_data.get("total_seconds", 0) > 0:
                streak += 1
                d -= timedelta(days=1)
            else:
                break
        self._data["streak"] = streak
        self._data["best_streak"] = max(streak, self._data.get("best_streak", 0))

    def get_today(self):
        return self._ensure_today()

    def get_week_total(self):
        """Total seconds for the current week (Mon-Sun)."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        total = 0
        for i in range(7):
            key = (monday + timedelta(days=i)).isoformat()
            day_data = self._data["days"].get(key, {})
            total += day_data.get("total_seconds", 0)
        return total

    def get_streak(self):
        return self._data.get("streak", 0)

    def get_best_streak(self):
        return self._data.get("best_streak", 0)

    def save(self):
        self._flush(force=True)


def fmt_duration(seconds):
    """Format seconds as 'Xh Ym'."""
    seconds = max(0, int(seconds))
    h, m = divmod(seconds // 60, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def progress_bar(current, goal, width=32):
    """ASCII progress bar."""
    ratio = min(current / goal, 1.0) if goal > 0 else 0
    filled = int(width * ratio)
    return "=" * filled + "-" * (width - filled)


class OverwatchApp(rumps.App):
    def __init__(self):
        super().__init__("Overwatch", title="0m", quit_button=None)
        self.detector = Detector()
        self.store = DataStore()
        logging.basicConfig(
            filename=str(DATA_DIR / "overwatch.log"),
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        self.paused = False
        self._debug_mode = False
        self._status = None  # True=present, False=away, None=unknown
        self._raw_status = None  # detector output before grace handling
        self._status_detail = None  # optional status suffix for dropdown
        self._next_check = time.time()  # immediate first detection
        self._last_reliable_status = None
        self._camera_failures = 0
        self._det_lock = threading.Lock()
        self._det_in_flight = False
        self._pending_result = None
        self._last_debug_info = None  # last detection result dict
        self._build_menu()

        # Set icon if available
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            self.icon = str(icon_path)

        # 1s timer keeps UI responsive; actual detection still happens every DETECT_INTERVAL
        self.timer = rumps.Timer(self._tick, UI_TICK_INTERVAL)
        self.timer.start()

        # Update menu every tick
        self._update_title()

    def _build_menu(self):
        self.menu = [
            rumps.MenuItem("today_stats", callback=None),
            rumps.MenuItem("progress", callback=None),
            None,  # separator
            rumps.MenuItem("status_line", callback=None),
            rumps.MenuItem("session_line", callback=None),
            None,
            rumps.MenuItem("week_line", callback=None),
            rumps.MenuItem("streak_line", callback=None),
            None,
            rumps.MenuItem("sessions_header", callback=None),
            None,
            rumps.MenuItem("next_check", callback=None),
            None,
            rumps.MenuItem("Run detection now", callback=self._run_now),
            rumps.MenuItem("Debug mode", callback=self._toggle_debug),
            rumps.MenuItem("debug_info", callback=None),
            None,
            rumps.MenuItem("Pause tracking", callback=self._toggle_pause),
            None,
            rumps.MenuItem("Quit Overwatch", callback=self._quit),
        ]

    def _tick(self, _timer):
        now = time.time()

        if not self.paused:
            self._consume_pending_detection()
            if now >= self._next_check:
                if self._start_detection():
                    self._next_check = now + DETECT_INTERVAL

        self._update_title()
        self._update_dropdown()

    def _start_detection(self):
        with self._det_lock:
            if self._det_in_flight:
                logging.warning("detection skipped: previous run still in flight")
                return False
            self._det_in_flight = True

        def worker():
            result = self.detector.detect()
            with self._det_lock:
                self._pending_result = result
                self._det_in_flight = False

        t = threading.Thread(target=worker, name="overwatch-detect", daemon=True)
        t.start()
        return True

    def _consume_pending_detection(self):
        with self._det_lock:
            result = self._pending_result
            self._pending_result = None

        if result is None:
            return

        self._last_debug_info = result
        raw_status = result["smoothed"]
        self._raw_status = result["raw"]
        self._status_detail = None

        if raw_status is None:
            self._camera_failures += 1
            logging.warning(
                "camera unavailable failures=%d reason=%s startup_ms=%.1f",
                self._camera_failures,
                result["reason"],
                result["startup_ms"],
            )
            if (
                self._last_reliable_status is True
                and self._camera_failures <= CAMERA_GRACE_MISSES
            ):
                self._status = True
                self._status_detail = "camera retrying"
                self.store.record_present()
            else:
                self._status = None
        else:
            self._camera_failures = 0
            self._last_reliable_status = raw_status
            self._status = raw_status
            if raw_status is True:
                self.store.record_present()

    def _update_title(self):
        today = self.store.get_today()
        total = today.get("total_seconds", 0)
        time_str = fmt_duration(total)

        if self.paused:
            self.title = f"{time_str} (paused)"
        elif self._status is None:
            self.title = f"{time_str} (?)"
        elif self._status is False:
            self.title = f"{time_str} (away)"
        else:
            self.title = time_str

    def _update_dropdown(self):
        today = self.store.get_today()
        total = today.get("total_seconds", 0)
        sessions = today.get("sessions", [])

        # Today stats
        if "today_stats" in self.menu:
            self.menu["today_stats"].title = (
                f"Today: {fmt_duration(total)} / {fmt_duration(DAILY_GOAL)} goal"
            )

        # Progress bar
        if "progress" in self.menu:
            self.menu["progress"].title = f"[{progress_bar(total, DAILY_GOAL)}]"

        # Status
        if "status_line" in self.menu:
            if self.paused:
                status = "Paused"
            elif self._status is True:
                status = "At desk"
                if self._status_detail:
                    status = f"{status} ({self._status_detail})"
            elif self._status is False:
                status = "Away"
            else:
                status = "Camera unavailable"
            self.menu["status_line"].title = f"Status: {status}"

        # Current session duration (only show if currently present)
        if "session_line" in self.menu:
            if sessions and self._status is True:
                last = sessions[-1]
                start = datetime.fromisoformat(last["start"])
                dur = (datetime.now() - start).total_seconds()
                self.menu["session_line"].title = (
                    f"Current session: {fmt_duration(dur)}"
                )
            else:
                self.menu["session_line"].title = "Current session: --"

        # Week total
        if "week_line" in self.menu:
            week = self.store.get_week_total()
            self.menu["week_line"].title = f"This week: {fmt_duration(week)}"

        # Streak
        if "streak_line" in self.menu:
            streak = self.store.get_streak()
            best = self.store.get_best_streak()
            self.menu["streak_line"].title = f"Streak: {streak} days (best: {best})"

        # Sessions list
        if "sessions_header" in self.menu:
            if sessions:
                lines = ["Sessions today:"]
                for s in sessions:
                    start = datetime.fromisoformat(s["start"])
                    end = datetime.fromisoformat(s["end"])
                    is_active = s == sessions[-1] and self._status
                    dur = (
                        (datetime.now() - start).total_seconds()
                        if is_active
                        else (end - start).total_seconds()
                    )
                    end_label = "now" if is_active else end.strftime("%H:%M")
                    lines.append(
                        f"  {start.strftime('%H:%M')} - {end_label}  ({fmt_duration(dur)})"
                    )
                self.menu["sessions_header"].title = "\n".join(lines)
            else:
                self.menu["sessions_header"].title = "No sessions yet"

        # Next check countdown
        if "next_check" in self.menu:
            if self._next_check:
                remaining = max(0, int(self._next_check - time.time()))
                self.menu["next_check"].title = f"Next check in: {remaining}s"
            else:
                self.menu["next_check"].title = "Next check in: --"

        # Debug info
        if "debug_info" in self.menu:
            if self._debug_mode and self._last_debug_info:
                d = self._last_debug_info
                raw_label = (
                    "Present"
                    if d["raw"]
                    else ("Away" if d["raw"] is False else "Failed")
                )
                hist = [("T" if h else "F") for h in d.get("history", [])]
                smoothed_label = d.get("smoothed")
                self.menu["debug_info"].title = (
                    f"Camera: idx={CAMERA_INDEX} backend={d['backend']}\n"
                    f"Faces: {d['face_count']}  Motion: {d['motion_score']:.1f}  Raw: {raw_label}\n"
                    f"Attempt: {d.get('attempt', '?')}/{DETECT_MAX_ATTEMPTS}  startup={d['startup_ms']:.0f}ms\n"
                    f"History: [{','.join(hist)}] -> {smoothed_label}"
                )
            else:
                self.menu["debug_info"].title = ""

    def _run_now(self, _sender):
        self._next_check = time.time()

    def _toggle_debug(self, sender):
        self._debug_mode = not self._debug_mode
        sender.state = self._debug_mode
        if self._debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.WARNING)
            if "debug_info" in self.menu:
                self.menu["debug_info"].title = ""

    def _toggle_pause(self, sender):
        self.paused = not self.paused
        sender.title = "Resume tracking" if self.paused else "Pause tracking"
        if not self.paused:
            self._next_check = time.time()
        self._update_title()
        self._update_dropdown()

    def _quit(self, _sender):
        self.store.save()
        rumps.quit_application()


if __name__ == "__main__":
    OverwatchApp().run()
