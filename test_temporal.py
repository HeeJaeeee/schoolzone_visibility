import unittest
from temporal import StopMonitor


def car(box=None):
    return {"kind":"vehicle","box":box or [.2,.2,.4,.4],"confidence":.9}


ZONE = [[.1,.1],[.5,.1],[.5,.5],[.1,.5]]


class StopMonitorTests(unittest.TestCase):
    def run_seconds(self,seconds,frame=lambda t:[car()],monitor=None,signal=lambda t:None):
        monitor = monitor or StopMonitor(ZONE,alert_seconds=5)
        for t in seconds:
            monitor.update(t,frame(t),signal(t))
        return monitor.finish()

    def test_alert_exactly_at_threshold(self):
        result = self.run_seconds(range(8))
        self.assertEqual(result["alert_count"],1)
        self.assertEqual(result["alerts"][0]["alert_seconds"],5)
        self.assertEqual(result["alerts"][0]["end_seconds"],7)
        self.assertEqual((result["alerts"][0]["alert_dwell_seconds"],result["alerts"][0]["dwell_seconds"]),(5,7))

    def test_short_stop_no_alert(self):
        self.assertEqual(self.run_seconds(range(5))["alert_count"],0)

    def test_short_detection_miss_keeps_timer(self):
        # Night B of v3 split into short pieces on one missed box; a parked car does not leave between samples.
        result = self.run_seconds([0,1,2,3,5,6])
        self.assertEqual(result["alerts"][0]["alert_seconds"],5)
        self.assertEqual(result["alerts"][0]["track"],1)

    def test_long_gap_restarts(self):
        self.assertEqual(self.run_seconds([0,1,2,3,7,8,9])["alert_count"],0)

    def test_moving_car_never_alerts(self):
        result = self.run_seconds(range(12),frame=lambda t:[car([.12+.03*t,.2,.22+.03*t,.3])])
        self.assertEqual(result["alert_count"],0)

    def test_box_jitter_is_still_stopped(self):
        result = self.run_seconds(range(8),frame=lambda t:[car([.2+.005*(t%2),.2,.4+.005*(t%2),.4])])
        self.assertEqual(result["alert_count"],1)

    def test_waiting_lane_never_counts(self):
        monitor = StopMonitor(ZONE,waiting_region=[[.1,.3],[.5,.3],[.5,.5],[.1,.5]],alert_seconds=5)
        self.assertEqual(self.run_seconds(range(10),monitor=monitor)["alert_count"],0)

    def test_red_light_pauses_timer(self):
        # Red until t=3: waiting at the signal, not counted. Counting starts with the green sample at t=4 -> alert at t=8.
        result = self.run_seconds(range(12),signal=lambda t:"red" if t<4 else "green")
        self.assertEqual(result["alerts"][0]["alert_seconds"],8)
        self.assertEqual(result["alerts"][0]["stopped_since_seconds"],0)

    def test_outside_zone_and_people_ignored(self):
        result = self.run_seconds(range(10),frame=lambda t:[car([.7,.7,.9,.9]),{"kind":"person","box":[.2,.2,.3,.4],"confidence":.9}])
        self.assertEqual(result["alert_count"],0)

    def test_two_cars_two_alerts(self):
        result = self.run_seconds(range(7),frame=lambda t:[car([.1,.1,.25,.3]),car([.3,.2,.45,.45])])
        self.assertEqual(sorted(a["track"] for a in result["alerts"]),[1,2])

    def test_partial_duplicate_box_does_not_alert_twice(self):
        # Night B at 5 s: a second, partial box on the already-alerting dump truck raised a second alert.
        full,part = [.1,.1,.45,.45],[.25,.15,.45,.45]
        result = self.run_seconds(range(16),frame=lambda t:[car(full)]+([car(part)] if t>=6 else []))
        self.assertEqual(result["alert_count"],1)

    def test_time_must_increase(self):
        monitor = StopMonitor(ZONE)
        monitor.update(1,[car()])
        with self.assertRaises(ValueError):
            monitor.update(1,[car()])
        self.assertEqual(monitor.finish(),monitor.finish())

    def test_bad_settings_rejected(self):
        with self.assertRaises(ValueError):
            StopMonitor(None)
        with self.assertRaises(ValueError):
            StopMonitor(ZONE,alert_seconds=0)


if __name__ == "__main__":
    unittest.main()
