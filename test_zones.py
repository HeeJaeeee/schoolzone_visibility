import unittest
import json
import tempfile
from pathlib import Path
from PIL import Image
from photo_analysis import assess, export_result, inside, polygon, signal_state


def car(box):
    return {'kind':'vehicle','box':box,'confidence':.9}


class ZoneTests(unittest.TestCase):
    region = [[.1,.1],[.5,.1],[.5,.5],[.1,.5]]
    waiting = [[.6,.1],[.9,.1],[.9,.5],[.6,.5]]

    def test_ground_point_decides_zone(self):
        # A tall truck whose box covers the zone but whose wheels are below it is not in the zone.
        self.assertEqual(assess([car([.2,.3,.4,.7])],self.region)['candidate_count'],0)
        self.assertEqual(assess([car([.2,.0,.4,.45])],self.region)['candidate_count'],1)

    def test_waiting_lane_wins_on_overlap(self):
        waiting = [[.3,.3],[.6,.3],[.6,.6],[.3,.6]]
        result = assess([car([.35,.3,.45,.45])],self.region,waiting)
        self.assertEqual(result['zones'][0]['zone'],'waiting')
        self.assertEqual(result['candidate_count'],0)

    def test_waiting_only_is_separate(self):
        result = assess([car([.65,.2,.85,.4])],self.region,self.waiting)
        self.assertEqual((result['candidate_count'],result['waiting_count'],result['status']),(0,1,'waiting'))

    def test_concave_polygon(self):
        l_shape = [[0,0],[1,0],[1,.2],[.2,.2],[.2,1],[0,1]]
        self.assertTrue(inside((.1,.9),l_shape))
        self.assertFalse(inside((.6,.6),l_shape))

    def test_rectangle_profiles_still_load(self):
        self.assertEqual(polygon([.1,.2,.3,.4]),[[.1,.2],[.3,.2],[.3,.4],[.1,.4]])

    def test_bad_polygons_rejected(self):
        for zone in ([[0,0],[1,1]],[[0,0],[.5,.5],[1,1]],[[0,0],[1,0],[1,1.2]]):
            with self.assertRaises(ValueError):
                polygon(zone)

    def test_vehicle_cut_off_at_frame_bottom(self):
        self.assertEqual(assess([car([.2,.8,.4,1])],[[0,.5],[1,.5],[1,1],[0,1]])['candidate_count'],1)

    def test_signal_colour_by_hsv(self):
        image = Image.new('RGB',(100,100),(20,20,20))
        image.paste((255,30,30),(10,10,20,20))
        image.paste((0,220,170),(60,10,70,20))
        self.assertEqual(signal_state(image,[.05,.05,.25,.25]),'red')
        self.assertEqual(signal_state(image,[.55,.05,.75,.25]),'green')
        self.assertEqual(signal_state(image,[.3,.5,.6,.9]),'unknown')
        self.assertIsNone(signal_state(image,None))

    def test_export_preserves_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'photo.png'
            Image.new('RGB',(100,100)).save(path)
            detections = [dict(car([.65,.2,.85,.4]),model_class=2)]
            files = export_result(path,self.region,detections,{},Path(folder)/'out',waiting_region=self.waiting)
            saved = json.loads(files[1].read_text(encoding='utf-8'))
            self.assertEqual(saved['analysis'],assess(detections,self.region,self.waiting))


if __name__=='__main__':unittest.main()
