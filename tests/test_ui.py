"""Exercise Streamlit page rendering and real prediction without browser emulation."""
import os,unittest
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
class Interface(unittest.TestCase):
    def test_all_pages_and_manual_prediction(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=45).run()
        self.assertEqual(len(a.exception),0,[e.message for e in a.exception])
        for page in ['Explore data','Model evidence','Decision rules','Talk to data','Risk assessment']:
            a.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(a.exception),0,[e.message for e in a.exception])
        a.button[0].click().run(timeout=60)
        self.assertEqual(len(a.exception),0,[e.message for e in a.exception])
        self.assertIn('Estimated difficulty probability',[v.label for v in a.metric])
if __name__=='__main__':unittest.main(verbosity=2)
