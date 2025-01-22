from ncps.mini_keras.backend.common import global_state
from ncps.mini_keras.testing import test_case
from ncps.mini_keras.utils.naming import auto_name


class GlobalStateTest(test_case.TestCase):
    def test_clear_session(self):
        name0 = auto_name("somename")
        self.assertEqual(name0, "somename")
        name1 = auto_name("somename")
        self.assertEqual(name1, "somename_1")
        global_state.clear_session()
        name0 = auto_name("somename")
        self.assertEqual(name0, "somename")
