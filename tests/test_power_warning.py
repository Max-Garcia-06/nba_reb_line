"""Battery warning text: on battery the Mac sleeps with the lid closed and scans stop."""
import power_warning as pw

AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=23199843)\t100%; charged; 0:00 remaining present: true
"""
BATT = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=23199843)\t75%; discharging; 10:02 remaining present: true
"""


def _run(tmp_path, out, now, sent):
    return pw.check(now=now, pmset_out=out, state=tmp_path / ".battery_warned", send=sent.append)


def test_parse_power_source():
    assert pw.read_power(AC) == (False, 100)
    assert pw.read_power(BATT) == (True, 75)
    assert pw.read_power("") == (False, None)  # not a Mac / pmset failed: stay quiet


def test_on_battery_texts_with_percent(tmp_path):
    sent = []
    assert _run(tmp_path, BATT, 1_000, sent) is True
    assert len(sent) == 1 and "on battery (75%)" in sent[0] and "Plug in" in sent[0]


def test_repeat_texts_are_throttled_to_once_an_hour(tmp_path):
    sent = []
    _run(tmp_path, BATT, 1_000, sent)
    assert _run(tmp_path, BATT, 1_000 + 59 * 60, sent) is False
    assert _run(tmp_path, BATT, 1_000 + 61 * 60, sent) is True
    assert len(sent) == 2


def test_plugging_in_resets_the_throttle(tmp_path):
    sent = []
    _run(tmp_path, BATT, 1_000, sent)
    assert _run(tmp_path, AC, 1_100, sent) is False
    assert not (tmp_path / ".battery_warned").exists()
    assert _run(tmp_path, BATT, 1_200, sent) is True  # unplugged again: warn right away
    assert len(sent) == 2


def test_on_ac_never_texts(tmp_path):
    sent = []
    assert _run(tmp_path, AC, 1_000, sent) is False
    assert sent == []
