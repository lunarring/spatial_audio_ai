import pytest
from tools.playback_stream import BlackHoleStereoRelayer

def test_toggle_playback_mode(capsys):
    # Initialize with 'alternating' mode
    relayer = BlackHoleStereoRelayer(mapping_scheme="alternating")
    
    # Toggle: should switch to 'stereo'
    relayer.handle_key_press('t')
    captured = capsys.readouterr().out
    assert "Playback mode toggled from alternating to stereo" in captured
    assert relayer.mapping_scheme == "stereo"
    
    # Toggle again: should switch back to 'alternating'
    relayer.handle_key_press('t')
    captured = capsys.readouterr().out
    assert "Playback mode toggled from stereo to alternating" in captured
    assert relayer.mapping_scheme == "alternating"