"""CRV config admission is identical for plain and streaming kits, before any job submission."""
import json
import shlex
import subprocess

import pytest

from mu2efastsim.endpoints.kit import FastsimKit
from test_kit_parameters import Endpoint, refused

RC = pytest.importorskip('crvfastsim.response_config')
from crvfastsim import kit_uva as KU


@pytest.mark.parametrize('config', [
    '/tmp/settings.json', None, [], True, '{"schema":1,"schema":2}',
    {'schema': 1, 'sipm': {'thermal_rate': 1}}, {'schema': 1, 'sipm': {'p1': 'nan'}},
    {'schema': 1, 'assets': {'calib': '$(INERT_SENTINEL)'}},
    {'schema': 1, 'waveform': {'single_pe_file': '${INERT_SENTINEL}'}},
])
def test_invalid_config_refused_before_plain_or_streaming_submit(tmp_path, config):
    kit = FastsimKit(endpoint=Endpoint(), state_dir=tmp_path)
    params = dict(task='crv_response', seed=1, response_config=config)
    refused(kit, params)
    with pytest.raises(ValueError) as exc:
        KU.validate(params)
    assert str(exc.value) == kit.check('point', params, [], [])['message']


def test_complete_kit_config_is_one_shell_argument(tmp_path):
    kit = FastsimKit(endpoint=Endpoint(), state_dir=tmp_path)
    params = dict(task='crv_response', seed=1, response_config={'schema': 1, 'sipm': {'overvoltage': 2.5}},
                  time_constant=14)
    assert kit.check('point', params, [], [])['ok']
    body, _ = kit.render(params)
    line = next(x for x in body.splitlines() if '--response-config' in x)
    words = shlex.split(line.rstrip('\\'))
    config = words[words.index('--response-config') + 1]
    assert RC.parse(config)['sipm']['overvoltage'] == 2.5
    assert KU.validate(params)['response_config'] == config
    subprocess.run(['bash', '-n'], input=body, text=True, check=True)


def test_effective_config_and_knobs_joint_validation(tmp_path):
    kit = FastsimKit(endpoint=Endpoint(), state_dir=tmp_path)
    params = dict(task='crv_response', seed=1, response_config={'schema': 1, 'sipm': {'p1': .0001,
                                                                                  'cross_talk': 0}}, cross_talk=.9)
    refused(kit, params)


def test_config_schema_is_not_registered_as_new_numeric_knobs():
    task = FastsimKit.describe()['tasks']['crv_response']
    assert task['params']['response_config']['role'] == 'fixed_configuration'
    assert 'overvoltage' not in task['params']
    assert 'p1' not in task['params']
