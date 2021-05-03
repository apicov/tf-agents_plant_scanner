from gym.envs.registration import register

register(
    id='ScannerEnv-v1',
    entry_point='scan_gym.envs.ScannerEnv:ScannerEnv',
)
