from gym.envs.registration import register

register(
    id='ScannerEnv-v0',
    entry_point='scan_gym.envs.ScannerEnv:ScannerEnv',
)
