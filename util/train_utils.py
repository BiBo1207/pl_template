import os


def last_model_path(root: str, v_num: int):
    return os.path.join('./', root, 'lightning_logs', f'version_{v_num}',
                        'checkpoints', 'last.ckpt')
