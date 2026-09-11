"""Run PECL using data generated previously by haory_util/homo_g.py."""
import os
from pathlib import Path
import time


def main():
    # Existing project modules resolve some paths against the working directory.
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    from util.conf import ModelConf

    conf = ModelConf(str(script_dir / 'conf' / 'PECL.yaml'))
    processed_file = (
        script_dir / 'data' / 'preprocessed'
        / '{}_dgl_all'.format(conf['dataset']) / 'homo_dataset.pkl'
    )
    if not processed_file.is_file():
        raise SystemExit(
            'Preprocessed data not found: {}\n'
            'Run `python haory_util/homo_g.py` first, then run `python main.py`.'
            .format(processed_file)
        )

    for key in ('training.set', 'test.set'):
        if not Path(conf[key]).is_file():
            raise SystemExit('Model input file not found ({}): {}'.format(key, conf[key]))

    # Import only after checking inputs: importing run_ml loads homo_dataset.pkl.
    from SELFRec import SELFRec

    print('Using preprocessed data: {}'.format(processed_file))
    start = time.time()
    rec = SELFRec(conf)
    rec.execute()
    print('Running time: {:.2f} s'.format(time.time() - start))


if __name__ == '__main__':
    main()
