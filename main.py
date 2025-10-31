from SELFRec import SELFRec
from util.conf import ModelConf
import time
# from haory_util import homo_g
import subprocess
import os


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "haory_util/homo_g.py")
    print('=' * 80)
    print("Running to preprocess data...")
    subprocess.run(["python", path], check=True)
    print("Finished running\n")
    # model = input('Please enter the model you want to run:')
    model = 'PECL'
    s = time.time()

    conf = ModelConf(f'./conf/{model}.yaml')
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print(f"Running time: {e - s:.2f} s")

