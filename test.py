import numpy as np
import os
from utils.evaluation_utils import rbox2txt, mergebypoly, evaluation_trans

def postprocessing(detoutput):
    result_before_merge = str(detoutput + '/labels')
    result_merged_path  = str(detoutput + '/result_merged')
    result_classname_path= str(detoutput + '/result_classname')

    mergebypoly(result_before_merge, result_merged_path)
    print("results have been merged")
    evaluation_trans(
        result_merged_path,
        result_classname_path
    )
    print("results have been classified by classnames")


if __name__  == '__main__':
    classsnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                   'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                   'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                   'container-crane']

    postprocessing(
        detoutput='/home/yanggang/PyCharmWorkspace/yolov5-obb-detection/runs/detect/exp2'
    )

