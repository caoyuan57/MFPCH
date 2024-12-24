from my_opt import *
import argparse
import torch
from utils import logger
import time
import numpy as np

def param_list(log, config):
    log.info('>>> Configs List <<<')
    log.info('--- Dadaset:{}'.format(config.DATASET))
    log.info('--- SEED:{}'.format(config.SEED))
    log.info('--- Bit:{}'.format(config.HASH_BIT))
    log.info('--- Batch:{}'.format(config.BATCH_SIZE))
    log.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    log.info('--- Lr_TXT:{}'.format(config.LR_TXT))
    log.info('--- LR_MyNet:{}'.format(config.LR_MyNet))

    log.info('--- lambda1:{}'.format(config.lambda1))
    log.info('--- lambda2:{}'.format(config.lambda2))
    log.info('--- beta:{}'.format(config.beta))
    log.info('--- K:{}'.format(config.K))
    log.info('--- a1:{}'.format(config.a1))
    log.info('--- a2:{}'.format(config.a2))

def main(config):
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.cuda.set_device(config.GPU_ID)

    logName = config.DATASET + '_' + str(config.HASH_BIT)
    log = logger(logName)
    param_list(log, config)

    wxz = PCIRH(log, config)
    best_it = best_ti = 0

    if config.TRAIN == True:
        start = time.time()
        for epoch in range(config.NUM_EPOCH):
            coll_B, record_index = wxz.train_method(epoch)
            wxz.train_Hashfunc(coll_B, record_index, epoch)

            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I, re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = wxz.performance_eval()
                log.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))

                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                    np.save(str(config.HASH_BIT) + "-" + "db_img_binary.npy", re_BI)
                    np.save(str(config.HASH_BIT) + "-" + "db_txt_binary.npy", re_BT)
                    np.save(str(config.HASH_BIT) + "-" + "db_label.npy", re_L)
                    np.save(str(config.HASH_BIT) + "-" + "tst_img_binary.npy", qu_BI)
                    np.save(str(config.HASH_BIT) + "-" + "tst_txt_binary.npy", qu_BT)
                    np.save(str(config.HASH_BIT) + "-" + "tst_label.npy", qu_L)
                    trn_time = time.time() - start
                    np.save(str(config.HASH_BIT) + "-" + "trn_time.npy", trn_time)
                    log.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (best_it, best_ti))
                    #wxz.save_checkpoints()

                log.info('--------------------------------------------------------------------')
    else:
        ckp = config.DATASET + '_' + str(config.HASH_BIT)+'bits.pth'
        wxz.load_checkpoints(ckp)
        MAP_I2T, MAP_T2I = wxz.performance_eval()
        log.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ours')
    parser.add_argument('--TRAIN', default=True, help='train or test', type=bool)
    parser.add_argument('--DATASET', default='MIRFlickr', help='MIRFlickr, NUSWIDE or COCO', type=str)

    parser.add_argument('--lambda1', default=10, type=float, help='10')
    parser.add_argument('--lambda2', default=1, type=float, help='1')
    parser.add_argument('--beta', default=0.01, type=float, help='0.01')

    parser.add_argument('--LR_IMG', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_TXT', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_MyNet', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--K', default=3300, help='3300', type=int)
    parser.add_argument('--a1', default=0.5, help='balance ST and SI (0.5)', type=float)
    parser.add_argument('--a2', default=0.5, help='balance S1 and S2 (0.5)',type=float)
    parser.add_argument('--HASH_BIT', default=128, help='code length', type=int)
    parser.add_argument('--BATCH_SIZE', default=512, type=int)
    parser.add_argument('--GPU_ID', default=0, type=int)
    parser.add_argument('--SEED', default=1, type=int)  # Please choose a suitable random seed.
    parser.add_argument('--NUM_WORKERS', default=8, type=int)
    parser.add_argument('--EPOCH_INTERVAL', default=2, type=int)
    parser.add_argument('--NUM_EPOCH', default=2, type=int)
    parser.add_argument('--EVAL_INTERVAL', default=1, type=int)
    parser.add_argument('--MODEL_DIR', default="./checkpoints", type=str)

    config = parser.parse_args()
    main(config)
