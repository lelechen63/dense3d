# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=500)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        # default="/mnt/disk1/dat/lchen63/grid/data/pickle/")
                        default = '/media/lele/DATA/spie')
    parser.add_argument("--model_dir",
                        type=str,
                        # default="/mnt/disk1/dat/lchen63/grid/model/")
                        default='/media/lele/DATA/spie/model')
    
    parser.add_argument("--log_dir",
                        type=str,
                        default="/media/lele/DATA/spie/log/")
    parser.add_argument("--model_number",
                        type=str,
                        default='baseline')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=12)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str) 
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()


if __name__ == "__main__":
    config = parse_args()
    config.is_train = True
    
    if config.model_number == 'baseline':
        import baseline_trainer as trainer
    
    else:
        print 'wrong model number!!!!!!!!!!!!!!!!!!!!'
    main(config)
