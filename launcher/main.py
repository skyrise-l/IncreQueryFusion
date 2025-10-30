
import argparse
from SystemFramework.systemFramework import CoreSystem
from utils.evaluate import Evaluate

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='fusion query')
    parser.add_argument('--dataset', type=str, default="book",help='dataset name')

    parser.add_argument('--mode', type=str, default="system",help='test mode or system mode')

    parser.add_argument("--types", nargs='+', default=['JSON','CSV','XML'])
    parser.add_argument("--thres_for_query", type=float, default=0.95, help="threshold the for query stage")
    parser.add_argument("--thres_for_fusion", type=float, default=0.5, help="threshold for value veracity")
    parser.add_argument("--gpu", type=int, default=1, help="gpu device")
    parser.add_argument("--full", action="store_true", default=False, help="enable full dataset")
    parser.add_argument("--fusion_model", type=str, default="DART",
                        help="select from (FusionQuery, CASE, DART, LTM, TruthFinder, MajorityVoter,MyFusion,None)")
    return parser.parse_args()

def main():
    args = parse_args()
    coreSystem = CoreSystem(args.dataset, args.mode,args.fusion_model, args.full)
    coreSystem.start()

if __name__ == '__main__':
    main()