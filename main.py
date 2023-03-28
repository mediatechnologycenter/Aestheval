
import argparse

def config_parser():
    parser = argparse.ArgumentParser(description="Downloader script")   
    # dataset options
    parser.add_argument("--data_path", type=str, help="Data directory to download the datasets", default="data/")
    parser.add_argument("--download_data", action='store_true', help="Whether to download data", default=False)
    parser.add_argument("--only_images", action='store_true', help="Whether to download only the images or not (only available for RPCD dataset).")
    parser.add_argument("--compute_sentiment_score", action='store_true', help="Whether to compute the sentiment score", default=False)
    parser.add_argument("--compute_informativeness_score", action='store_true', help="Whether to compute the informativeness score", default=False)
    parser.add_argument("--run_baseline", action='store_true', help="Whether to compute the sentiment score", default=False)
    parser.add_argument("--run_probbing", action='store_true', help="Whether to compute the sentiment score", default=False)
    parser.add_argument("--dataset_name", required=True, type=str, choices=['PCCD', 'Reddit', 'AVA', 'all'])
    parser.add_argument("--evaluate", action='store_true', help="Whether to run only evaluation")
    parser.add_argument("--model_name", type=str, help="nima when running baseline, vit version when running probbing experiments")
    parser.add_argument("--scoring", type=str, choices=['original', 'sentiment'], help="Types of scoring to use", default="original")
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    if args.download_data:
        from aestheval.data.datautils.data_downloader import download_datasets
        download_datasets(data_path=args.data_path, dataset=args.dataset_name, args=args)
    
    if args.compute_sentiment_score:
        from aestheval.text_prediction.sentiment_score import *
        if args.dataset_name == "PCCD":
            sentiment_pccd(root_dir=args.data_path)
        if args.dataset_name == "Reddit":
            sentiment_reddit(root_dir=args.data_path)
        if args.dataset_name == "AVA":
            sentiment_ava(root_dir=args.data_path)

    if args.compute_informativeness_score:
        from aestheval.text_prediction.compute_info_score import *
        if args.dataset_name == "PCCD":
            info_pccd(root_dir=args.data_path)
        if args.dataset_name == "Reddit":
            info_reddit(root_dir=args.data_path)
        if args.dataset_name == "AVA":
            info_ava(root_dir=args.data_path)

    if args.run_baseline:
        from aestheval.baselines.run_baseline import run
        run(args.dataset_name,  args.model_name, args.data_path, args.evaluate)
    
    if args.run_probbing:
        from aestheval.baselines.probbing import run as run_probbing
        run_probbing(args.dataset_name, args.model_name, args.data_path, args.scoring)
