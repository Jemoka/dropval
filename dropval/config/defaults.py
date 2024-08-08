from pathlib import Path
from dropval.utils import load_base

DEFAULTS = {
    "bmask": {
        "lr": 0.01,
        "epochs": 32,
        "batch_size": 16,
        "val_split": 0.1
    },
    "mend": {
        "lr": 0.00005,
        "epochs": 1,
        "batch_size": 12,
        "val_split": 0.05
    },
    "squad": {
        "lr": 0.00001,
        "epochs": 2,
        "batch_size": 12,
        "val_split": 0.1
    },
    "reft": {
        "lr": 0.0005,
        "epochs": 1,
        "batch_size": 10,
        "val_split": 0.1
    },
    "consistency": {
        "lr": None,
        "epochs": None,
        "batch_size": 16,
        "val_split": 0.1
    }
}

def load(args):
    # make output dir
    actual_out_dir = (Path(args.out_dir) / args.experiment)
    actual_out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(actual_out_dir)

    model, tokenizer, config = load_base(args)
    args.__dict__["model_config"] = config

    if DEFAULTS.get(args.task):
        # hydrate any shared argumensts
        args.lr = args.lr if args.lr else DEFAULTS[args.task]["lr"]
        args.batch_size = args.batch_size if args.batch_size else DEFAULTS[args.task]["batch_size"]
        args.epochs = args.epochs if args.epochs else DEFAULTS[args.task]["epochs"]
        args.val_split = args.val_split if args.val_split else DEFAULTS[args.task]["val_split"]

    return args, model, tokenizer




