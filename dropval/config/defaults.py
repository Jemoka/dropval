from pathlib import Path

DEFAULTS = {
    "bmask": {
        "lr": 0.01,
        "epochs": 32,
        "batch_size": 16
    },
    "mend": {
        "lr": 0.0001,
        "epochs": 4,
        "batch_size": 16
    },
    "squad": {
        "lr": 0.00001,
        "epochs": 2,
        "batch_size": 12
    }
}

def load(args):
    # make output dir
    actual_out_dir = (Path(args.out_dir) / args.experiment)
    actual_out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = actual_out_dir

    # hydrate data dir
    args.data_dir = Path(args.data_dir)

    model, tokenizer, config = load_base(args)
    args.__dict__["model_config"] = config

    # hydrate any shared argumensts
    args.lr = args.lr if args.lr else DEFAULTS[args.task]["lr"]
    args.batch_size = args.batch_size if args.batch_size else DEFAULTS[args.task]["batch_size"]
    args.epochs = args.epochs if args.epochs else DEFAULTS[args.task]["epochs"]

    return args, model, tokenizer




