from dropval.trainers import MENDTrainer, BMaskTrainer, SquadTrainer
from dropval.measurements import BMask, Consistency, KN
from dropval.utils import get_accelerator

from pathlib import Path

from accelerate.logging import get_logger
L = get_logger("dropval", log_level="DEBUG")


def dispatch_bmask_(args, accelerator, model, tokenizer):
    # for each concept, if it isn't prepared already, prepare it
    concepts = BMaskTrainer.concepts(args)
    for indx, concept in enumerate(concepts):
        L.info(f"CONCEPT | BMASK | {concept} | {indx} / {len(concepts)}")
        if (Path(args.out_dir) / args.results_dir / "bmask"  / f"bmask_{concept}.json").exists():
            L.info(f"CONCEPT | BMASK | {concept} | SKIPPING")
            continue

        trainer = BMaskTrainer(args, accelerator, model, tokenizer, concept)

        for i in range(args.epochs):
            L.info(f"EPOCH | BMASK | {i} / {args.epochs}")
            trainer.epoch(i)

        evaluator = BMask(args, accelerator, trainer, concept)
        evaluator()

def dispatch_mend_(args, accelerator, model, tokenizer):
    if (Path(args.out_dir) / args.results_dir / "mend.json").exists():
        L.info("MEND DONE... SKIPPING")
        return

    trainer = MENDTrainer(args, accelerator, model, tokenizer)
    for i in range(args.epochs):
        L.info(f"EPOCH | MEND | {i} / {args.epochs}")
        trainer.epoch(i)
    trainer.finish()

def dispatch_squad_(args, accelerator, model, tokenizer):
    if (Path(args.out_dir) / args.results_dir / "squad.json").exists():
        L.info("SQUAD DONE... SKIPPING")
        return

    trainer = SquadTrainer(args, accelerator, model, tokenizer)
    for i in range(args.epochs):
        L.info(f"EPOCH | SQUAD | {i} / {args.epochs}")
        trainer.epoch(i)
    trainer.finish()

def dispatch_consistency_(args, accelerator, model, tokenizer):
    evaluator = Consistency(args, accelerator, model, tokenizer)
    evaluator()

def dispatch_kns_(args, accelerator, model, tokenizer):
    evaluator = KN(args, accelerator, model, tokenizer)
    evaluator()

def execute(args, model, tokenizer):
    accelerator = get_accelerator(args)
    if args.task.lower() == "bmask":
        dispatch_bmask_(args, accelerator, model, tokenizer)
    elif args.task.lower() == "mend":
        dispatch_mend_(args, accelerator, model, tokenizer)
    elif args.task.lower() == "squad":
        dispatch_squad_(args, accelerator, model, tokenizer)
    elif args.task.lower() == "consistency":
        dispatch_consistency_(args, accelerator, model, tokenizer)
    elif args.task.lower() == "kn":
        dispatch_kns_(args, accelerator, model, tokenizer)






